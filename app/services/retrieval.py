import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings
from app.services.ingestion import DocumentChunk

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service to interact with the ChromaDB vector database."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.collection_name = collection_name or settings.chroma_collection_name
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None

    def _get_client(self) -> chromadb.PersistentClient:
        """Initialize and return persistent Chroma client."""
        if self._client is None:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        return self._client

    def _get_collection(self):
        """Retrieve or create Chroma collection configured for cosine similarity."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def upsert_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> int:
        """
        Upsert document chunks and their precomputed embeddings into ChromaDB.
        Idempotent: updates existing chunk IDs or inserts new ones.
        Safely handles dimension changes by recreating collection if needed.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks provided but {len(embeddings)} embeddings given.")

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        collection = self._get_collection()

        batch_size = 100
        try:
            for i in range(0, len(ids), batch_size):
                collection.upsert(
                    ids=ids[i : i + batch_size],
                    documents=documents[i : i + batch_size],
                    metadatas=metadatas[i : i + batch_size],
                    embeddings=embeddings[i : i + batch_size]
                )
        except Exception as e:
            # If embedding dimension mismatch occurs against an old collection, recreate it
            logger.warning(f"Error during upsert ({e}). Recreating collection with new embedding dimension...")
            self.reset_collection()
            collection = self._get_collection()
            for i in range(0, len(ids), batch_size):
                collection.upsert(
                    ids=ids[i : i + batch_size],
                    documents=documents[i : i + batch_size],
                    metadatas=metadatas[i : i + batch_size],
                    embeddings=embeddings[i : i + batch_size]
                )

        logger.info(f"Successfully upserted {len(chunks)} chunks into collection '{self.collection_name}'.")
        return len(chunks)

    def query_similar(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Query ChromaDB using a query embedding and return top-k nearest chunks with metadata and distances.
        """
        return self.query_similar_with_filter(
            query_embedding=query_embedding,
            top_k=top_k,
            filename_filter=None
        )

    def query_similar_with_filter(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        filename_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query ChromaDB with an optional filename metadata filter for category-aware retrieval.

        When ``filename_filter`` is provided, only chunks whose ``filename``
        metadata field matches that value are considered, restricting retrieval
        to a single knowledge-base source document.  This implements
        category-aware RAG without maintaining separate collections.

        Args:
            query_embedding: The query vector to search with.
            top_k: Maximum number of results to return.
            filename_filter: Optional exact filename to restrict retrieval to
                (e.g. 'warranty_policy.txt').  Pass ``None`` for unrestricted search.

        Returns:
            List of result dicts with keys: chunk_id, text, metadata, distance.
        """
        k = top_k or settings.top_k
        collection = self._get_collection()

        total_count = collection.count()
        if total_count == 0:
            logger.warning("Chroma collection is currently empty. Ingestion required.")
            return []

        # Adjust k if collection has fewer items than requested
        k_actual = min(k, total_count)

        # Build optional where clause for category filtering
        where_clause = None
        if filename_filter:
            where_clause = {"filename": {"$eq": filename_filter}}
            logger.info(f"[CATEGORY FILTER] Restricting retrieval to filename='{filename_filter}'")

        try:
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": k_actual,
                "include": ["documents", "metadatas", "distances"],
            }
            if where_clause:
                query_kwargs["where"] = where_clause

            results = collection.query(**query_kwargs)
        except Exception as e:
            # If filter returns empty (ChromaDB raises on 0-result where clauses in some versions),
            # fall back gracefully with an empty result set.
            logger.warning(
                f"[CATEGORY FILTER] Filtered query failed ({e}). "
                "The selected category may have no indexed chunks. Returning empty results."
            )
            return []

        matched_items: List[Dict[str, Any]] = []

        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

            for doc, meta, dist, cid in zip(docs, metas, dists, ids):
                matched_items.append({
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": meta,
                    "distance": float(dist)  # Cosine distance: 0 = identical, >0.65 = dissimilar
                })

        return matched_items

    def count(self) -> int:
        """Return total number of chunks stored in the collection."""
        collection = self._get_collection()
        return collection.count()

    def reset_collection(self) -> None:
        """Reset/clear all items in the current collection."""
        client = self._get_client()
        try:
            client.delete_collection(name=self.collection_name)
        except Exception as e:
            logger.debug(f"Collection deletion notice: {e}")
        self._collection = None
        logger.info(f"Collection '{self.collection_name}' has been reset.")
