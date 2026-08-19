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
            # Ensure storage directory exists
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
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks provided but {len(embeddings)} embeddings given.")

        collection = self._get_collection()

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Batch upsert to prevent payload limits
        batch_size = 100
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
        k = top_k or settings.top_k
        collection = self._get_collection()

        total_count = collection.count()
        if total_count == 0:
            logger.warning("Chroma collection is currently empty. Ingestion required.")
            return []

        # Adjust k if collection has fewer items than requested
        k_actual = min(k, total_count)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k_actual,
            include=["documents", "metadatas", "distances"]
        )

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
        client.delete_collection(name=self.collection_name)
        self._collection = None
        logger.info(f"Collection '{self.collection_name}' has been reset.")
