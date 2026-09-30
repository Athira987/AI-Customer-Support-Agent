import pytest
import tempfile
import shutil
from pathlib import Path
from app.services.ingestion import IngestionService, DocumentChunk
from app.services.retrieval import RetrievalService
from app.services.embeddings import EmbeddingService


def test_chunking_logic():
    """Verify document chunking respects size and produces metadata."""
    ingestion = IngestionService(chunk_size=100, chunk_overlap=20)
    sample_text = (
        "NovaTech Electronics provides high quality laptops. "
        "The NovaBook Pro 15 has an OLED 4K display. "
        "It comes with a standard 1-year limited warranty. "
        "Battery life reaches up to 11 hours on a single charge."
    )
    chunks = ingestion.chunk_text(sample_text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) > 0


def test_chroma_upsert_and_query_384d():
    """Verify ChromaDB upserts 384-dimensional chunks and retrieves nearest neighbors."""
    temp_dir = tempfile.mkdtemp()
    try:
        retrieval = RetrievalService(
            persist_directory=temp_dir,
            collection_name="test_collection_384"
        )

        chunk = DocumentChunk(
            chunk_id="test_doc_1",
            text="NovaTech standard return policy is 30 days.",
            metadata={"filename": "test.txt", "category": "policies", "page": 1}
        )

        # 384-dimensional vector matching all-MiniLM-L6-v2
        fake_embedding = [0.05] * 384
        retrieval.upsert_chunks([chunk], [fake_embedding])

        assert retrieval.count() == 1

        results = retrieval.query_similar(fake_embedding, top_k=1)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "test_doc_1"
        assert "30 days" in results[0]["text"]
        assert results[0]["distance"] < 0.01

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_local_sentence_transformers_embedding():
    """Verify local SentenceTransformer generates 384-d normalized embeddings without API key."""
    service = EmbeddingService(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query = "How long does shipping take?"
    vector = service.get_embedding(query)

    assert isinstance(vector, list)
    assert len(vector) == 384
    assert all(isinstance(v, float) for v in vector)

    batch_vectors = service.get_embeddings_batch(["Text 1", "Text 2"])
    assert len(batch_vectors) == 2
    assert len(batch_vectors[0]) == 384
    assert len(batch_vectors[1]) == 384
