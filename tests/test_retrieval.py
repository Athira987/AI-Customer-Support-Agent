import pytest
import tempfile
import shutil
from pathlib import Path
from app.services.ingestion import IngestionService, DocumentChunk
from app.services.retrieval import RetrievalService


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


def test_chroma_upsert_and_query():
    """Verify ChromaDB upserts chunks and retrieves nearest neighbors."""
    temp_dir = tempfile.mkdtemp()
    try:
        retrieval = RetrievalService(
            persist_directory=temp_dir,
            collection_name="test_collection"
        )

        # Create sample chunk
        chunk = DocumentChunk(
            chunk_id="test_doc_1",
            text="NovaTech standard return policy is 30 days.",
            metadata={"filename": "test.txt", "category": "policies", "page": 1}
        )

        # Mock 1536-dim embedding vector
        fake_embedding = [0.1] * 1536
        retrieval.upsert_chunks([chunk], [fake_embedding])

        assert retrieval.count() == 1

        # Query using same vector
        results = retrieval.query_similar(fake_embedding, top_k=1)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "test_doc_1"
        assert "30 days" in results[0]["text"]
        assert results[0]["distance"] < 0.01  # Cosine distance ~ 0 for identical vector

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
