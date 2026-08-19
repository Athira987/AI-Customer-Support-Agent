"""
Document Ingestion Script for NovaTech Knowledge Base.
Loads all .txt and .pdf documents, chunks text, generates embeddings, and indexes into ChromaDB.

Usage:
    python scripts/ingest_documents.py
    python scripts/ingest_documents.py --reset
"""

import sys
import argparse
from pathlib import Path
import time

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.config import settings
from app.services.ingestion import IngestionService
from app.services.embeddings import EmbeddingService
from app.services.retrieval import RetrievalService


def run_ingestion(reset: bool = False):
    print("=" * 60)
    print("🚀 NOVATECH KNOWLEDGE BASE INGESTION PIPELINE")
    print("=" * 60)

    kb_dir = Path(settings.knowledge_base_dir)
    if not kb_dir.exists():
        print(f"❌ Knowledge base directory not found at: {kb_dir}")
        sys.exit(1)

    print(f"📁 Knowledge Base Path: {kb_dir}")
    print(f"📊 Target Chroma Collection: {settings.chroma_collection_name}")
    print(f"⚙️ Chunk Size: {settings.chunk_size} | Overlap: {settings.chunk_overlap}")
    print(f"🧠 Embedding Model: {settings.embedding_model}")
    print("-" * 60)

    # Initialize services
    ingestion = IngestionService(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    retrieval = RetrievalService()

    if reset:
        print("⚠️ Reset flag provided. Clearing existing collection...")
        retrieval.reset_collection()
        print("✅ Collection cleared.")

    # 1. Process files into chunks
    start_time = time.time()
    print("⏳ Scanning and chunking documents...")
    chunks = ingestion.process_directory(kb_dir)

    if not chunks:
        print("❌ No document chunks produced. Check your knowledge_base folder.")
        sys.exit(1)

    print(f"✅ Created {len(chunks)} text chunks from documents.")

    # 2. Generate embeddings
    print("⏳ Generating vector embeddings via OpenAI API...")
    embedding_service = EmbeddingService()
    chunk_texts = [chunk.text for chunk in chunks]

    try:
        embeddings = embedding_service.get_embeddings_batch(chunk_texts, batch_size=64)
    except Exception as e:
        print(f"\n❌ Embedding generation failed: {e}")
        print("💡 Hint: Ensure a valid OPENAI_API_KEY is configured in your .env file.")
        sys.exit(1)

    # 3. Store in ChromaDB
    print("⏳ Upserting chunks and embeddings into ChromaDB...")
    upserted_count = retrieval.upsert_chunks(chunks, embeddings)
    elapsed = time.time() - start_time

    print("-" * 60)
    print(f"🎉 INGESTION COMPLETE! Indexed {upserted_count} chunks in {elapsed:.2f} seconds.")
    print(f"📦 Total chunks now in ChromaDB: {retrieval.count()}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest company knowledge base into ChromaDB.")
    parser.add_argument("--reset", action="store_true", help="Clear existing ChromaDB collection before indexing.")
    args = parser.parse_args()

    run_ingestion(reset=args.reset)
