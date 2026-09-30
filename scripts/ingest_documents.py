"""
Document Ingestion Script for NovaTech Knowledge Base.
Loads all .txt and .pdf documents, chunks text, generates embeddings locally using SentenceTransformers,
and indexes vectors into ChromaDB without making OpenAI embedding API calls.

Usage:
    python scripts/ingest_documents.py
    python scripts/ingest_documents.py --reset
"""

import sys
import argparse
from pathlib import Path
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.config import settings
from app.services.ingestion import IngestionService
from app.services.embeddings import EmbeddingService
from app.services.retrieval import RetrievalService


def run_ingestion(reset: bool = False):
    print("=" * 65)
    print("🚀 NOVATECH KNOWLEDGE BASE LOCAL INGESTION PIPELINE")
    print("=" * 65)

    kb_dir = Path(settings.knowledge_base_dir)
    if not kb_dir.exists():
        print(f"❌ Knowledge base directory not found at: {kb_dir}")
        sys.exit(1)

    print(f"📁 Knowledge Base Path: {kb_dir}")
    print(f"📊 Target Chroma Collection: {settings.chroma_collection_name}")
    print(f"⚙️ Chunk Size: {settings.chunk_size} | Overlap: {settings.chunk_overlap}")
    print(f"🧠 Embedding Model (Local): {settings.embedding_model}")
    print("🔒 Mode: 100% Local Inference (Zero OpenAI Embedding API calls)")
    print("-" * 65)

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
    print("⏳ Scanning and chunking documents from knowledge base...")
    chunks = ingestion.process_directory(kb_dir)

    if not chunks:
        print("❌ No document chunks produced. Check your knowledge_base folder.")
        sys.exit(1)

    print(f"✅ Created {len(chunks)} text chunks from documents.")

    # 2. Generate local embeddings
    print("⏳ Generating local vector embeddings via SentenceTransformers...")
    embedding_service = EmbeddingService()
    chunk_texts = [chunk.text for chunk in chunks]

    emb_start = time.time()
    embeddings = embedding_service.get_embeddings_batch(chunk_texts, batch_size=64)
    emb_elapsed = time.time() - emb_start
    print(f"✅ Generated {len(embeddings)} local embeddings in {emb_elapsed:.2f}s ({len(embeddings)/max(emb_elapsed,0.01):.1f} chunks/sec).")

    # 3. Store in ChromaDB
    print("⏳ Upserting chunks and embeddings into ChromaDB...")
    upserted_count = retrieval.upsert_chunks(chunks, embeddings)
    total_elapsed = time.time() - start_time

    print("-" * 65)
    print(f"🎉 INGESTION COMPLETE! Indexed {upserted_count} chunks in {total_elapsed:.2f} seconds.")
    print(f"📦 Total chunks now in ChromaDB: {retrieval.count()}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest company knowledge base locally into ChromaDB.")
    parser.add_argument("--reset", action="store_true", help="Clear existing ChromaDB collection before indexing.")
    args = parser.parse_args()

    run_ingestion(reset=args.reset)
