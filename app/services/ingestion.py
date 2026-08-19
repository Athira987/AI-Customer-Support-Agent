import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a chunk of a document with associated metadata."""

    def __init__(
        self,
        chunk_id: str,
        text: str,
        metadata: Dict[str, Any]
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata
        }


class IngestionService:
    """Service to load, clean, chunk, and prepare documents for vector storage."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def extract_text_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text from a file (.txt or .pdf).
        Returns a list of pages/sections with text and page numbers.
        """
        suffix = file_path.suffix.lower()
        extracted: List[Dict[str, Any]] = []

        if suffix == ".txt":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as f:
                    content = f.read()
            extracted.append({"text": content, "page": 1})

        elif suffix == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                for page_idx, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        extracted.append({"text": page_text, "page": page_idx})
            except ImportError:
                logger.error("pypdf is not installed. PDF extraction will be skipped.")
                raise RuntimeError("pypdf required for PDF parsing. Install with 'pip install pypdf'.")
            except Exception as e:
                logger.error(f"Error reading PDF {file_path}: {e}")
                raise

        else:
            logger.warning(f"Unsupported file format ignored: {file_path}")

        return extracted

    def clean_text(self, text: str) -> str:
        """Clean and normalize raw document text."""
        # Replace multiple spaces/tabs with single space
        cleaned = re.sub(r"[ \t]+", " ", text)
        # Normalize excessive newlines
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
        return cleaned.strip()

    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Chunk text into overlapping windows while respecting paragraph and sentence boundaries.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        if len(cleaned) <= chunk_size:
            return [cleaned]

        chunks = []
        start = 0
        text_length = len(cleaned)

        while start < text_length:
            end = start + chunk_size

            if end >= text_length:
                chunks.append(cleaned[start:].strip())
                break

            # Attempt to split at paragraph or sentence boundary near chunk_size
            split_pos = cleaned.rfind("\n\n", start, end)
            if split_pos == -1 or split_pos <= start:
                split_pos = cleaned.rfind(". ", start, end)
                if split_pos != -1:
                    split_pos += 1  # Include the period
            if split_pos == -1 or split_pos <= start:
                split_pos = cleaned.rfind(" ", start, end)

            if split_pos == -1 or split_pos <= start:
                split_pos = end

            chunk = cleaned[start:split_pos].strip()
            if chunk:
                chunks.append(chunk)

            # Advance with overlap
            start = max(start + 1, split_pos - chunk_overlap)

        return chunks

    def process_document(self, file_path: Path) -> List[DocumentChunk]:
        """Process a single document file into metadata-tagged DocumentChunks."""
        pages = self.extract_text_from_file(file_path)
        chunks: List[DocumentChunk] = []

        category = file_path.parent.name
        filename = file_path.name
        doc_type = file_path.suffix.replace(".", "")

        chunk_counter = 0
        for page_info in pages:
            raw_text = page_info["text"]
            page_num = page_info.get("page", 1)

            text_chunks = self.chunk_text(raw_text, self.chunk_size, self.chunk_overlap)
            for idx, text in enumerate(text_chunks):
                chunk_counter += 1
                # Generate deterministic, idempotent chunk ID
                id_seed = f"{filename}_{page_num}_{idx}_{text[:30]}"
                chunk_id = hashlib.md5(id_seed.encode("utf-8")).hexdigest()[:16]

                metadata = {
                    "source": str(file_path),
                    "filename": filename,
                    "document_type": doc_type,
                    "category": category,
                    "page": page_num,
                    "chunk_index": chunk_counter,
                    "char_count": len(text)
                }

                chunks.append(DocumentChunk(
                    chunk_id=f"{filename}_{chunk_counter}_{chunk_id}",
                    text=text,
                    metadata=metadata
                ))

        return chunks

    def process_directory(self, directory_path: Path) -> List[DocumentChunk]:
        """Recursively process all supported documents in a directory."""
        supported_extensions = {".txt", ".pdf"}
        all_chunks: List[DocumentChunk] = []

        if not directory_path.exists():
            logger.warning(f"Knowledge base directory {directory_path} does not exist.")
            return []

        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in supported_extensions:
                    doc_chunks = self.process_document(file_path)
                    all_chunks.extend(doc_chunks)

        logger.info(f"Processed {len(all_chunks)} chunks from {directory_path}")
        return all_chunks
