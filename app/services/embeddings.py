import logging
from typing import List, Optional, ClassVar, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Local vector embedding service using Hugging Face Sentence Transformers.
    Runs 100% locally on CPU/GPU with zero OpenAI API credit usage.
    """

    _model_cache: ClassVar[Dict[str, Any]] = {}

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self._model: Optional[Any] = None

    def _get_model(self) -> Any:
        """Lazy-load and cache the SentenceTransformer model instance."""
        if self.model_name not in EmbeddingService._model_cache:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local SentenceTransformer model: '{self.model_name}'...")
            model = SentenceTransformer(self.model_name)
            EmbeddingService._model_cache[self.model_name] = model
            logger.info(f"Successfully loaded '{self.model_name}'.")

        if self._model is None:
            self._model = EmbeddingService._model_cache[self.model_name]

        return self._model

    def get_embedding(self, text: str) -> List[float]:
        """Generate a 384-dimensional vector embedding for a single text query or chunk."""
        cleaned_text = text.replace("\n", " ").strip()
        if not cleaned_text:
            raise ValueError("Cannot generate embedding for empty text.")

        model = self._get_model()
        # encode returns numpy ndarray, convert to standard Python list of floats
        embedding = model.encode(cleaned_text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Generate vector embeddings for a list of texts in batches."""
        if not texts:
            return []

        cleaned_texts = [t.replace("\n", " ").strip() for t in texts]
        model = self._get_model()

        # Batch encode with progress bar for CLI feedback
        embeddings = model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return [emb.tolist() for emb in embeddings]
