import logging
from typing import List
from openai import OpenAI, OpenAIError
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service to generate vector embeddings using the official OpenAI API."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.embedding_model
        self._client: OpenAI = None

    def _get_client(self) -> OpenAI:
        """Lazy-initialize OpenAI client."""
        if not self.api_key or self.api_key == "your_openai_api_key_here":
            raise ValueError(
                "OPENAI_API_KEY is not set or contains default placeholder. "
                "Please configure a valid key in your .env file."
            )
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def get_embedding(self, text: str) -> List[float]:
        """Generate a single vector embedding for a query or document text."""
        cleaned_text = text.replace("\n", " ").strip()
        if not cleaned_text:
            raise ValueError("Cannot generate embedding for empty text.")

        try:
            client = self._get_client()
            response = client.embeddings.create(
                input=[cleaned_text],
                model=self.model
            )
            return response.data[0].embedding
        except OpenAIError as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise RuntimeError(f"Embedding service error: {str(e)}") from e

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Generate embeddings for a list of texts in batches."""
        if not texts:
            return []

        client = self._get_client()
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = [t.replace("\n", " ").strip() for t in texts[i : i + batch_size]]
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except OpenAIError as e:
                logger.error(f"Failed to generate embeddings for batch {i // batch_size}: {e}")
                raise RuntimeError(f"Batch embedding generation failed: {str(e)}") from e

        return all_embeddings
