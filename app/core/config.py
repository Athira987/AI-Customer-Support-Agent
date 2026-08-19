from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings and configuration parameters."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # OpenAI Settings
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")

    # ChromaDB Settings
    chroma_persist_directory: str = Field(default=str(BASE_DIR / "data" / "chroma"), alias="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field(default="novatech_knowledge_base", alias="CHROMA_COLLECTION_NAME")

    # Knowledge Base Path
    knowledge_base_dir: str = Field(default=str(BASE_DIR / "knowledge_base"), alias="KNOWLEDGE_BASE_DIR")

    # RAG Retrieval Parameters
    top_k: int = Field(default=4, alias="TOP_K")
    similarity_distance_threshold: float = Field(default=0.65, alias="SIMILARITY_DISTANCE_THRESHOLD")

    # Chunking Configuration
    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")

    # API Server Settings
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")


settings = Settings()
