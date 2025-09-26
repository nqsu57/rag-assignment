from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "demo_collection"
    EMBEDDING_DIM: int = 384
    RECREATE_COLLECTION: bool = True

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    LLM_PROVIDER: str = "mock"
    HF_API_TOKEN: str | None = None
    HF_MODEL: str = "google/flan-t5-small"
    EMBED_MODEL: str = "all-MiniLM-L6-v2"

    TOP_K: int = 3
    MAX_CONTEXT_CHARS: int = 3000
    SIM_THRESHOLD: float = 0.7
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()