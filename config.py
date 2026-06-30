from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key: str = ""
    supabase_url: str = ""
    supabase_key: str = ""
    postgres_url: str = "postgresql://localhost:5432/ragdb"

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "llama3-70b-8192"

    faiss_index_path: str = "./storage/faiss.index"
    bm25_index_path: str = "./storage/bm25.pkl"

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 50
    top_k_rerank: int = 8

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
