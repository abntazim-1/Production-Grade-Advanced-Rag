from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── API Keys ───────────────────────────────────────────────────────────────
    groq_api_key: str = ""

    # ── Models ─────────────────────────────────────────────────────────────────
    embedding_model: str  = "BAAI/bge-m3"
    # TinyBERT is 4x faster than MiniLM-L-6 with minimal quality loss
    reranker_model: str   = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    llm_model: str        = "llama-3.3-70b-versatile"   # main generation model
    metadata_model: str   = "llama-3.1-8b-instant"      # fast model for metadata + query rewrite

    # ── Retrieval ──────────────────────────────────────────────────────────────
    top_k_retrieval: int  = 15     # candidates passed to reranker
    top_k_rerank: int     = 5      # final chunks sent to LLM
    rrf_k: int            = 60     # Reciprocal Rank Fusion constant
    use_reranker: bool    = True
    cache_size: int       = 1024   # LRU cache entries for retrieve+rerank

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_size: int       = 400    # characters (not tokens)
    chunk_overlap: int    = 3      # lines carried into next chunk on size-split

    # ── Ingestion ──────────────────────────────────────────────────────────────
    generate_metadata: bool = True
    metadata_workers: int   = 6    # parallel threads for LLM metadata generation
    embed_batch_size: int   = 256  # BGE-M3 batch size (tuned for 8 GB VRAM)

    # ── Generation ─────────────────────────────────────────────────────────────
    max_context_chars: int  = 12000
    max_tokens: int         = 512
    metadata_max_tokens: int = 300

    # ── Memory ─────────────────────────────────────────────────────────────────
    memory_maxlen: int     = 50    # max turns per session before eviction

    # ── Vector Store (Qdrant) ──────────────────────────────────────────────────
    qdrant_path: str       = "qdrant_db"
    qdrant_collection: str = "chunks"

    # ── Legacy FAISS paths (kept for reference, not used) ─────────────────────
    faiss_index_path: str  = "./storage/faiss.index"
    bm25_index_path: str   = "./storage/bm25.pkl"

    class Config:
        env_file = ".env"
        extra = "ignore"   # silently ignore unknown .env keys (e.g. legacy supabase_* fields)


@lru_cache
def get_settings() -> Settings:
    return Settings()
