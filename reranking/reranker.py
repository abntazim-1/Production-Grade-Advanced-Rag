"""
Cross-encoder re-ranker — port from mini_rag.py.

Matches mini_rag.py's rerank() exactly:
  - Filters out negative-score chunks (cross-encoder says: not relevant)
  - Falls back to keeping top-1 if ALL chunks score negative
  - Respects USE_RERANKER flag from config
  - ContextCompressor is a no-op pass-through (mini_rag.py also does this:
    "pass the full chunk unmodified to preserve accuracy")
"""
import torch
from sentence_transformers import CrossEncoder
from models import RetrievedChunk
from config import get_settings
from retrieval.embedder import _best_device

settings = get_settings()


class Reranker:
    def __init__(self):
        device = _best_device()
        print(f"[Reranker] Loading {settings.reranker_model} on {device}...")
        self.model = CrossEncoder(settings.reranker_model, max_length=512, device=device)
        # Warm up (matches mini_rag.py's warmup call)
        _ = self.model.predict([("warmup", "warmup")])
        print("[Reranker] Ready.")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = None,
    ) -> list[RetrievedChunk]:
        """
        Mirrors mini_rag.py's rerank() exactly:
          1. If USE_RERANKER is False → just truncate to top_k
          2. Score all (query, chunk) pairs with cross-encoder
          3. Sort descending by rerank_score
          4. Filter negative-score chunks (cross-encoder says: not relevant)
          5. Fall back to top-1 if ALL chunks score negative
        """
        top_k = top_k or settings.top_k_rerank

        if not candidates or not settings.use_reranker:
            return candidates[:top_k]

        pairs  = [(query, c.chunk.content) for c in candidates]
        scores = self.model.predict(pairs)

        for rc, score in zip(candidates, scores):
            rc.rerank_score = float(score)

        ranked   = sorted(candidates, key=lambda x: -x.rerank_score)
        top      = ranked[:top_k]
        positive = [h for h in top if h.rerank_score >= 0]

        # Keep negatives only if ALL chunks score negative — guarantees >= 1 result
        return positive if positive else top[:1]


class ContextCompressor:
    """
    No-op compressor — matches mini_rag.py's compress() decision:

    'ENTERPRISE FIX: Naive keyword sentence filtering destroys semantic context.
    Since we are using a powerful CrossEncoder reranker to select the best chunks,
    we should pass the full chunk unmodified to the LLM to preserve accuracy.'
    """

    def compress(self, query: str, chunk_content: str) -> str:
        return chunk_content.strip()

    def compress_all(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        # Content passed through unmodified — matches mini_rag.py
        return chunks
