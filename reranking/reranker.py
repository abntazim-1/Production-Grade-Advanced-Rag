"""
Cross-encoder re-ranker + context compressor.

Re-ranker:
  Takes top-50 candidates from hybrid retrieval.
  Scores each (query, chunk) pair with a cross-encoder — more accurate than bi-encoder.
  Returns top-K by cross-encoder score.

Context compressor:
  Strips sentences from each chunk that aren't relevant to the query.
  Reduces LLM prompt length and noise.
"""
import re
from sentence_transformers import CrossEncoder
from models import RetrievedChunk
from config import get_settings

settings = get_settings()


class Reranker:
    def __init__(self):
        print(f"[Reranker] Loading {settings.reranker_model} ...")
        self.model = CrossEncoder(settings.reranker_model, max_length=512)
        print("[Reranker] Ready")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.top_k_rerank
        if not candidates:
            return []

        pairs = [(query, c.chunk.content) for c in candidates]
        scores = self.model.predict(pairs)

        for rc, score in zip(candidates, scores):
            rc.rerank_score = float(score)

        ranked = sorted(candidates, key=lambda x: -x.rerank_score)
        return ranked[:top_k]


class ContextCompressor:
    """
    Sentence-level compression.
    Keeps only sentences that contain at least one query keyword.
    Falls back to full content if compression leaves < 2 sentences.
    """

    def compress(self, query: str, chunk_content: str) -> str:
        query_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        sentences = re.split(r"(?<=[.!?])\s+", chunk_content)

        relevant = []
        for s in sentences:
            s_tokens = set(re.findall(r"\b\w+\b", s.lower()))
            overlap = query_tokens & s_tokens
            if len(overlap) >= 1:
                relevant.append(s)

        if len(relevant) < 2:
            return chunk_content  # don't over-compress

        return " ".join(relevant)

    def compress_all(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        for rc in chunks:
            rc.chunk.content = self.compress(query, rc.chunk.content)
        return chunks
