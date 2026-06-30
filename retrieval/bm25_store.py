"""
BM25 sparse retriever — port from mini_rag.py.

Key changes vs previous version:
  - Tokenizer indexes content + keywords + summary + questions (matches mini_rag.py)
  - Expanded stopword list (matches mini_rag.py)
  - No pickle persistence needed — rebuilt from VectorStore.all_chunks on startup
"""
import re
from rank_bm25 import BM25Okapi
from models import Chunk

STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "with", "this", "that", "are",
    "was", "were", "be", "been", "has", "have", "had",
}


def tokenize(text: str) -> list[str]:
    """
    Matches mini_rag.py tokenize() exactly.
    Lowercases, extracts word tokens, removes stopwords.
    """
    return [w for w in re.findall(r"\b\w+\b", text.lower()) if w not in STOPWORDS]


class BM25Store:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """
        Builds BM25 index over:
            chunk.content + keywords + summary + questions
        Mirrors mini_rag.py's build_bm25() exactly.
        """
        self.chunks = list(chunks)
        if not self.chunks:
            return
        corpus = [
            tokenize(
                c.content
                + " " + " ".join(str(k) for k in c.keywords)
                + " " + (c.summary or "")
                + " " + " ".join(str(q) for q in c.questions)
            )
            for c in self.chunks
        ]
        self.bm25 = BM25Okapi(corpus)
        print(f"[BM25Store] Indexed {len(self.chunks)} chunks.")

    def search(self, query: str, top_k: int = None) -> list[tuple[str, float]]:
        """Returns list of (chunk_id, score) sorted descending. Skips zero scores."""
        from config import get_settings
        top_k = top_k or get_settings().top_k_retrieval
        if not self.bm25 or not self.chunks:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return [
            (self.chunks[i].id, float(s))
            for i, s in ranked[:top_k]
            if s > 0
        ]

    # ── Legacy persistence stubs (kept so api/app.py won't crash) ─────────────

    def save(self, path: str = None) -> None:
        pass  # BM25 is rebuilt from Qdrant on every restart

    def load(self, path: str = None) -> None:
        pass
