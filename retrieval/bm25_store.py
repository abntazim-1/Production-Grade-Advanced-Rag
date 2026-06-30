"""
BM25 sparse retriever.
Builds a keyword index over chunk content + keywords.
Complements dense retrieval for exact-match queries.
"""
import os
import pickle
import re
from rank_bm25 import BM25Okapi
from models import Chunk
from config import get_settings

settings = get_settings()

STOPWORDS = {
    "the","a","an","is","it","in","on","at","to","for","of","and","or","but",
    "not","with","this","that","are","was","were","be","been","has","have","had",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class BM25Store:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        # Index: chunk content + keywords combined
        corpus = []
        for c in chunks:
            text = c.content + " " + " ".join(c.keywords)
            corpus.append(tokenize(text))
        self.bm25 = BM25Okapi(corpus)
        print(f"[BM25Store] Indexed {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        if not self.bm25 or not self.chunks:
            return []
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        results = []
        for idx, score in ranked[:top_k]:
            if score > 0:
                results.append((self.chunks[idx].id, float(score)))
        return results

    def save(self, path: str = None) -> None:
        path = path or settings.bm25_index_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        print(f"[BM25Store] Saved to {path}")

    def load(self, path: str = None) -> None:
        path = path or settings.bm25_index_path
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.bm25 = state["bm25"]
        self.chunks = state["chunks"]
        print(f"[BM25Store] Loaded {len(self.chunks)} chunks from {path}")
