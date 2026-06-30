"""
BGE-M3 embedding engine.
Generates dense embeddings for: chunk body, summary, and synthetic questions.
This enables multi-vector retrieval — a query can match any of the three.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from config import get_settings
from models import Chunk

settings = get_settings()


class Embedder:
    def __init__(self):
        print(f"[Embedder] Loading {settings.embedding_model} ...")
        self.model = SentenceTransformer(settings.embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Ready. Dim={self.dim}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings. Returns (N, dim) float32 array."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,   # cosine similarity via dot product
            show_progress_bar=False,
            batch_size=32,
        )
        return np.array(vecs, dtype=np.float32)

    def embed_chunk(self, chunk: Chunk) -> dict[str, np.ndarray]:
        """
        Returns three embeddings per chunk:
          - 'body'     : the chunk text itself
          - 'summary'  : the LLM-generated summary (if available)
          - 'questions': average of synthetic question embeddings
        """
        texts = {"body": chunk.content}

        if chunk.summary:
            texts["summary"] = chunk.summary

        result: dict[str, np.ndarray] = {}
        for key, text in texts.items():
            result[key] = self.embed([text])[0]

        if chunk.questions:
            q_vecs = self.embed(chunk.questions)
            result["questions"] = q_vecs.mean(axis=0)

        return result

    def embed_query(self, query: str) -> np.ndarray:
        """Single query embedding."""
        return self.embed([query])[0]
