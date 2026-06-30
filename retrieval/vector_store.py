"""
FAISS vector store.
Three separate indices: body, summary, questions.
RRF fusion happens in the hybrid retriever — this file only handles
index building and dense search.
"""
import os
import pickle
import numpy as np
import faiss
from models import Chunk, RetrievedChunk
from retrieval.embedder import Embedder
from config import get_settings

settings = get_settings()


class VectorStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.dim = embedder.dim

        # One index per vector type
        self.indices: dict[str, faiss.IndexFlatIP] = {
            "body": faiss.IndexFlatIP(self.dim),
            "summary": faiss.IndexFlatIP(self.dim),
            "questions": faiss.IndexFlatIP(self.dim),
        }

        # chunk_id → Chunk mapping (in-memory, swap for DB in production)
        self.id_to_chunk: dict[int, Chunk] = {}
        # Maps each index slot → chunk id (same for all 3 indices)
        self.index_to_id: list[str] = []

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add(self, chunks: list[Chunk]) -> None:
        body_vecs, summary_vecs, question_vecs = [], [], []

        for chunk in chunks:
            idx = len(self.index_to_id)
            self.index_to_id.append(chunk.id)
            self.id_to_chunk[idx] = chunk

            vecs = self.embedder.embed_chunk(chunk)
            body_vecs.append(vecs["body"])
            summary_vecs.append(vecs.get("summary", vecs["body"]))   # fallback to body
            question_vecs.append(vecs.get("questions", vecs["body"]))

        self.indices["body"].add(np.array(body_vecs, dtype=np.float32))
        self.indices["summary"].add(np.array(summary_vecs, dtype=np.float32))
        self.indices["questions"].add(np.array(question_vecs, dtype=np.float32))

        print(f"[VectorStore] Indexed {len(chunks)} chunks. Total={len(self.index_to_id)}")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 50,
        index_type: str = "body",
    ) -> list[tuple[str, float]]:
        """
        Returns list of (chunk_id, score) sorted by descending score.
        Searches across body, summary, AND questions indices, then takes max score.
        """
        index = self.indices[index_type]
        if index.ntotal == 0:
            return []

        k = min(top_k, index.ntotal)
        scores, ids = index.search(query_vec.reshape(1, -1), k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0:
                chunk_id = self.index_to_id[idx]
                results.append((chunk_id, float(score)))
        return results

    def multi_vector_search(
        self,
        query_vec: np.ndarray,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """
        Search all three indices, take max score per chunk_id.
        This way a chunk can be found via body, summary, or question match.
        """
        best: dict[str, float] = {}
        for idx_type in ["body", "summary", "questions"]:
            for chunk_id, score in self.search(query_vec, top_k, idx_type):
                if chunk_id not in best or score > best[chunk_id]:
                    best[chunk_id] = score

        return sorted(best.items(), key=lambda x: -x[1])[:top_k]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        for idx, cid in enumerate(self.index_to_id):
            if cid == chunk_id:
                return self.id_to_chunk[idx]
        return None

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = None) -> None:
        path = path or settings.faiss_index_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "indices": {k: faiss.serialize_index(v) for k, v in self.indices.items()},
            "id_to_chunk": self.id_to_chunk,
            "index_to_id": self.index_to_id,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"[VectorStore] Saved to {path}")

    def load(self, path: str = None) -> None:
        path = path or settings.faiss_index_path
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.indices = {k: faiss.deserialize_index(v) for k, v in state["indices"].items()}
        self.id_to_chunk = state["id_to_chunk"]
        self.index_to_id = state["index_to_id"]
        print(f"[VectorStore] Loaded {len(self.index_to_id)} chunks from {path}")
