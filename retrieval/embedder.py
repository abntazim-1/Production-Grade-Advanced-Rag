"""
BGE-M3 embedding engine — exact port from mini_rag.py.

Key features:
  - Intel Arc XPU → NVIDIA CUDA → CPU device selection (same priority as mini_rag)
  - @lru_cache on single-query embedding (0.01 ms cache hit vs 50-150 ms GPU)
  - Batch embedding for ingestion (no cache, always recomputes)
  - Model warm-up on startup to eliminate first-query cold start
"""
import functools
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from config import get_settings

settings = get_settings()


def _best_device() -> str:
    """
    Identical priority order to mini_rag.py:
      1. Intel Arc / Intel GPU (XPU)
      2. NVIDIA CUDA
      3. CPU with diagnostics
    """
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        name = (torch.xpu.get_device_name(0)
                if hasattr(torch.xpu, "get_device_name") else "Intel GPU")
        print(f"[GPU] Intel Arc XPU detected: {name} — using GPU for embeddings!")
        return "xpu"

    if torch.cuda.is_available():
        print(f"[GPU] CUDA GPU detected: {torch.cuda.get_device_name(0)} — using GPU for embeddings!")
        return "cuda"

    print("[CPU] No GPU detected — falling back to CPU for embeddings.")
    print("      -> Ensure torch+xpu or torch+cu is installed.")
    return "cpu"


class Embedder:
    def __init__(self):
        self.device = _best_device()
        print(f"[Embedder] Loading {settings.embedding_model} on {self.device} ...")
        self.model = SentenceTransformer(settings.embedding_model, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Ready. Dim={self.dim}")

        # Warm up — eliminates first-query cold start (same as mini_rag.py)
        print("[Embedder] Warming up to eliminate first-query cold start...")
        _ = self.model.encode(["warmup"], batch_size=1, show_progress_bar=False)
        print("[Embedder] Warm-up done.")

    # ── Internal cached single-string embed ───────────────────────────────────

    @functools.lru_cache(maxsize=2048)
    def _embed_single_cached(self, text: str) -> tuple:
        """
        LRU-cached single-string embedding (matches mini_rag.py).
        Returns a Python tuple (hashable) so lru_cache can store it.
        Convert back to np.ndarray at call sites.
        """
        vec = self.model.encode(
            [text],
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False,
        )
        return tuple(vec[0].tolist())

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings. Returns (N, dim) float32 array.

        Single-text calls hit the LRU cache (fast path for queries).
        Multi-text calls bypass the cache (batch ingestion).
        Mirrors mini_rag.py's embed() exactly.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        if len(texts) == 1:
            return np.array([self._embed_single_cached(texts[0])], dtype=np.float32)
        # Batch path — ingestion, always recompute
        return np.array(
            self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=settings.embed_batch_size,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Single query embedding — hits the LRU cache."""
        return self.embed([query])[0]

    def embed_chunk(self, chunk) -> dict[str, np.ndarray]:
        """
        Returns up to three embeddings per chunk:
          - 'body'      : the chunk text itself
          - 'summary'   : the LLM-generated summary (if available)
          - 'questions' : mean of synthetic question embeddings (if available)
        Mirrors the multi-vector approach in mini_rag.py.
        """
        texts_to_embed = [chunk.content]
        if chunk.summary:
            texts_to_embed.append(chunk.summary)
        if chunk.questions:
            texts_to_embed.extend(chunk.questions)

        vecs = self.embed(texts_to_embed)

        result: dict[str, np.ndarray] = {"body": vecs[0]}
        idx = 1
        if chunk.summary:
            result["summary"] = vecs[idx]; idx += 1
        if chunk.questions:
            result["questions"] = vecs[idx : idx + len(chunk.questions)].mean(axis=0)

        return result
