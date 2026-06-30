"""
Hybrid retriever — port from mini_rag.py.

Key features matching mini_rag.py:
  - Dense (Qdrant) + Sparse (BM25) run in PARALLEL via ThreadPoolExecutor
    (OPT-4 from mini_rag.py: wall-clock = max(dense, sparse) not dense+sparse)
  - Reciprocal Rank Fusion (RRF) with RRF_K=60
  - LRU cache on retrieve+rerank result (matches cached_retrieve_and_rerank)
  - Returns immutable tuple from cache; callers convert with list()
"""
import functools
from concurrent.futures import ThreadPoolExecutor

from models import Chunk, RetrievedChunk
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.bm25_store import BM25Store
from retrieval.query_rewriter import QueryRewriter
from config import get_settings

settings = get_settings()

# Dedicated thread pool for parallel dense+sparse search (OPT-4 from mini_rag.py)
_search_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="retriever")

RRF_K = 60  # standard RRF constant (matches mini_rag.py)


def _rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]],
    top_k: int = None,
) -> list[tuple[str, float]]:
    """
    Merge multiple ranked lists via Reciprocal Rank Fusion.
    Matches rrf() in mini_rag.py exactly.
    """
    top_k = top_k or settings.top_k_retrieval
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, (chunk_id, _) in enumerate(lst):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        embedder: Embedder,
        query_rewriter: QueryRewriter | None = None,
    ):
        self.vector_store   = vector_store
        self.bm25_store     = bm25_store
        self.embedder       = embedder
        self.query_rewriter = query_rewriter

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        history: str = "",
        use_query_rewriting: bool = True,
    ) -> list[RetrievedChunk]:
        """
        Full hybrid retrieval pipeline:
          1. Query rewriting (skipped when no history, matching mini_rag.py)
          2. Dense + Sparse search in PARALLEL (OPT-4 from mini_rag.py)
          3. RRF fusion
          4. Build RetrievedChunk list
        """
        top_k = top_k or settings.top_k_retrieval

        # ── Step 1: Query rewriting ───────────────────────────────────────────
        if use_query_rewriting and self.query_rewriter:
            rewritten = self.query_rewriter.rewrite(query, history)
        else:
            rewritten = query

        # ── Step 2: Parallel dense + sparse search ────────────────────────────
        q_vec = self.embedder.embed_query(rewritten)
        dense_fut  = _search_executor.submit(self.vector_store.search,  q_vec, top_k)
        sparse_fut = _search_executor.submit(self.bm25_store.search,    rewritten, top_k)
        dense  = dense_fut.result()
        sparse = sparse_fut.result()

        # ── Step 3: RRF fusion ────────────────────────────────────────────────
        fused = _rrf_fuse([dense, sparse], top_k=top_k)

        # ── Step 4: Build RetrievedChunk objects ──────────────────────────────
        dense_scores  = dict(dense)
        sparse_scores = dict(sparse)

        results: list[RetrievedChunk] = []
        for chunk_id, rrf_score in fused:
            chunk = self.vector_store.get_chunk(chunk_id)
            if chunk is None:
                # Fallback: scan BM25 store
                for c in self.bm25_store.chunks:
                    if c.id == chunk_id:
                        chunk = c
                        break
            if chunk:
                results.append(RetrievedChunk(
                    chunk        = chunk,
                    dense_score  = dense_scores.get(chunk_id, 0.0),
                    sparse_score = sparse_scores.get(chunk_id, 0.0),
                    rrf_score    = rrf_score,
                ))

        return results, rewritten   # return rewritten query so callers can log it


def build_cached_retriever(retriever: HybridRetriever, reranker):
    """
    Returns a cached retrieve-and-rerank function.
    Mirrors cached_retrieve_and_rerank() in mini_rag.py.

    Returns an immutable tuple (BUG 2 fix from mini_rag.py) — callers
    must convert with list() before mutating.
    Cache is cleared automatically when new chunks are added.
    """
    @functools.lru_cache(maxsize=settings.cache_size)
    def cached(query: str) -> tuple:
        hits, _ = retriever.retrieve(query)
        reranked = reranker.rerank(query, hits)
        return tuple(reranked)

    return cached
