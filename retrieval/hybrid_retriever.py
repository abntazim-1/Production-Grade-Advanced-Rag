"""
Hybrid retriever.
Combines dense multi-vector search (FAISS) and sparse BM25.
Merges results with Reciprocal Rank Fusion (RRF).
Runs rewritten query variants in parallel.
"""
from models import Chunk, RetrievedChunk
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.bm25_store import BM25Store
from retrieval.query_rewriter import QueryRewriter
from config import get_settings

settings = get_settings()

RRF_K = 60  # standard RRF constant


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)


def _rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]],
) -> list[tuple[str, float]]:
    """
    Merge multiple ranked lists into one via Reciprocal Rank Fusion.
    Each list is (chunk_id, score). Score is ignored; only rank matters.
    Returns (chunk_id, rrf_score) sorted descending.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + _rrf_score(rank)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        embedder: Embedder,
        query_rewriter: QueryRewriter | None = None,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedder = embedder
        self.query_rewriter = query_rewriter

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_query_rewriting: bool = True,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.top_k_retrieval

        # Step 1: Query rewriting
        if use_query_rewriting and self.query_rewriter:
            variants = self.query_rewriter.rewrite(query)
        else:
            variants = {"original": query}

        all_dense_lists = []
        all_sparse_lists = []

        # Step 2: Retrieve for each query variant
        for variant_name, variant_query in variants.items():
            # Dense
            q_vec = self.embedder.embed_query(variant_query)
            dense = self.vector_store.multi_vector_search(q_vec, top_k=top_k)
            all_dense_lists.append(dense)

            # Sparse
            sparse = self.bm25_store.search(variant_query, top_k=top_k)
            all_sparse_lists.append(sparse)

        # Step 3: RRF fusion across all dense lists
        fused_dense = _rrf_fuse(all_dense_lists)

        # Step 4: RRF fusion across all sparse lists
        fused_sparse = _rrf_fuse(all_sparse_lists)

        # Step 5: Final RRF fusion of dense + sparse
        final_ranked = _rrf_fuse([fused_dense, fused_sparse])[:top_k]

        # Step 6: Build RetrievedChunk objects
        # Build score lookup
        dense_scores = dict(fused_dense)
        sparse_scores = dict(fused_sparse)

        results = []
        for chunk_id, rrf_score in final_ranked:
            chunk = self.vector_store.get_chunk(chunk_id)
            if not chunk:
                # fallback to BM25 store
                for c in self.bm25_store.chunks:
                    if c.id == chunk_id:
                        chunk = c
                        break
            if chunk:
                results.append(RetrievedChunk(
                    chunk=chunk,
                    dense_score=dense_scores.get(chunk_id, 0.0),
                    sparse_score=sparse_scores.get(chunk_id, 0.0),
                    rrf_score=rrf_score,
                ))

        return results
