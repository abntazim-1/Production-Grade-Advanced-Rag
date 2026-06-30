"""
Qdrant vector store — exact port from mini_rag.py.

Replaces the previous FAISS implementation with Qdrant for:
  - Persistent storage (survives restarts)
  - Multi-vector indexing (content + summary + questions per chunk)
  - O(1) chunk lookup via _chunk_index dict (BUG 5 fix from mini_rag.py)
  - Startup restore: re-hydrates all_chunks and BM25 from Qdrant on boot
  - Deduplication: rejects re-ingestion of the same source (BUG 3 fix)
"""
import threading
import uuid
import atexit
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance,
    Filter, FieldCondition, MatchValue,
)
from models import Chunk
from retrieval.embedder import Embedder
from config import get_settings

settings = get_settings()


class VectorStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.dim = embedder.dim

        # ── Qdrant client (persistent local storage, same as mini_rag.py) ──────
        self.client = QdrantClient(path=settings.qdrant_path)
        self.collection = settings.qdrant_collection
        atexit.register(self.client.close)

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

        # ── In-memory state ───────────────────────────────────────────────────
        # Lock protects all_chunks and _chunk_index from concurrent read/write.
        self._lock: threading.Lock = threading.Lock()
        self.all_chunks: list[Chunk] = []
        # O(1) chunk lookup dict (BUG 5 fix from mini_rag.py)
        self._chunk_index: dict[str, Chunk] = {}
        # Track ingested sources to prevent duplicate indexing (BUG 3 fix)
        self.ingested_sources: set[str] = set()

        # Restore persisted state from Qdrant on startup (BUG 2 fix from mini_rag.py)
        self._restore_from_qdrant()

    # ── Startup restore ───────────────────────────────────────────────────────

    def _restore_from_qdrant(self) -> None:
        """
        Scrolls Qdrant for all 'content'-type vectors and rebuilds all_chunks
        and _chunk_index in memory so BM25 and get_chunk() work after a restart.
        Matches load_chunks_from_qdrant() in mini_rag.py exactly.
        """
        offset = None
        restored = 0
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key="vector_type", match=MatchValue(value="content"))
                ]),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                p = rec.payload
                chunk = Chunk(
                    id        = p["chunk_id"],
                    doc_id    = p.get("doc_id", ""),
                    content   = p["content"],
                    source    = p.get("source", ""),
                    heading   = p.get("heading"),
                    summary   = p.get("summary"),
                    keywords  = [str(k) for k in p.get("keywords", [])],
                    questions = [str(q) for q in p.get("questions", [])],
                )
                self.all_chunks.append(chunk)
                self._chunk_index[chunk.id] = chunk
                if chunk.source:
                    self.ingested_sources.add(chunk.source)
                restored += 1
            if next_offset is None:
                break
            offset = next_offset
        if restored:
            print(f"[VectorStore] Restored {restored} chunks from Qdrant on startup.")

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add(self, chunks: list[Chunk]) -> None:
        """
        Multi-vector ingestion: embeds content + summary + questions separately
        and stores each as its own Qdrant point (same as add_chunks() in mini_rag.py).
        """
        if not chunks:
            return
        points = []
        for c in chunks:
            texts_to_embed = [c.content]
            if c.summary:     texts_to_embed.append(c.summary)
            if c.questions:   texts_to_embed.extend(c.questions)

            vecs = self.embedder.embed(texts_to_embed)

            base_payload = {
                "chunk_id"  : c.id,
                "doc_id"    : c.doc_id,
                "content"   : c.content,
                "source"    : c.source,
                "heading"   : c.heading,
                "summary"   : c.summary or "",
                "keywords"  : c.keywords,
                "questions" : c.questions,
            }

            vec_types = ["content"]
            if c.summary:     vec_types.append("summary")
            if c.questions:   vec_types.extend(["question"] * len(c.questions))

            for vec_type, vec in zip(vec_types, vecs):
                pt_id   = c.id if vec_type == "content" else str(uuid.uuid4())
                payload = {**base_payload, "vector_type": vec_type}
                points.append(PointStruct(id=pt_id, vector=vec.tolist(), payload=payload))

        self.client.upsert(collection_name=self.collection, points=points)

        with self._lock:
            self.all_chunks.extend(chunks)
            for c in chunks:
                self._chunk_index[c.id] = c
                if c.source:
                    self.ingested_sources.add(c.source)

        print(f"[VectorStore] Indexed {len(chunks)} chunks ({len(points)} total vectors).")

    # ── Dense Search ──────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = None,
    ) -> list[tuple[str, float]]:
        """
        Dense search across all vector types.
        Deduplicates by keeping the best score per logical chunk_id.
        Mirrors dense_search() in mini_rag.py.
        """
        top_k = top_k or settings.top_k_retrieval
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vec.tolist(),
            limit=top_k * 3,  # over-fetch because many vectors map to one chunk
        )
        best: dict[str, float] = {}
        for res in response.points:
            cid = res.payload["chunk_id"]
            if cid not in best or res.score > best[cid]:
                best[cid] = res.score
        return sorted(best.items(), key=lambda x: -x[1])[:top_k]

    # ── Chunk Lookup ──────────────────────────────────────────────────────────

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """O(1) lookup — BUG 5 fix from mini_rag.py."""
        return self._chunk_index.get(str(chunk_id))

    # ── Source Management ─────────────────────────────────────────────────────

    def delete_source(self, source_name: str) -> int:
        """
        Remove all vectors and in-memory chunks belonging to a source.
        Mirrors DELETE /source/{name} in mini_rag.py.
        """
        from qdrant_client.models import FilterSelector
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(must=[
                    FieldCondition(key="source", match=MatchValue(value=source_name))
                ])
            ),
        )
        with self._lock:
            before = len(self.all_chunks)
            self.all_chunks[:] = [c for c in self.all_chunks if c.source != source_name]
            removed = before - len(self.all_chunks)
            for cid in [k for k, v in self._chunk_index.items() if v.source == source_name]:
                del self._chunk_index[cid]
        self.ingested_sources.discard(source_name)
        return removed

    # ── Index info ────────────────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        return len(self.all_chunks)

    def get_collection_info(self):
        return self.client.get_collection(self.collection)

    # ── Legacy FAISS compat (no-ops, kept so api/app.py won't crash) ──────────

    def save(self, path: str = None) -> None:
        pass  # Qdrant persists automatically

    def load(self, path: str = None) -> None:
        pass  # Qdrant restores on __init__ via _restore_from_qdrant
