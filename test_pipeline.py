"""
Smoke test — runs the full pipeline on a toy document.
Usage: python test_pipeline.py
Does NOT require Groq API (skips LLM steps if key is missing).
"""
import os
from models import Document
from ingestion.chunker import StructureAwareChunker
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.bm25_store import BM25Store
from retrieval.hybrid_retriever import HybridRetriever
from reranking.reranker import Reranker, ContextCompressor

SAMPLE_TEXT = """
# Introduction to Neural Networks

Neural networks are computational models inspired by the human brain.
They consist of layers of interconnected nodes called neurons.
Each connection has a weight that is updated during training.

## Backpropagation

Backpropagation is the algorithm used to train neural networks.
It computes the gradient of the loss function with respect to each weight.
The gradients are used to update weights via gradient descent.
The learning rate controls how large each update step is.

## Activation Functions

Activation functions introduce non-linearity into the network.
Common choices include ReLU, sigmoid, and tanh.
ReLU is defined as f(x) = max(0, x).

| Function | Range    | Common Use        |
|----------|----------|-------------------|
| ReLU     | [0, ∞)   | Hidden layers     |
| Sigmoid  | (0, 1)   | Binary output     |
| Softmax  | (0, 1)   | Multi-class output|
"""


def run():
    print("=== RAG Pipeline Smoke Test ===\n")

    # 1. Chunking
    print("1. Chunking...")
    doc = Document(content=SAMPLE_TEXT, source="sample.md")
    chunker = StructureAwareChunker(chunk_size=256)
    chunks = chunker.chunk(doc)
    print(f"   Produced {len(chunks)} chunks")
    for c in chunks:
        print(f"   [{c.chunk_type}] heading={c.heading!r} | {c.content[:60]}...")

    # 2. Embedding
    print("\n2. Embedding...")
    embedder = Embedder()
    vector_store = VectorStore(embedder)
    vector_store.add(chunks)

    # 3. BM25
    print("\n3. BM25 indexing...")
    bm25_store = BM25Store()
    bm25_store.build(chunks)

    # 4. Hybrid retrieval
    print("\n4. Hybrid retrieval...")
    retriever = HybridRetriever(vector_store, bm25_store, embedder, query_rewriter=None)
    results = retriever.retrieve("how does backpropagation work", top_k=5, use_query_rewriting=False)
    print(f"   Retrieved {len(results)} candidates")
    for r in results:
        print(f"   RRF={r.rrf_score:.4f} | {r.chunk.content[:60]}...")

    # 5. Re-ranking
    print("\n5. Re-ranking...")
    reranker = Reranker()
    reranked = reranker.rerank("how does backpropagation work", results, top_k=3)
    print(f"   Top {len(reranked)} after re-ranking:")
    for r in reranked:
        print(f"   Score={r.rerank_score:.4f} | {r.chunk.content[:60]}...")

    # 6. Context compression
    print("\n6. Context compression...")
    compressor = ContextCompressor()
    compressed = compressor.compress_all("how does backpropagation work", reranked)
    for r in compressed:
        print(f"   Compressed: {r.chunk.content[:80]}...")

    print("\n✅ Smoke test passed!")


if __name__ == "__main__":
    run()
