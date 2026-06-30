"""
RAG Ingestion Speed Benchmark
Measures each stage independently so you can see exactly where time is spent.
Run with:  python benchmark.py
"""
import sys, time, statistics
sys.stdout.reconfigure(encoding="utf-8")

from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage

# -- sample document (realistic size, tripled to simulate a real doc) ----------
SAMPLE_TEXT = """
# Introduction to Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) is a hybrid AI framework that combines the strengths
of large language models with external knowledge retrieval systems.

## How RAG Works

The RAG pipeline consists of two main phases: indexing and retrieval. During indexing,
documents are split into chunks, embedded into vector representations, and stored in a
vector database. During retrieval, a user query is embedded, similar chunks are fetched,
and the LLM generates an answer grounded in that context.

## Why RAG Matters

Traditional LLMs are limited by their training data cutoff and cannot access private or
real-time information. RAG solves this by providing dynamic context injection at inference
time, making the model responses accurate and up-to-date without retraining.

## Vector Databases

Vector databases like Qdrant, Weaviate, and Pinecone store embeddings and support
approximate nearest-neighbor search. They are the backbone of any production RAG system.

## Chunking Strategies

Effective chunking is critical. Chunks that are too large lose precision; chunks that are
too small lose context. Typical strategies include fixed-size chunking, sentence-level
splitting, and structure-aware chunking based on headings and paragraphs.

## Hybrid Search

Combining dense (vector) search with sparse (BM25) search via Reciprocal Rank Fusion
consistently outperforms either method alone, especially for domain-specific terminology.

## Reranking

Cross-encoder rerankers like ms-marco re-score retrieved candidates with full attention
over query and passage pairs. This significantly improves precision at the cost of latency.

## Evaluation

RAGAS provides automated evaluation metrics including faithfulness, answer relevancy,
and context precision. These metrics are essential for measuring RAG quality objectively.
""" * 3


# -- helpers ------------------------------------------------------------------
def fmt(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.0f} ms"
    return f"{seconds:.2f} s"

def bar(label: str, value: float, max_val: float):
    width  = 38
    filled = int((value / max_val) * width) if max_val > 0 else 0
    b      = "#" * filled + "-" * (width - filled)
    print(f"  {label:<33} [{b}] {fmt(value)}")


# -- Stage 1: Chunking --------------------------------------------------------
def benchmark_chunking():
    import re

    def parse(text):
        return re.sub(r'\n{3,}', '\n\n', text).split('\n')

    def chunk(lines):
        chunks, block, heading, in_table = [], [], "", False
        def save():
            if block:
                content = "\n".join(block).strip()
                if len(content) > 30:
                    chunks.append({"content": content, "heading": heading})
                block.clear()
        for line in lines:
            if re.match(r'^\s*\|.*\|\s*$', line):
                in_table = True; block.append(line); continue
            elif in_table:
                in_table = False; save()
            if re.match(r"^#{1,6}\s", line) or re.match(r"^[A-Z][A-Z\s]{3,}:?\s*$", line):
                save(); heading = line.strip()
            else:
                block.append(line)
                if sum(len(w) for w in block) > 400 * 5:
                    save()
        save()
        return chunks

    t0     = time.perf_counter()
    chunks = chunk(parse(SAMPLE_TEXT))
    return time.perf_counter() - t0, len(chunks), chunks


# -- Stage 2: Embedding -------------------------------------------------------
def benchmark_embedding(chunks: list, batch_size: int = 256):
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cpu"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    elif torch.cuda.is_available():
        device = "cuda"

    print(f"  Loading embedder on {device}...")
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    texts = [c["content"] for c in chunks]

    t0 = time.perf_counter()
    model.encode(texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    return time.perf_counter() - t0, device


# -- Stage 3: Metadata LLM comparison ----------------------------------------
def benchmark_metadata(model_name: str, chunks: list, n: int = 5):
    llm    = ChatOllama(model=model_name, temperature=0, num_predict=300)
    times  = []

    print(f"  Warming up {model_name}...")
    try:
        llm.invoke([HumanMessage(content='Return JSON: {"ok": true}')])
    except Exception as e:
        return None, str(e)

    for c in chunks[:n]:
        prompt = (
            f"Analyze this text (Heading: {c['heading']}):\n{c['content']}\n\n"
            'Return ONLY valid JSON: {"summary": "1 sentence", '
            '"keywords": ["k1","k2"], "questions": ["Q1?","Q2?"]}'
        )
        t0 = time.perf_counter()
        try:
            llm.invoke([HumanMessage(content=prompt)])
            times.append(time.perf_counter() - t0)
        except Exception as e:
            print(f"    Error: {e}")

    return (times, None) if times else (None, "all calls failed")


# -- Main ---------------------------------------------------------------------
def main():
    SEP = "=" * 62

    print(SEP)
    print("  RAG INGESTION SPEED BENCHMARK")
    print(SEP)

    # Stage 1
    print("\n[Stage 1] Document Parsing & Chunking")
    t_chunk, n_chunks, chunks = benchmark_chunking()
    print(f"  {n_chunks} chunks created in {fmt(t_chunk)}")

    # Stage 2
    print("\n[Stage 2] Embedding  (BAAI/bge-m3, batch=256)")
    t_embed, device = benchmark_embedding(chunks)
    print(f"  Device   : {device.upper()}")
    print(f"  Total    : {fmt(t_embed)} for {n_chunks} chunks")
    print(f"  Per chunk: {fmt(t_embed / n_chunks)}")

    # Stage 3
    N = min(5, n_chunks)
    print(f"\n[Stage 3] Metadata Generation  (testing {N}/{n_chunks} chunks, extrapolated)")
    print()

    results = {}
    for model in ["llama3.2:3b", "qwen2.5:0.5b"]:
        print(f"  -- {model} --")
        times, err = benchmark_metadata(model, chunks, n=N)
        if err:
            print(f"  SKIP: {err}\n")
            continue
        avg   = statistics.mean(times)
        total = avg * n_chunks
        results[model] = {"avg": avg, "total": total}
        print(f"  Avg per chunk     : {fmt(avg)}")
        print(f"  Full doc (seq.)   : {fmt(total)}")
        print(f"  Full doc (x6 par.): {fmt(total / 6)}")
        print()

    # Summary
    print(SEP)
    print("  SUMMARY  (bar chart)")
    print(SEP)

    max_val = max([r["total"] for r in results.values()] + [t_embed]) if results else t_embed

    print("\n  Chunking")
    bar("parse + split", t_chunk, max_val)

    print(f"\n  Embedding ({device.upper()})")
    bar(f"bge-m3  {n_chunks} chunks", t_embed, max_val)

    if results:
        print(f"\n  Metadata ({n_chunks} chunks, extrapolated)")
        for m, r in results.items():
            bar(f"{m}  sequential", r["total"], max_val)
            bar(f"{m}  x6 parallel", r["total"] / 6, max_val)

    if len(results) == 2:
        speedup = results["llama3.2:3b"]["avg"] / results["qwen2.5:0.5b"]["avg"]
        print(f"\n  >> qwen2.5:0.5b is {speedup:.1f}x faster per chunk than llama3.2:3b")

    no_meta = t_chunk + t_embed
    print(f"\n  >> GENERATE_METADATA=False  -> {fmt(no_meta)} total (chunking + embedding only)")
    if results:
        best_m, best_r = min(results.items(), key=lambda x: x[1]["total"])
        print(f"  >> Best with metadata       -> {fmt(best_r['total']/6 + t_embed)} total ({best_m} x6 parallel)")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
