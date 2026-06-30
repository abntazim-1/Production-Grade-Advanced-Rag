# ============================================================
# MINI PRODUCTION RAG — single file
# pip install sentence-transformers qdrant-client rank-bm25
#             langchain-ollama langchain-core langgraph ragas fastapi uvicorn
#             sse-starlette python-dotenv datasets langchain-huggingface
# ============================================================
import os, re, json, asyncio, uuid, threading, time, functools, math
import atexit
import torch                                     # MINOR 1 FIX — moved to top with all imports
from collections import deque
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
import concurrent.futures as _cf
import sys
import types

# ─── RAGAS PATCH FOR LANGCHAIN 0.4+ COMPATIBILITY ──────────────────────────
try:
    import langchain_community.chat_models
    if not hasattr(langchain_community.chat_models, "vertexai"):
        dummy = types.ModuleType("langchain_community.chat_models.vertexai")
        dummy.ChatVertexAI = type("ChatVertexAI", (object,), {})
        sys.modules["langchain_community.chat_models.vertexai"] = dummy
        langchain_community.chat_models.vertexai = dummy
except Exception:
    pass

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance,
    Filter, FieldCondition, MatchValue
)

load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────
EMBED_MODEL    = "BAAI/bge-m3"
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL      = "llama3.2:3b"    # main query model — quality + speed balance
METADATA_MODEL = "qwen2.5:0.5b"  # tiny fast model only for metadata JSON
CHUNK_SIZE        = 400
TOP_K             = 25    # increased to 25 to cast a wider net for hybrid fusion
RERANK_TOP_K      = 5     # keep top 5 chunks for LLM context
RRF_K             = 60
MEMORY_MAXLEN     = 50    # max turns per session before eviction
MAX_CONTEXT_CHARS = 12000 # Increased to allow full chunks to reach the LLM

# ─── ENTERPRISE LATENCY KNOBS ────────────────────────────────
# To achieve < 10ms retrieval for new queries, disable the reranker.
# Reranking uses a heavy CrossEncoder which takes ~100ms.
USE_RERANKER   = True
# Caching reduces repeated query retrieval to 0.00ms.
CACHE_SIZE     = 1024

# ─── PERFORMANCE KNOBS (tuned for Intel Arc A750 8GB VRAM + 24GB RAM) ─────────
# Set to True to generate LLM-based summaries/keywords/questions per chunk.
# WARNING: This calls the LLM once per chunk and is the #1 ingestion bottleneck.
# Leave False for fast ingestion; enable only if you need enriched metadata.
GENERATE_METADATA   = True   # enabled — using fast qwen2.5:0.5b, not the main LLM

# Parallel LLM threads for metadata generation.
# llama3.2:3b (Q4) ≈ 2GB RAM per instance. With 24GB RAM you can safely run 6.
# IMPORTANT: also set OLLAMA_NUM_PARALLEL=6 env var before starting Ollama,
# otherwise Ollama will still serialize requests regardless of worker count.
# PowerShell: $env:OLLAMA_NUM_PARALLEL = "6"; ollama serve
METADATA_WORKERS    = 6

# Embedding batch size.
# bge-m3 ≈ 1.5GB VRAM. Arc A750 has 8GB → plenty of headroom.
# 256 gives maximum GPU throughput without OOM on your hardware.
EMBED_BATCH_SIZE    = 256

# num_predict=512 caps generation at ~512 tokens (~10s max on Arc A750).
# num_ctx=2048  matches MAX_CONTEXT_CHARS; prevents Ollama from pre-allocating an
#               8192-token KV cache which adds 1-3s of overhead before the first token.
# num_gpu=-1   tells Ollama to offload ALL layers to GPU.
# keep_alive=-1 keeps model pinned in VRAM permanently — eliminates 5-15s cold starts.
llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
    num_predict=512,
    num_ctx=4096,     # Increased to accommodate more chunks (MAX_CONTEXT_CHARS)
    num_gpu=-1,
    keep_alive=-1,
)

# Separate tiny LLM just for metadata generation during ingestion.
# qwen2.5:0.5b is ~400MB, runs 2-3x faster than 3b, and reliably outputs JSON.
# num_predict=300 caps output length so it never over-generates.
# Pull it once with: ollama pull qwen2.5:0.5b
metadata_llm = ChatOllama(
    model=METADATA_MODEL,
    temperature=0,
    num_predict=300,
    num_ctx=1024,
    num_gpu=-1,
    keep_alive=-1,
)

# ─── MODELS ──────────────────────────────────────────────────
@dataclass
class Chunk:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    source: str = ""
    heading: str = ""
    keywords: list = field(default_factory=list)
    summary: str = ""
    questions: list = field(default_factory=list)

@dataclass
class Hit:
    chunk: Chunk
    rrf_score: float = 0.0
    rerank_score: float = 0.0

# ─── DATA PROCESSING PIPELINE ────────────────────────────────
# 1. Re-Structuring Data
def parse_document(text: str) -> list[str]:
    """Document Parser & Structure Analyzer"""
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.split('\n')

# 2. Structure-Aware Chunking
def structure_aware_chunking(lines: list[str], source: str = "") -> list[Chunk]:
    """Table Preserver, Heading Detector, Boundary Detector"""
    chunks, current_block = [], []
    heading = ""
    in_table = False

    def save_chunk(overlap: bool = False):
        """Save current_block as a Chunk. If overlap=True, keep the last
        CHUNK_OVERLAP lines as the start of the next chunk (size-triggered splits).
        Hard semantic boundaries (headings, table ends) always clear fully.
        """
        if current_block:
            content = "\n".join(current_block).strip()
            if len(content) > 30:
                chunks.append(Chunk(content=content, source=source, heading=heading))
            if overlap and len(current_block) > CHUNK_OVERLAP:
                kept = current_block[-CHUNK_OVERLAP:]
                current_block.clear()
                current_block.extend(kept)
            else:
                current_block.clear()

    for line in lines:
        is_table_row = bool(re.match(r'^\s*\|.*\|\s*$', line))

        if is_table_row:
            in_table = True
            current_block.append(line)
            continue
        elif in_table:
            in_table = False
            save_chunk()  # hard boundary — no overlap across table end

        if re.match(r"^#{1,6}\s", line) or re.match(r"^[A-Z][A-Z\s]{3,}:?\s*$", line):
            save_chunk()  # hard boundary at headings — no overlap
            heading = line.strip()
        else:
            current_block.append(line)
            if sum(len(w) for w in current_block) > CHUNK_SIZE * 5:
                save_chunk(overlap=True)  # size-triggered — carry overlap

    save_chunk()
    return chunks

# 3. Metadata Creation
def generate_metadata_for_chunk(chunk: Chunk):
    """Summary Generator, Keyword Extractor, Question Generator"""
    prompt = f'''Analyze this text chunk (Heading: {chunk.heading}):\n{chunk.content}\n\nReturn ONLY a JSON object with this exact structure:\n{{"summary": "A 1-sentence summary", "keywords": ["keyword1", "keyword2"], "questions": ["Question 1?", "Question 2?"]}}'''
    try:
        res = metadata_llm.invoke([HumanMessage(content=prompt)]).content
        start = res.find('{')
        end = res.rfind('}') + 1
        if start != -1 and end != -1:
            data = json.loads(res[start:end])
            chunk.summary   = data.get("summary", "")
            chunk.keywords  = data.get("keywords", [])
            chunk.questions = data.get("questions", [])
    except Exception as e:
        print(f"Metadata generation failed for chunk '{chunk.heading}': {e}")

def process_document(text: str, source: str = "") -> list[Chunk]:
    lines  = parse_document(text)
    chunks = structure_aware_chunking(lines, source)
    if GENERATE_METADATA:
        print(f"Generating metadata for {len(chunks)} chunks (this may take a while)...")
        with ThreadPoolExecutor(max_workers=METADATA_WORKERS) as executor:
            list(executor.map(generate_metadata_for_chunk, chunks))
    else:
        # PERF 4 FIX — no emoji in print; emoji crashes Windows CP1252 terminals
        print(f"Skipping LLM metadata - fast ingestion mode. "
              f"({len(chunks)} chunks, set GENERATE_METADATA=True to enable enrichment)")
    return chunks

# ─── EMBEDDER ────────────────────────────────────────────────
# Device priority: Intel Arc XPU → NVIDIA CUDA → CPU

def _best_device() -> str:
    # ── Intel Arc / Intel GPU (XPU) ──────────────────────────
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        dev_name = torch.xpu.get_device_name(0) if hasattr(torch.xpu, "get_device_name") else "Intel GPU"
        print(f"[GPU] Intel Arc XPU detected: {dev_name} - using GPU for embeddings!")
        return "xpu"

    # ── NVIDIA CUDA ───────────────────────────────────────────
    if torch.cuda.is_available():
        print(f"[GPU] CUDA GPU detected: {torch.cuda.get_device_name(0)} - using GPU for embeddings!")
        return "cuda"

    # ── Diagnostic: explain WHY GPU was not found ─────────────
    print("[CPU] No GPU detected - falling back to CPU for embeddings.")
    print("      -> Ensure torch+xpu or torch+cu is installed.")
    return "cpu"

_DEVICE = _best_device()
print(f"Loading embedder on {_DEVICE}..."); embedder = SentenceTransformer(EMBED_MODEL, device=_DEVICE)
print(f"Loading reranker on {_DEVICE}..."); reranker = CrossEncoder(RERANK_MODEL, max_length=512, device=_DEVICE)

print("Warming up models to eliminate first-query cold start...")
_ = embedder.encode(["warmup"], batch_size=1, show_progress_bar=False)
_ = reranker.predict([("warmup", "warmup")])

def embed(texts: list[str]) -> np.ndarray:
    return np.array(
        embedder.encode(texts, normalize_embeddings=True, batch_size=EMBED_BATCH_SIZE,
                        show_progress_bar=False),
        dtype=np.float32
    )

# ─── QDRANT VECTOR STORE ─────────────────────────────────────
dim    = embedder.get_sentence_embedding_dimension()
qdrant = QdrantClient(path="qdrant_db")   # persistent local storage
atexit.register(qdrant.close)
COLLECTION_NAME = "chunks"

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

# Lock protects all_chunks and _chunk_index from concurrent read/write.
_chunks_lock: threading.Lock = threading.Lock()
all_chunks: list[Chunk] = []
# BUG 5 FIX — O(1) dict index replaces O(n) linear scan in get_chunk().
_chunk_index: dict[str, Chunk] = {}

# FIX [BUG 2] — Restore persisted chunks from Qdrant into RAM on startup so
# BM25 and get_chunk() work correctly after a server restart.
def load_chunks_from_qdrant():
    """Scroll only the 'content' vectors so we don't create duplicate Chunk objects."""
    offset = None
    restored = 0
    while True:
        records, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
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
                content   = p["content"],
                source    = p.get("source", ""),
                heading   = p.get("heading", ""),
                summary   = p.get("summary", ""),
                # Coerce list items to str — Qdrant can return dicts for list payloads
                keywords  = [str(k) for k in p.get("keywords", [])],
                questions = [str(q) for q in p.get("questions", [])],
            )
            all_chunks.append(chunk)
            _chunk_index[chunk.id] = chunk   # BUG 5 FIX
            restored += 1
        if next_offset is None:
            break
        offset = next_offset
    if restored:
        print(f"[OK] Restored {restored} chunks from Qdrant on startup.")

load_chunks_from_qdrant()

# FIX [BUG 3] — Track ingested sources to prevent duplicate indexing.
_ingested_sources: set[str] = {c.source for c in all_chunks}

def add_chunks(chunks: list[Chunk]):
    if not chunks: return
    points = []
    for c in chunks:
        # Build the text list for multi-vector embedding
        texts_to_embed = [c.content]
        if c.summary:    texts_to_embed.append(c.summary)
        if c.questions:  texts_to_embed.extend(c.questions)

        vecs = embed(texts_to_embed)

        base_payload = {
            "chunk_id" : c.id,
            "content"  : c.content,
            "source"   : c.source,
            "heading"  : c.heading,
            "summary"  : c.summary,
            "keywords" : c.keywords,
            "questions": c.questions,
        }

        # Assign each vector a deterministic type label
        vec_types = ["content"]
        if c.summary:    vec_types.append("summary")
        if c.questions:  vec_types.extend(["question"] * len(c.questions))

        for vec_type, vec in zip(vec_types, vecs):
            pt_id   = c.id if vec_type == "content" else str(uuid.uuid4())
            payload = {**base_payload, "vector_type": vec_type}
            points.append(PointStruct(id=pt_id, vector=vec.tolist(), payload=payload))

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    with _chunks_lock:
        all_chunks.extend(chunks)
        for c in chunks:
            _chunk_index[c.id] = c   # BUG 5 FIX — keep index in sync

    cached_retrieve_and_rerank.cache_clear()

    print(f"Indexed {len(chunks)} chunks ({len(points)} total vectors) to Qdrant.")

def dense_search(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
    q_vec = embed([query])[0]
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec.tolist(),
        limit=k * 3,   # over-fetch because many vectors map to one chunk
    )
    results = response.points
    # Deduplicate: keep the best score per logical chunk_id
    best: dict[str, float] = {}
    for res in results:
        cid = res.payload["chunk_id"]
        if cid not in best or res.score > best[cid]:
            best[cid] = res.score
    return sorted(best.items(), key=lambda x: -x[1])[:k]

# ─── BM25 ────────────────────────────────────────────────────
bm25: BM25Okapi | None = None
STOP = {"the","a","an","is","it","in","on","at","to","for","of","and","or"}

def tokenize(t: str) -> list[str]:
    return [w for w in re.findall(r"\b\w+\b", t.lower()) if w not in STOP]

def build_bm25():
    global bm25
    with _chunks_lock:
        snapshot = list(all_chunks)
    if not snapshot: return
    bm25 = BM25Okapi([
        tokenize(
            c.content + " " + " ".join(str(k) for k in c.keywords) + " " +
            c.summary  + " " + " ".join(str(q) for q in c.questions)
        )
        for c in snapshot
    ])

# Rebuild BM25 from restored chunks on startup
build_bm25()

def sparse_search(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
    if not bm25: return []
    with _chunks_lock:
        snapshot = list(all_chunks)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    return [(snapshot[i].id, float(s)) for i, s in ranked[:k] if s > 0]

# ─── RRF ─────────────────────────────────────────────────────
def rrf(lists: list[list[tuple[str, float]]], k: int = TOP_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for lst in lists:
        for rank, (cid, _) in enumerate(lst):
            scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])[:k]

def get_chunk(cid: str) -> Chunk | None:
    # BUG 5 FIX — O(1) dict lookup; no lock needed (dict reads are thread-safe in CPython)
    return _chunk_index.get(str(cid))

# ─── HYBRID RETRIEVE ─────────────────────────────────────────
def retrieve(query: str) -> list[Hit]:
    dense  = dense_search(query)
    sparse = sparse_search(query)
    fused  = rrf([dense, sparse])
    hits   = []
    for cid, score in fused:
        c = get_chunk(cid)
        if c: hits.append(Hit(chunk=c, rrf_score=score))
    return hits

# ─── RERANKER ────────────────────────────────────────────────
def rerank(query: str, hits: list[Hit]) -> list[Hit]:
    if not hits or not USE_RERANKER:
        return hits[:RERANK_TOP_K]
    pairs  = [(query, h.chunk.content) for h in hits]
    scores = reranker.predict(pairs)
    for h, s in zip(hits, scores):
        h.rerank_score = float(s)
    ranked = sorted(hits, key=lambda x: -x.rerank_score)
    # Filter chunks with negative reranker scores (cross-encoder says: not relevant).
    # Keep negatives only if ALL chunks score negative (guarantees >= 1 result).
    top    = ranked[:RERANK_TOP_K]
    positive = [h for h in top if h.rerank_score >= 0]
    return positive if positive else top[:1]

@functools.lru_cache(maxsize=CACHE_SIZE)
def cached_retrieve_and_rerank(query: str) -> tuple[Hit, ...]:
    """BUG 2 FIX — returns an immutable tuple so callers cannot mutate the
    cached value and corrupt future cache hits. Convert with list() at call sites."""
    hits = retrieve(query)
    return tuple(rerank(query, hits))

# ─── CONTEXT COMPRESS ────────────────────────────────────────
def compress(original_query: str, rewritten_query: str, content: str) -> str:
    # ENTERPRISE FIX: Naive keyword sentence filtering destroys semantic context. 
    # Since we are using a powerful CrossEncoder reranker to select the best chunks,
    # we should pass the full chunk unmodified to the LLM to preserve accuracy.
    return content.strip()

# ─── MEMORY ──────────────────────────────────────────────────
# FIX [ISSUE 3] — Use a bounded deque (maxlen) per session to prevent unbounded
# memory growth. Old turns are automatically evicted when the limit is exceeded.
_memory_lock: threading.Lock = threading.Lock()
memory: dict[str, deque] = {}

def get_history(sid: str) -> str:
    with _memory_lock:
        turns = list(memory.get(sid, deque()))[-6:]
    return "\n".join(f"{t['role'].title()}: {t['content']}" for t in turns)

def save_turn(sid: str, role: str, content: str):
    with _memory_lock:
        if sid not in memory:
            memory[sid] = deque(maxlen=MEMORY_MAXLEN)
        memory[sid].append({"role": role, "content": content})

# ─── GENERATE ────────────────────────────────────────────────
# Concise directive prompt — shorter prefill = faster TTFT, more focused answer.
_SYSTEM_PROMPT = (
    "You are a professional enterprise AI assistant. "
    "Answer the user's question using ONLY the provided context below. "
    "Be concise, highly accurate, and factual. "
    "Always cite your sources using the provided markers (e.g., [1], [2]). "
    "If the answer is not contained in the context, explicitly state 'I do not have enough information in the context to answer this.' Do not guess."
)

def _build_context(original_q: str, rewritten_q: str, hits: list[Hit]) -> str:
    """Build a context string capped at MAX_CONTEXT_CHARS to bound LLM input size."""
    parts = []
    total = 0
    for i, h in enumerate(hits):
        snippet = compress(original_q, rewritten_q, h.chunk.content)
        entry   = f"[{i+1}] {snippet}"
        if total + len(entry) > MAX_CONTEXT_CHARS:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts)

def _build_user_prompt(context: str, question: str) -> str:
    """Build the user message.
    History is intentionally excluded from the generation prompt to prevent
    answer drift and to keep prefill tokens minimal (faster TTFT).
    History is used ONLY by rewrite_query for pronoun/co-reference resolution.
    """
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

def generate(original_q: str, rewritten_q: str, hits: list[Hit], history: str) -> str:
    """Blocking full-response generation (used by /query REST endpoint)."""
    context = _build_context(original_q, rewritten_q, hits)
    user    = _build_user_prompt(context, rewritten_q)
    return llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user)]).content.strip()

def generate_stream(original_q: str, rewritten_q: str, hits: list[Hit], history: str):
    """True token streaming. History excluded from prompt — prevents drift and
    reduces prefill tokens for faster TTFT. Yields one token string at a time.
    """
    context = _build_context(original_q, rewritten_q, hits)
    user    = _build_user_prompt(context, rewritten_q)
    for chunk in llm.stream([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user)]):
        if chunk.content:
            yield chunk.content

# ─── LANGGRAPH ───────────────────────────────────────────────
class State(TypedDict):
    query: str; session_id: str
    rewritten_query: str
    hits: list[Hit]; answer: str; sources: list[dict]
    timings: dict

def rewrite_query(query: str, history: str) -> str:
    prompt = (
        "Given the chat history and the latest user query, rewrite the query to be "
        "a standalone, search-optimized query. If it is already optimal, return it as is. "
        "ONLY output the rewritten query.\n\n"
        f"History:\n{history}\n\nQuery: {query}\n\nRewritten:"
    )
    return metadata_llm.invoke([HumanMessage(content=prompt)]).content.strip()

def node_rewrite(s: State) -> State:
    """PHASE 2 — Parallel rewrite + prefetch.
    On first-turn queries (no history) rewriting is skipped entirely.
    On multi-turn queries, rewrite (qwen2.5:0.5b) and dense prefetch (BGE-M3 embed)
    run simultaneously so neither blocks the other.
    """
    t0      = time.time()
    history = get_history(s["session_id"])
    query   = s["query"]

    if not history.strip():
        # No history → no rewrite needed, skip LLM call entirely.
        s["rewritten_query"] = query
    else:
        # Fire rewrite and dense-embed in parallel — they are independent.
        with _cf.ThreadPoolExecutor(max_workers=2) as ex:
            rewrite_fut = ex.submit(rewrite_query, query, history)
            # Pre-embed the original query while rewrite is running.
            # If rewrite changes the query, we re-embed below (cache handles repeats).
            embed_fut   = ex.submit(embed, [query])
            rewritten        = rewrite_fut.result()
            _                = embed_fut.result()   # warm the embedding cache
        s["rewritten_query"] = rewritten

    s.setdefault("timings", {})["rewrite"] = round(time.time() - t0, 2)
    return s

def node_retrieve(s: State) -> State:
    t0 = time.time()
    q  = s.get("rewritten_query") or s["query"]
    # BUG 2 FIX — convert immutable tuple back to list for local mutation safety
    s["hits"] = list(cached_retrieve_and_rerank(q))
    s.setdefault("timings", {})["retrieve"] = round(time.time() - t0, 3)
    return s

def node_generate(s: State) -> State:
    t0 = time.time()
    history    = get_history(s["session_id"])
    original_q = s["query"]
    rewritten_q = s.get("rewritten_query") or original_q
    answer = generate(original_q, rewritten_q, s["hits"], history)

    s["answer"] = answer
    s["sources"] = [
        {
            "id": i + 1,
            "source": h.chunk.source,
            "heading": h.chunk.heading,
            "score": round(h.rerank_score if USE_RERANKER else h.rrf_score, 3),
            "content": h.chunk.content,
        }
        for i, h in enumerate(s["hits"])
    ]
    save_turn(s["session_id"], "user",      original_q)
    save_turn(s["session_id"], "assistant", s["answer"])
    s.setdefault("timings", {})["generate"] = round(time.time() - t0, 2)
    return s

graph = StateGraph(State)
graph.add_node("rewrite",  node_rewrite)
graph.add_node("retrieve", node_retrieve)
graph.add_node("generate", node_generate)
graph.set_entry_point("rewrite")
graph.add_edge("rewrite",  "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
rag = graph.compile()

def run(query: str, session_id: str = "default") -> dict:
    t0 = time.time()
    result = rag.invoke({
        "query": query, "session_id": session_id,
        "rewritten_query": "",
        "hits": [], "answer": "", "sources": [],
        "timings": {}
    })
    total_time = round(time.time() - t0, 2)
    timings = result.get("timings", {})
    timings["total"] = total_time
    return {
        "answer": result["answer"], 
        "sources": result["sources"], 
        "timings": timings,
        "rewritten_query": result.get("rewritten_query", "")
    }

# ─── EVALUATION (RAGAS) ────────────────────────────────────────────────────────────
def evaluate_rag(questions: list[str], ground_truths: list[str] | None = None) -> dict:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics.collections import answer_relevancy, context_precision, faithfulness
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Import error: {e}"}

    print(f"Evaluating {len(questions)} queries with RAGAS...")
    ragas_emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    data: dict[str, list] = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for i, q in enumerate(questions):
        res = run(q, session_id=f"eval_{uuid.uuid4()}")
        data["question"].append(q)
        data["answer"].append(res["answer"])
        data["contexts"].append([s["content"] for s in res["sources"]])
        data["ground_truth"].append(
            ground_truths[i] if ground_truths and i < len(ground_truths) else ""
        )

    dataset = Dataset.from_dict(data)
    metrics = [answer_relevancy, faithfulness]
    if ground_truths and len(ground_truths) == len(questions):
        metrics.append(context_precision)

    results = evaluate(dataset, metrics=metrics, llm=llm, embeddings=ragas_emb)
    return dict(results)

def evaluate_single_response(question: str, answer: str, contexts: list[str]) -> dict:
    """Evaluates a single chat turn without re-running the retrieval pipeline."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics.collections import answer_relevancy, faithfulness
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Import error: {e}"}

    ragas_emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    data = {"question": [question], "answer": [answer], "contexts": [contexts]}
    dataset = Dataset.from_dict(data)
    
    results = evaluate(dataset, metrics=[answer_relevancy, faithfulness], llm=llm, embeddings=ragas_emb)
    return dict(results)

# ─── FASTAPI ─────────────────────────────────────────────────
app = FastAPI(title="Mini RAG")

class IngestReq(BaseModel):
    text: str; source: str = "manual"

class QueryReq(BaseModel):
    query: str; session_id: str = "default"

class EvalReq(BaseModel):
    questions: list[str]
    ground_truths: list[str] = []

@app.post("/ingest")
def ingest(req: IngestReq):
    # FIX [BUG 3] — Reject re-ingestion of the same source to prevent duplicate vectors.
    if req.source in _ingested_sources:
        raise HTTPException(
            status_code=409,
            detail=f"Source '{req.source}' is already indexed. Delete it first or use a unique source name."
        )
    try:
        chunks = process_document(req.text, req.source)
        add_chunks(chunks)
        build_bm25()
        _ingested_sources.add(req.source)
        return {
            "status": "ok",
            "source": req.source,
            "chunks_created": len(chunks),
            "total_chunks": len(all_chunks),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_endpoint(req: QueryReq):
    try:
        return run(req.query, req.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_stream(req: QueryReq):
    """True token streaming over SSE.
    - Retrieval runs in a thread pool (non-blocking for the event loop).
    - LLM generation also runs in a thread pool via an asyncio.Queue bridge so
      the synchronous generate_stream() never blocks the uvicorn event loop.
    """
    loop = asyncio.get_running_loop()  # get_event_loop() is deprecated in 3.10+

    # ── Step 1: rewrite + retrieval in a thread (blocking IO, ~100-300ms)
    def _retrieve_only(query: str, session_id: str):
        history = get_history(session_id)
        rewritten = query if not history.strip() else rewrite_query(query, history)
        hits = list(cached_retrieve_and_rerank(rewritten))
        return rewritten, hits

    rewritten_q, hits = await loop.run_in_executor(
        None, _retrieve_only, req.query, req.session_id
    )

    # ── Step 2: source metadata for the final SSE frame
    sources = [
        {
            "id": i + 1,
            "source": h.chunk.source,
            "heading": h.chunk.heading,
            "score": round(h.rerank_score if USE_RERANKER else h.rrf_score, 3),
            "content": h.chunk.content,
        }
        for i, h in enumerate(hits)
    ]

    # ── Step 3: bridge sync generator → async via a Queue
    # Running generate_stream() directly in an async for-loop blocks the event loop
    # for ALL other clients. Instead, push tokens from a thread into an asyncio.Queue
    # and drain it asynchronously — the event loop stays free between tokens.
    q: asyncio.Queue = asyncio.Queue(maxsize=32)
    _DONE = object()  # sentinel

    def _stream_to_queue():
        try:
            for token in generate_stream(req.query, rewritten_q, hits, ""):
                loop.call_soon_threadsafe(q.put_nowait, token)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, _DONE)

    full_answer_parts: list[str] = []

    async def gen():
        loop.run_in_executor(None, _stream_to_queue)
        while True:
            item = await q.get()
            if item is _DONE:
                break
            full_answer_parts.append(item)
            yield f"data: {json.dumps({'token': item})}\n\n"
        full_answer = "".join(full_answer_parts).strip()
        save_turn(req.session_id, "user",      req.query)
        save_turn(req.session_id, "assistant", full_answer)
        yield f"data: {json.dumps({'sources': sources, 'done': True})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/health")
def health():
    info = qdrant.get_collection(COLLECTION_NAME)
    return {
        "status": "ok",
        "chunks_in_memory": len(all_chunks),
        "vectors_in_qdrant": info.points_count,
        "active_sessions":   len(memory),
        "indexed_sources":   list(_ingested_sources),
    }

@app.delete("/source/{source_name}")
def delete_source(source_name: str):
    """Remove all vectors and in-memory chunks belonging to a source."""
    global bm25
    from qdrant_client.models import FilterSelector
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_name))])
        )
    )
    with _chunks_lock:
        before = len(all_chunks)
        all_chunks[:] = [c for c in all_chunks if c.source != source_name]
        removed = before - len(all_chunks)
        # Keep _chunk_index in sync with all_chunks after deletion
        for cid in [k for k, v in _chunk_index.items() if v.source == source_name]:
            del _chunk_index[cid]
    _ingested_sources.discard(source_name)
    build_bm25()
    cached_retrieve_and_rerank.cache_clear()
    return {"status": "ok", "source": source_name, "chunks_removed": removed}

@app.post("/evaluate")
def evaluate_endpoint(req: EvalReq):
    try:
        return evaluate_rag(req.questions, req.ground_truths or None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── GRADIO UI ───────────────────────────────────────────────
import os as _os
import gradio as gr

# ─── FILE TEXT EXTRACTOR ─────────────────────────────────────
def extract_text_from_file(file_path: str) -> tuple[str, str]:
    """Extract text content and derive a source name from the uploaded file."""
    filename    = os.path.basename(file_path)  # MINOR 2 FIX
    source_name = filename
    ext         = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    try:
        if ext in ("txt", "md", "rst", "log", "csv"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(), source_name

        elif ext == "pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
                return text, source_name
            except ImportError:
                return "", "__error__:Install pypdf: pip install pypdf"

        elif ext == "docx":
            try:
                import docx
                doc  = docx.Document(file_path)
                text = "\n".join(p.text for p in doc.paragraphs)
                return text, source_name
            except ImportError:
                return "", "__error__:Install python-docx: pip install python-docx"

        elif ext == "json":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                import json as _json
                data = _json.load(f)
                return _json.dumps(data, indent=2), source_name

        else:
            # Fall back: try reading as plain text
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(), source_name

    except Exception as e:
        return "", f"__error__:{e}"

SESSION_ID = f"gradio_session_{uuid.uuid4().hex[:8]}"

def ingest_file(files) -> str:
    """Handle one or more uploaded files, extract text, and index them."""
    if not files:
        return "⚠️ Please upload at least one file."

    results = []
    for file in (files if isinstance(files, list) else [files]):
        file_path = file.name if hasattr(file, "name") else str(file)
        text, source_name = extract_text_from_file(file_path)

        # Report extraction errors
        if source_name.startswith("__error__:"):
            results.append(f"Error: {os.path.basename(file_path)}: {source_name[10:]}")  # MINOR 2 FIX
            continue

        if not text.strip():
            results.append(f"⚠️ {source_name}: No text could be extracted.")
            continue

        # Skip already-indexed sources
        if source_name in _ingested_sources:
            results.append(f"⚠️ '{source_name}' is already indexed. Delete it first or rename the file.")
            continue

        try:
            chunks = process_document(text, source=source_name)
            add_chunks(chunks)
            # PERF 3 FIX: don't call build_bm25 here; called once after loop
            _ingested_sources.add(source_name)
            results.append(f"Indexed '{source_name}' -> {len(chunks)} chunks (total: {len(all_chunks)})")
        except Exception as e:
            results.append(f"Error '{source_name}': {e}")

    # PERF 3 FIX — rebuild BM25 once after all files are processed, not per-file
    build_bm25()
    return "\n".join(results)

# ─── RAGAS background helper ──────────────────────────────────────────────────────────
_ragas_pool = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="ragas")

def _run_ragas_background(question: str, answer: str, contexts: list[str]):
    """BUG 1 FIX — True fire-and-forget RAGAS. Never called from the Gradio
    generator; results are logged to console so no blocking occurs.
    CORRECT 2 NOTE: RAGAS currently uses the same LLM that generated the answer
    (circular evaluation). Replace with a dedicated judge model for production."""
    try:
        res = evaluate_single_response(question, answer, contexts)
        if "error" not in res:
            parts = []
            for k, v in res.items():
                try:
                    val = float(v)
                    parts.append(f"{k.replace('_', ' ').title()}: {val:.2f}")
                except (TypeError, ValueError):
                    pass
            print(f"[RAGAS] {' | '.join(parts)}")
        else:
            print(f"[RAGAS] Skipped: {res['error']}")
    except Exception as e:
        print(f"[RAGAS] Error: {e}")

def chat_fn(message: str, history: list, session_id: str):
    """Streaming chat with background RAGAS evaluation.
    BUG 4 FIX — session_id is passed in from gr.State, one per browser tab.
    """
    try:
        t0          = time.time()
        history_str = get_history(session_id)
        if not history_str.strip():
            rewritten_q = message
            rewrite_ms  = 0.0
        else:
            t_rw        = time.time()
            rewritten_q = rewrite_query(message, history_str)
            rewrite_ms  = round(time.time() - t_rw, 2)

        t_ret  = time.time()
        hits   = list(cached_retrieve_and_rerank(rewritten_q))  # BUG 2 FIX
        ret_ms = round(time.time() - t_ret, 3)

        sources = [
            {
                "id": i + 1,
                "source": h.chunk.source,
                "heading": h.chunk.heading,
                "score": round(h.rerank_score if USE_RERANKER else h.rrf_score, 3),
                "content": h.chunk.content,
            }
            for i, h in enumerate(hits)
        ]

        # ── Step 2: stream tokens live
        answer_parts: list[str] = []
        partial = ""
        ttft_ms = None
        t_stream_start = time.time()
        for token in generate_stream(message, rewritten_q, hits, history_str):
            if ttft_ms is None:
                ttft_ms = round(time.time() - t_stream_start, 3)
            answer_parts.append(token)
            partial = "".join(answer_parts)
            yield partial
            
        stream_total_ms = round(time.time() - t_stream_start, 3)
        if ttft_ms is None:
            ttft_ms = 0.0

        answer = partial.strip()
        gen_ms = round(time.time() - t0, 2)

        # CORRECT 3 FIX — guard against empty LLM response
        if not answer:
            yield "The model returned an empty response. Please try again."
            return

        save_turn(session_id, "user",      message)   # BUG 4 FIX
        save_turn(session_id, "assistant", answer)

        # ── Step 3: append footer (sources + timings)
        footer = ""
        timing_parts = []
        if rewrite_ms:  timing_parts.append(f"Rewrite: {rewrite_ms}s")
        timing_parts.append(f"Retrieve: {ret_ms}s")
        timing_parts.append(f"TTFT (Model Boot/Prefill): {ttft_ms}s")
        timing_parts.append(f"Generation: {round(stream_total_ms - ttft_ms, 3)}s")
        timing_parts.append(f"Total: {gen_ms}s")
        footer += f"\n\n⏱️ **Timings:** `{'  |  '.join(timing_parts)}`"

        if rewritten_q and rewritten_q != message:
            footer += f"\n\n🔍 **Rewritten Query:** `{rewritten_q}`"

        if sources:
            footer += "\n\n**Sources Used:**"
            for s in sources:
                footer += f"\n- [{s['id']}] `{s['source']}` (Confidence: {s['score']})"

        final_msg = answer + footer
        yield final_msg

        # ── Step 4: RAGAS evaluation
        contexts = [s["content"] for s in sources]
        if contexts and "blocked by" not in answer.lower():
            _ragas_pool.submit(_run_ragas_background, message, answer, contexts)

    except Exception as e:
        yield f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# Mini RAG System")
    gr.Markdown("Production RAG: LangGraph + BM25 + Qdrant Hybrid Search + Cross-Encoder Reranking.")
    # BUG 4 FIX — each browser tab gets its own session ID via gr.State.
    # Previously a single module-level SESSION_ID was shared by all users.
    session_state = gr.State(lambda: f"gradio_{uuid.uuid4().hex[:8]}")
    with gr.Tabs():
        with gr.TabItem("Chat"):
            gr.ChatInterface(
                fn=chat_fn,
                additional_inputs=[session_state],
                chatbot=gr.Chatbot(height=500),
                textbox=gr.Textbox(placeholder="Ask a question about the indexed documents...", container=False, scale=7),
            )
        with gr.TabItem("Add Knowledge"):
            gr.Markdown(
                "### Drop your files below to index them\n"
                "Supported formats: **PDF, DOCX, TXT, MD, CSV, JSON** (and most plain-text formats).  "
                "The filename is automatically used as the source name."
            )
            file_drop = gr.File(
                label="Drop files here or click to upload",
                file_count="multiple",
                type="filepath",
            )
            ingest_btn    = gr.Button("Ingest Files", variant="primary")
            ingest_status = gr.Textbox(label="Ingestion Status", interactive=False, lines=6)
            ingest_btn.click(fn=ingest_file, inputs=[file_drop], outputs=ingest_status)

# Mount Gradio onto the FastAPI app
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)