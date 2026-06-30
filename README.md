# Production RAG Platform

A production-grade, enterprise-ready Retrieval-Augmented Generation system with hybrid search, cross-encoder reranking, LangGraph orchestration, streaming inference, and an embedded Gradio UI — all served through a single FastAPI application.

---

## Architecture

### Query Pipeline

```
┌──────────────────────────┐
│        User / Client     │
│  Web UI / API / Gradio   │
└─────────────┬────────────┘
              │ Query
              ▼
┌─────────────────────────────┐
│       FastAPI Server        │
│  /query  /stream  /ingest   │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│    LangGraph Orchestrator   │
└──────┬──────────┬───────────┘
       │          │           │
       ▼          ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Session  │ │  Query   │ │  Timing  │
│  Memory  │ │ Rewriter │ │ Tracker  │
│          │ │ Groq 8B  │ │          │
└──────────┘ └────┬─────┘ └──────────┘
                  │
                  ▼
     ┌────────────────────────┐
     │   Hybrid Retrieval     │
     └───────┬────────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
┌──────────┐  ┌──────────┐
│  Dense   │  │  Sparse  │
│ BGE-M3   │  │  BM25    │
│  Qdrant  │  │ Keyword  │
└──────────┘  └──────────┘
      │             │
      └──────┬──────┘
             ▼
    Reciprocal Rank Fusion
             │
             ▼
┌────────────────────────────┐
│  Cross-Encoder Reranker    │
│  TinyBERT (4x faster than  │
│  MiniLM, minimal quality   │
│  loss)                     │
└────────────┬───────────────┘
             │
             ▼
      Top 5 Relevant Chunks
             │
             ▼
┌────────────────────────────┐
│   Context Builder          │
│   Source Citation Tagger   │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│      Llama 3.3 70B         │
│     (Groq Inference)       │
└────────────┬───────────────┘
             │
             ▼
      Streaming Response
             │
             ▼
       User gets Answer
```

### Ingestion Pipeline

```
 Documents
 PDF / DOCX / TXT / JSON / MD
          │
          ▼
 Document Parser & Extractor
 (loaders.py)
          │
          ▼
 Structure-Aware Chunking Engine
 • Heading Detection (Markdown + ALL CAPS)
 • Table Preservation (atomic chunks)
 • Code Block Preservation
 • Overlap carry-over on size splits
          │
          ▼
 Metadata Generation (Groq 8B — fast)
 • Summary
 • Keywords
 • Synthetic Questions
 (Parallel: 6 workers via ThreadPoolExecutor)
          │
          ▼
 BGE-M3 Embedding Model
 (GPU-accelerated: XPU → CUDA → CPU)
          │
          ▼
 Multi-Vector Embeddings per Chunk
 ┌────────┬────────┬──────────┐
 │Content │Summary │Questions │
 └────────┴────────┴──────────┘
          │
          ▼
 Qdrant Vector Database
 (persistent local storage)
          │
          ▼
 BM25 Index (rebuilt on startup
 from persisted Qdrant chunks)
```

### Deployment Overview

```
            Internet
                │
                ▼
     FastAPI Application (:8000)
                │
   ┌────────────┼────────────────┐
   ▼            ▼                ▼
LangGraph    Qdrant DB      Session Memory
   │         (local disk)   (per-tab, gr.State)
   ▼
Groq API
(Llama 3.3 70B + Llama 3.1 8B)

   ▲
   │
BGE-M3 Embedding (local GPU)
TinyBERT Reranker (local GPU)
```

---

## Stack

| Layer | Technology |
|---|---|
| **LLM (Generation)** | `llama-3.3-70b-versatile` via Groq |
| **LLM (Metadata / Rewrite)** | `llama-3.1-8b-instant` via Groq |
| **Embedding** | `BAAI/bge-m3` (sentence-transformers, GPU) |
| **Reranker** | `cross-encoder/ms-marco-TinyBERT-L-2-v2` (4× faster than MiniLM) |
| **Vector Store** | Qdrant (persistent local, multi-vector) |
| **Sparse Search** | BM25 (rank-bm25) |
| **Fusion** | Reciprocal Rank Fusion (RRF, k=60) |
| **Orchestration** | LangGraph |
| **Evaluation** | Embedding-based (faithfulness + answer relevancy, ~10ms, no LLM) |
| **API** | FastAPI + SSE token streaming |
| **UI** | Gradio (Chat tab + File Upload tab) |
| **GPU Support** | Intel Arc XPU → NVIDIA CUDA → CPU (auto-detected) |

---

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
# Create .env with your Groq API key (free at console.groq.com)
GROQ_API_KEY=your_key_here
LLM_MODEL=llama-3.3-70b-versatile
METADATA_MODEL=llama-3.1-8b-instant

# 4. Run the server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# 5. Open the Gradio UI
# http://localhost:8000
```

---

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
# Returns: chunks_in_memory, vectors_in_qdrant, active_sessions, indexed_sources
```

### Ingest Text
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document content...", "source": "my_doc.txt"}'
```

### Query (Blocking)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is backpropagation?", "session_id": "abc123"}'
# Returns: answer, sources, timings, rewritten_query
```

### Query (Streaming SSE)
```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain gradient descent", "session_id": "abc123"}'
# Streams: data: {"token": "..."} ... data: {"sources": [...], "done": true}
```

### Batch Evaluation
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"questions": ["What is RAG?", "How does BM25 work?"]}'
# Returns: faithfulness, answer_relevancy (embedding-based, no LLM judge)
```

### Delete a Source
```bash
curl -X DELETE http://localhost:8000/source/my_doc.txt
# Removes all vectors + in-memory chunks for that source
```

### CLI Client
```bash
python client.py --url http://localhost:8000
# Interactive multi-turn streaming chat in the terminal
```

---

## Project Structure

```
rag_system/
├── api/
│   └── app.py              # FastAPI endpoints + SSE streaming + Gradio UI
├── agent/
│   ├── graph.py            # LangGraph orchestrator (simple + advanced graphs)
│   └── multi_agent.py      # Multi-agent decompose → parallel → synthesize
├── ingestion/
│   ├── loaders.py          # PDF, DOCX, TXT, MD, JSON, code loaders
│   ├── chunker.py          # Structure-aware chunker (headings, tables, overlap)
│   └── metadata.py         # Parallel LLM metadata enrichment (summary, keywords, Q&A)
├── retrieval/
│   ├── embedder.py         # BGE-M3 + GPU detection + LRU query cache
│   ├── vector_store.py     # Qdrant backend (persist, restore, multi-vector, dedup)
│   ├── bm25_store.py       # BM25 sparse retrieval (content + keywords + summary + Q)
│   ├── query_rewriter.py   # Standalone query rewriting (skips on first turn)
│   └── hybrid_retriever.py # Parallel dense+sparse → RRF → LRU cache
├── reranking/
│   └── reranker.py         # TinyBERT cross-encoder + positive-score filter
├── memory/
│   └── conversation.py     # Thread-safe bounded deque per session
├── guardrails/
│   └── guards.py           # Prompt injection + out-of-scope detection
├── evaluation/
│   └── evaluator.py        # Embedding-based faithfulness + answer relevancy
├── config.py               # Pydantic settings (all tuning knobs)
├── models.py               # Chunk, RetrievedChunk, QueryRequest, QueryResponse
├── mini_rag.py             # Original single-file reference implementation
├── test_pipeline.py        # Smoke test (no API key needed)
├── stress_test.py          # Adversarial red-team test suite
├── client.py               # Interactive CLI streaming client
└── requirements.txt
```

---

## Performance Characteristics

| Metric | Value |
|---|---|
| Retrieval latency (cached) | ~0ms (LRU cache hit) |
| Retrieval latency (cold) | ~400–600ms (parallel dense+sparse) |
| Reranking (TinyBERT, 15 candidates) | ~50–100ms |
| TTFT (Groq Llama 3.3 70B) | ~0.5–1.5s |
| Embedding (BGE-M3, GPU) | ~50–150ms per query |
| Evaluation (embedding-based) | ~10ms (no LLM call) |
| Startup restore (Qdrant → RAM) | automatic on boot |

### Key Optimizations
- **OPT-1** Dense + Sparse search run in **parallel** (`ThreadPoolExecutor`) — wall-clock = `max(dense, sparse)` not `dense + sparse`
- **OPT-2** `@lru_cache(maxsize=1024)` on retrieve+rerank — repeated queries hit at ~0ms
- **OPT-3** `@lru_cache(maxsize=2048)` on single-string embedding — query re-embeds cached
- **OPT-4** Model warm-up on startup — zero cold-start on first query
- **OPT-5** TinyBERT reranker (4× faster than MiniLM-L-6 with minimal quality loss)
- **OPT-6** Metadata enrichment parallelized with 6 worker threads

---

## Extending

- **Swap LLM**: change `llm_model` in `.env` to any Groq-supported model
- **Add web search**: add a LangGraph node calling Tavily/SerpAPI before retrieval
- **Multi-hop queries**: the `multi_hop` route in `agent/graph.py` decomposes → runs N sub-agents in parallel → synthesizes
- **Human-in-the-loop**: add a `human_validation` node for low-confidence answers
- **Production DB**: swap `VectorStore` Qdrant path for a remote Qdrant Cloud cluster
