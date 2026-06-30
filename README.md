# Production RAG System

End-to-end RAG pipeline with smart chunking, hybrid retrieval, re-ranking, agentic routing, and RAGAS evaluation.

## Stack

| Layer | Tech |
|---|---|
| Embedding | BGE-M3 (sentence-transformers) |
| Vector store | Qdrant (Persistent local storage) |
| Sparse search | BM25 (rank-bm25) |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Ollama (llama3.2:3b for generation, qwen2.5:0.5b for metadata) |
| Orchestration | LangGraph |
| Evaluation | RAGAS |
| API | FastAPI + SSE streaming |

## Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Fill in GROQ_API_KEY (free at console.groq.com)

# 3. Smoke test (no API key needed)
python test_pipeline.py

# 4. Ingest your documents
python ingest.py --path ./documents

# 5. Run the API
uvicorn api.app:app --reload --port 8000
```

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Ingest
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "./documents"}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is backpropagation?", "session_id": "abc123"}'

# Streaming query
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain gradient descent", "session_id": "abc123", "stream": true}'
```

## Project Structure

```
rag_system/
├── ingestion/
│   ├── loaders.py          # PDF, DOCX, TXT, code loaders
│   ├── chunker.py          # Structure-aware chunker
│   └── metadata.py         # Summary + keyword + question generation
├── retrieval/
│   ├── embedder.py         # BGE-M3 multi-vector embedding
│   ├── vector_store.py     # FAISS (body + summary + questions indices)
│   ├── bm25_store.py       # BM25 sparse retrieval
│   ├── query_rewriter.py   # HyDE + paraphrase + entity expansion
│   └── hybrid_retriever.py # RRF fusion of dense + sparse
├── reranking/
│   └── reranker.py         # Cross-encoder re-ranking + context compression
├── agent/
│   └── graph.py            # LangGraph agentic router
├── memory/
│   └── conversation.py     # Per-session conversation memory
├── guardrails/
│   └── guards.py           # Input + output guardrails
├── evaluation/
│   └── evaluator.py        # RAGAS metrics
├── api/
│   └── app.py              # FastAPI endpoints + SSE streaming
├── ingest.py               # CLI ingestion script
├── test_pipeline.py        # Smoke test (no API key needed)
├── config.py               # Settings via pydantic
├── models.py               # Pydantic data models
└── requirements.txt
```

## Retrieval Flow

```text
User query
  → Parallel Query Rewriting (if history exists) & Dense Embedding Prefetch
  → Hybrid Retrieval (Dense Qdrant + Sparse BM25, RRF fusion)
  → Cross-encoder Re-ranking (top-25 → top-5)
  → LLM Generation (llama3.2:3b) with bounded conversation memory
  → True Token Streaming Response (SSE)
```

## Extending

- **Swap FAISS → Qdrant**: change `VectorStore` to use `qdrant-client`
- **Swap Groq → OpenAI**: change `ChatGroq` to `ChatOpenAI` in `graph.py`
- **Add web search tool**: add a LangGraph node that calls Tavily/SerpAPI
- **True multi-hop**: extend the `multi_hop` route in `graph.py` to chain retrieval
- **Human-in-the-loop**: add a `human_validation` node to the graph for low-confidence answers
