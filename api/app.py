"""
FastAPI application.
Endpoints:
  POST /ingest          — ingest a directory or file
  POST /query           — standard JSON response
  POST /query/stream    — Server-Sent Events streaming response
  GET  /health          — health check
"""
import asyncio
import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import get_settings
from models import QueryRequest, QueryResponse
from ingestion.loaders import load_directory, load_file
from ingestion.chunker import StructureAwareChunker
from ingestion.metadata import MetadataEnricher
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.bm25_store import BM25Store
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.query_rewriter import QueryRewriter
from reranking.reranker import Reranker, ContextCompressor
from memory.conversation import ConversationMemory
from evaluation.evaluator import RAGASEvaluator
from agent.graph import build_graph

settings = get_settings()

# ─── Singletons ───────────────────────────────────────────────────────────────

embedder = Embedder()
vector_store = VectorStore(embedder)
bm25_store = BM25Store()
query_rewriter = QueryRewriter()
retriever = HybridRetriever(vector_store, bm25_store, embedder, query_rewriter)
reranker = Reranker()
compressor = ContextCompressor()
memory = ConversationMemory()
evaluator = RAGASEvaluator()

# Try to load persisted indices
os.makedirs("./storage", exist_ok=True)
if os.path.exists(settings.faiss_index_path):
    vector_store.load()
if os.path.exists(settings.bm25_index_path):
    bm25_store.load()

# Build LangGraph
rag_graph = build_graph(retriever, reranker, compressor, memory)


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] RAG system ready")
    yield
    vector_store.save()
    bm25_store.save()
    print("[API] Indices saved")


app = FastAPI(title="Production RAG API", version="1.0.0", lifespan=lifespan)


# ─── Request models ───────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    path: str             # file or directory path
    save_index: bool = True


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "chunks_indexed": len(vector_store.index_to_id),
    }


@app.post("/ingest")
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest documents from a file or directory.
    Runs chunking + metadata enrichment + embedding in the background.
    """
    if not os.path.exists(req.path):
        raise HTTPException(status_code=400, detail=f"Path not found: {req.path}")

    def _run_ingestion():
        print(f"[Ingest] Loading from {req.path}")
        if os.path.isdir(req.path):
            docs = load_directory(req.path)
        else:
            docs = [load_file(req.path)]

        chunker = StructureAwareChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        enricher = MetadataEnricher()

        all_chunks = []
        for doc in docs:
            chunks = chunker.chunk(doc)
            enriched = enricher.enrich_batch(chunks)
            all_chunks.extend(enriched)

        vector_store.add(all_chunks)
        bm25_store.build(all_chunks)

        if req.save_index:
            vector_store.save()
            bm25_store.save()

        print(f"[Ingest] Done. {len(all_chunks)} chunks indexed.")

    background_tasks.add_task(_run_ingestion)
    return {"message": "Ingestion started in background"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Standard query — returns full response as JSON."""
    initial_state = {
        "query": req.query,
        "session_id": req.session_id,
        "route": "",
        "rewritten_queries": {},
        "candidates": [],
        "reranked": [],
        "compressed": [],
        "answer": "",
        "sources": [],
        "guard_passed": True,
        "guard_reason": "",
        "conversation_history": "",
    }

    result = rag_graph.invoke(initial_state)

    # Async RAGAS evaluation (non-blocking)
    contexts = [rc.chunk.content for rc in result.get("compressed", [])]
    eval_result = evaluator.evaluate_single(
        question=req.query,
        answer=result["answer"],
        contexts=contexts,
    )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=req.session_id,
        faithfulness_score=eval_result.faithfulness,
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """
    Streaming query via Server-Sent Events.
    Streams the answer token-by-token.
    """
    async def event_generator():
        # Run the graph (non-streaming for now; swap LLM for streaming in prod)
        initial_state = {
            "query": req.query,
            "session_id": req.session_id,
            "route": "",
            "rewritten_queries": {},
            "candidates": [],
            "reranked": [],
            "compressed": [],
            "answer": "",
            "sources": [],
            "guard_passed": True,
            "guard_reason": "",
            "conversation_history": "",
        }

        result = rag_graph.invoke(initial_state)
        answer = result["answer"]
        sources = result["sources"]

        # Simulate token streaming
        words = answer.split(" ")
        for word in words:
            yield f"data: {json.dumps({'token': word + ' '})}\n\n"
            await asyncio.sleep(0.02)

        # Send sources at the end
        yield f"data: {json.dumps({'sources': sources, 'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
