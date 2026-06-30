"""
LangGraph agentic RAG — port from mini_rag.py.

Graph nodes:
  rewrite → retrieve → generate → END

Matches mini_rag.py's graph exactly:
  - node_rewrite: skips LLM call on first-turn queries (no history)
  - node_retrieve: uses cached_retrieve_and_rerank (LRU cache)
  - node_generate: builds context with MAX_CONTEXT_CHARS cap, cites sources
  - Streaming via generate_stream()
  - Timing dict attached to state

Also retains the extended graph from agent/graph.py (with guards, planning,
multi-agent) as an optional advanced mode.
"""
import time
import functools
import concurrent.futures as _cf
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_settings
from models import RetrievedChunk
from retrieval.hybrid_retriever import HybridRetriever, build_cached_retriever
from retrieval.query_rewriter import QueryRewriter
from reranking.reranker import Reranker, ContextCompressor
from memory.conversation import ConversationMemory
from guardrails.guards import InputGuard, OutputGuard

settings = get_settings()

# ─── System prompt — matches mini_rag.py exactly ────────────────────────────
_SYSTEM_PROMPT = (
    "You are a professional enterprise AI assistant. "
    "Answer the user's question using ONLY the provided context below. "
    "Be concise, highly accurate, and factual. "
    "Always cite your sources using the provided markers (e.g., [1], [2]). "
    "If the answer is not contained in the context, explicitly state "
    "'I do not have enough information in the context to answer this.' Do not guess."
)


# ─── State ───────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    query: str
    session_id: str
    route: str
    rewritten_query: str
    candidates: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    compressed: list[RetrievedChunk]
    answer: str
    sources: list[dict]
    guard_passed: bool
    guard_reason: str
    conversation_history: str
    timings: dict


# ─── Context builders — exact port from mini_rag.py ─────────────────────────

def _build_context(hits: list[RetrievedChunk], original_q: str, rewritten_q: str) -> str:
    """Context string capped at MAX_CONTEXT_CHARS. Matches mini_rag.py."""
    parts, total = [], 0
    for i, rc in enumerate(hits):
        snippet = rc.chunk.content.strip()
        entry   = f"[{i+1}] {snippet}"
        if total + len(entry) > settings.max_context_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts)


def _build_user_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


# ─── Simple RAG graph (matches mini_rag.py structure) ───────────────────────

def build_simple_graph(
    retriever: HybridRetriever,
    reranker: Reranker,
    memory: ConversationMemory,
):
    """
    Builds the simple 3-node graph from mini_rag.py:
      rewrite → retrieve → generate → END

    Also creates:
      - cached_retrieve_and_rerank (LRU cache)
      - generate() / generate_stream() callables
    """
    llm = ChatGroq(
        model=settings.llm_model,
        temperature=0,
        max_tokens=settings.max_tokens,
    )

    # LRU cache over retrieve+rerank (matches cached_retrieve_and_rerank in mini_rag.py)
    cached_retrieve_and_rerank = build_cached_retriever(retriever, reranker)

    # ── Generation helpers ────────────────────────────────────────────────────

    def generate(original_q: str, rewritten_q: str, hits: list[RetrievedChunk]) -> str:
        context = _build_context(hits, original_q, rewritten_q)
        user    = _build_user_prompt(context, rewritten_q)
        return llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user),
        ]).content.strip()

    def generate_stream(original_q: str, rewritten_q: str, hits: list[RetrievedChunk]):
        """True token streaming — yields one token at a time."""
        context = _build_context(hits, original_q, rewritten_q)
        user    = _build_user_prompt(context, rewritten_q)
        for chunk in llm.stream([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user),
        ]):
            if chunk.content:
                yield chunk.content

    # ── Graph nodes ───────────────────────────────────────────────────────────

    def node_rewrite(s: RAGState) -> RAGState:
        """
        PHASE 1 — Parallel rewrite + prefetch (OPT from mini_rag.py).
        On first-turn queries (no history) skips LLM call entirely.
        """
        t0      = time.time()
        history = memory.get_history(s["session_id"])
        query   = s["query"]

        if not history.strip():
            s["rewritten_query"]      = query
            s["conversation_history"] = ""
        else:
            # Fire rewrite and dense embed in parallel
            with _cf.ThreadPoolExecutor(max_workers=2) as ex:
                rewrite_fut = ex.submit(retriever.query_rewriter.rewrite, query, history)
                embed_fut   = ex.submit(retriever.embedder.embed, [query])
                rewritten   = rewrite_fut.result()
                _           = embed_fut.result()   # warm embed cache
            s["rewritten_query"]      = rewritten
            s["conversation_history"] = history

        s.setdefault("timings", {})["rewrite"] = round(time.time() - t0, 2)
        return s

    def node_retrieve(s: RAGState) -> RAGState:
        t0 = time.time()
        q  = s.get("rewritten_query") or s["query"]
        # Convert immutable cached tuple back to list (BUG 2 fix from mini_rag.py)
        s["reranked"] = list(cached_retrieve_and_rerank(q))
        s["candidates"] = s["reranked"]
        s.setdefault("timings", {})["retrieve"] = round(time.time() - t0, 3)
        return s

    def node_generate(s: RAGState) -> RAGState:
        t0          = time.time()
        original_q  = s["query"]
        rewritten_q = s.get("rewritten_query") or original_q
        hits        = s.get("reranked") or s.get("candidates") or []

        answer = generate(original_q, rewritten_q, hits)
        s["answer"] = answer
        s["sources"] = [
            {
                "id":      i + 1,
                "chunk_id": rc.chunk.id,
                "source":  rc.chunk.source,
                "heading": rc.chunk.heading,
                "score":   round(
                    rc.rerank_score if settings.use_reranker else rc.rrf_score, 3
                ),
                "content": rc.chunk.content,
            }
            for i, rc in enumerate(hits)
        ]
        memory.add(s["session_id"], "user",      original_q)
        memory.add(s["session_id"], "assistant", answer)
        s.setdefault("timings", {})["generate"] = round(time.time() - t0, 2)
        return s

    # ── Assemble graph ────────────────────────────────────────────────────────
    graph = StateGraph(RAGState)
    graph.add_node("rewrite",  node_rewrite)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("generate", node_generate)
    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite",  "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    compiled = graph.compile()

    def run(query: str, session_id: str = "default") -> dict:
        """Public entry point — mirrors mini_rag.py's run() exactly."""
        t0 = time.time()
        result = compiled.invoke({
            "query": query, "session_id": session_id,
            "route": "simple",
            "rewritten_query": "",
            "candidates": [], "reranked": [], "compressed": [],
            "answer": "", "sources": [],
            "guard_passed": True, "guard_reason": "",
            "conversation_history": "",
            "timings": {},
        })
        timings = result.get("timings", {})
        timings["total"] = round(time.time() - t0, 2)
        return {
            "answer":          result["answer"],
            "sources":         result["sources"],
            "timings":         timings,
            "rewritten_query": result.get("rewritten_query", ""),
        }

    return compiled, run, generate, generate_stream, cached_retrieve_and_rerank


# ─── Advanced graph (guards + planner + multi-agent) ─────────────────────────

def build_graph(
    retriever: HybridRetriever,
    reranker: Reranker,
    compressor: ContextCompressor,
    memory: ConversationMemory,
):
    """
    Extended graph with input/output guards and multi-agent routing.
    Kept from the original agent/graph.py design.
    """
    llm = ChatGroq(
        model=settings.llm_model,
        temperature=0,
        max_tokens=settings.max_tokens,
    )
    input_guard  = InputGuard()
    output_guard = OutputGuard()
    cached_retrieve_and_rerank = build_cached_retriever(retriever, reranker)

    def node_input_guard(state: RAGState) -> RAGState:
        result = input_guard.check(state["query"])
        state["guard_passed"] = result.passed
        state["guard_reason"] = result.reason
        return state

    def node_plan(state: RAGState) -> RAGState:
        if not state["guard_passed"]:
            state["route"] = "out_of_scope"
            return state
        prompt = (
            "Classify this query. Reply with ONLY one word: simple, multi_hop, or out_of_scope.\n"
            "- simple: single factual lookup\n"
            "- multi_hop: requires chaining multiple facts\n"
            "- out_of_scope: harmful or completely irrelevant\n"
            f"Query: {state['query']}"
        )
        try:
            resp  = llm.invoke([HumanMessage(content=prompt)])
            route = resp.content.strip().lower()
            if route not in ("simple", "multi_hop", "out_of_scope"):
                route = "simple"
        except Exception:
            route = "simple"
        state["route"] = route
        return state

    def node_retrieve(state: RAGState) -> RAGState:
        t0      = time.time()
        history = memory.get_history(state["session_id"])
        state["conversation_history"] = history
        q = state["query"]
        state["reranked"] = list(cached_retrieve_and_rerank(q))
        state["candidates"] = state["reranked"]
        state.setdefault("timings", {})["retrieve"] = round(time.time() - t0, 3)
        return state

    def node_rerank(state: RAGState) -> RAGState:
        # Already reranked by cached_retrieve_and_rerank — pass through
        state["reranked"] = state.get("reranked") or state.get("candidates", [])
        return state

    def node_compress(state: RAGState) -> RAGState:
        state["compressed"] = compressor.compress_all(
            state["query"], state["reranked"]
        )
        return state

    def node_generate(state: RAGState) -> RAGState:
        t0      = time.time()
        hits    = state.get("compressed") or state.get("reranked") or []
        context = _build_context(hits, state["query"], state.get("rewritten_query", ""))
        user    = _build_user_prompt(context, state["query"])
        try:
            resp = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user)])
            state["answer"] = resp.content.strip()
        except Exception as e:
            state["answer"] = f"Error generating answer: {e}"
        state["sources"] = [
            {
                "id":      i + 1,
                "chunk_id": rc.chunk.id,
                "source":  rc.chunk.source,
                "heading": rc.chunk.heading,
                "score":   round(rc.rerank_score if settings.use_reranker else rc.rrf_score, 3),
                "content": rc.chunk.content,
            }
            for i, rc in enumerate(hits)
        ]
        memory.add(state["session_id"], "user",      state["query"])
        memory.add(state["session_id"], "assistant", state["answer"])
        state.setdefault("timings", {})["generate"] = round(time.time() - t0, 2)
        return state

    def node_multi_agent(state: RAGState) -> RAGState:
        try:
            from agent.multi_agent import MultiAgentExecutor
            executor = MultiAgentExecutor(retriever, reranker, n_agents=3)
            result   = executor.run(state["query"])
            state["answer"]  = result["answer"]
            state["sources"] = [{"source": s} for s in result["sources"]]
        except Exception:
            state = node_retrieve(state)
            state = node_compress(state)
            state = node_generate(state)
        return state

    def node_output_guard(state: RAGState) -> RAGState:
        chunk_texts = [rc.chunk.content for rc in (state.get("compressed") or [])]
        result = output_guard.check(state["answer"], chunk_texts)
        if not result.passed:
            state["answer"] += f"\n\n⚠️ Confidence note: {result.reason}"
        return state

    def node_out_of_scope(state: RAGState) -> RAGState:
        state["answer"]  = f"I can't help with that. {state.get('guard_reason', '')}"
        state["sources"] = []
        return state

    def route_after_plan(state: RAGState) -> str:
        return state["route"]

    graph = StateGraph(RAGState)
    graph.add_node("input_guard",  node_input_guard)
    graph.add_node("plan",         node_plan)
    graph.add_node("retrieve",     node_retrieve)
    graph.add_node("rerank",       node_rerank)
    graph.add_node("compress",     node_compress)
    graph.add_node("generate",     node_generate)
    graph.add_node("multi_agent",  node_multi_agent)
    graph.add_node("output_guard", node_output_guard)
    graph.add_node("out_of_scope", node_out_of_scope)

    graph.set_entry_point("input_guard")
    graph.add_edge("input_guard", "plan")
    graph.add_conditional_edges("plan", route_after_plan, {
        "simple":      "retrieve",
        "multi_hop":   "multi_agent",
        "out_of_scope":"out_of_scope",
    })
    graph.add_edge("retrieve",     "rerank")
    graph.add_edge("rerank",       "compress")
    graph.add_edge("compress",     "generate")
    graph.add_edge("generate",     "output_guard")
    graph.add_edge("multi_agent",  "output_guard")
    graph.add_edge("out_of_scope", END)
    graph.add_edge("output_guard", END)

    return graph.compile()
