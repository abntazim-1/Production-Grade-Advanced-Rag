"""
LangGraph agentic RAG.

Graph nodes:
  input_guard → plan → [simple: retrieve→rerank→compress→generate]
                      [multi_hop: multi_agent_execute]
                      [out_of_scope: reject]
             → output_guard → save_memory → END
"""
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

from config import get_settings
from models import RetrievedChunk
from retrieval.hybrid_retriever import HybridRetriever
from reranking.reranker import Reranker, ContextCompressor
from memory.conversation import ConversationMemory
from guardrails.guards import InputGuard, OutputGuard

settings = get_settings()


class RAGState(TypedDict):
    query: str
    session_id: str
    route: str
    rewritten_queries: dict
    candidates: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    compressed: list[RetrievedChunk]
    answer: str
    sources: list[dict]
    guard_passed: bool
    guard_reason: str
    conversation_history: str


def build_graph(
    retriever: HybridRetriever,
    reranker: Reranker,
    compressor: ContextCompressor,
    memory: ConversationMemory,
):
    llm = ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=settings.llm_model,
        temperature=0,
    )
    input_guard = InputGuard()
    output_guard = OutputGuard()

    def get_multi_agent():
        from agent.multi_agent import MultiAgentExecutor
        return MultiAgentExecutor(retriever, reranker, n_agents=3)

    def node_input_guard(state: RAGState) -> RAGState:
        result = input_guard.check(state["query"])
        state["guard_passed"] = result.passed
        state["guard_reason"] = result.reason
        return state

    def node_plan(state: RAGState) -> RAGState:
        if not state["guard_passed"]:
            state["route"] = "out_of_scope"
            return state
        prompt = f"""Classify this query. Reply with ONLY one word: simple, multi_hop, or out_of_scope.
- simple: single factual lookup
- multi_hop: requires chaining multiple facts or comparing multiple things
- out_of_scope: harmful or completely irrelevant
Query: {state["query"]}"""
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            route = resp.content.strip().lower()
            if route not in ("simple", "multi_hop", "out_of_scope"):
                route = "simple"
        except Exception:
            route = "simple"
        state["route"] = route
        return state

    def node_retrieve(state: RAGState) -> RAGState:
        history = memory.format_for_prompt(state["session_id"])
        state["conversation_history"] = history
        augmented = state["query"]
        if history:
            augmented = f"Context: {history[-300:]}\nQuestion: {state['query']}"
        state["candidates"] = retriever.retrieve(augmented, top_k=settings.top_k_retrieval)
        return state

    def node_rerank(state: RAGState) -> RAGState:
        state["reranked"] = reranker.rerank(
            state["query"], state["candidates"], top_k=settings.top_k_rerank
        )
        return state

    def node_compress(state: RAGState) -> RAGState:
        state["compressed"] = compressor.compress_all(state["query"], state["reranked"])
        return state

    def node_generate(state: RAGState) -> RAGState:
        chunks = state["compressed"] or state["reranked"]
        context_parts, sources = [], []
        for i, rc in enumerate(chunks):
            context_parts.append(f"[{i+1}] {rc.chunk.content}")
            sources.append({
                "id": i + 1,
                "chunk_id": rc.chunk.id,
                "source": rc.chunk.metadata.get("source", "unknown"),
                "heading": rc.chunk.heading,
                "rerank_score": rc.rerank_score,
            })
        context_block = "\n\n".join(context_parts)
        history = state.get("conversation_history", "")
        system = (
            "You are a helpful assistant. Answer using ONLY the provided context. "
            "Cite sources as [1], [2], etc. If the answer is not in the context, say so."
        )
        user_msg = f"""Conversation history:
{history}

Context:
{context_block}

Question: {state["query"]}

Answer:"""
        try:
            resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user_msg)])
            state["answer"] = resp.content.strip()
        except Exception as e:
            state["answer"] = f"Error generating answer: {e}"
        state["sources"] = sources
        return state

    def node_multi_agent(state: RAGState) -> RAGState:
        try:
            executor = get_multi_agent()
            result = executor.run(state["query"])
            state["answer"] = result["answer"]
            state["sources"] = [{"source": s} for s in result["sources"]]
        except Exception:
            state = node_retrieve(state)
            state = node_rerank(state)
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
        reason = state.get("guard_reason") or "Query is out of scope."
        state["answer"] = f"I can't help with that. {reason}"
        state["sources"] = []
        return state

    def node_save_memory(state: RAGState) -> RAGState:
        memory.add(state["session_id"], "user", state["query"])
        memory.add(state["session_id"], "assistant", state["answer"])
        return state

    def route_after_plan(state: RAGState) -> str:
        return state["route"]

    graph = StateGraph(RAGState)
    graph.add_node("input_guard", node_input_guard)
    graph.add_node("plan", node_plan)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("rerank", node_rerank)
    graph.add_node("compress", node_compress)
    graph.add_node("generate", node_generate)
    graph.add_node("multi_agent", node_multi_agent)
    graph.add_node("output_guard", node_output_guard)
    graph.add_node("out_of_scope", node_out_of_scope)
    graph.add_node("save_memory", node_save_memory)

    graph.set_entry_point("input_guard")
    graph.add_edge("input_guard", "plan")
    graph.add_conditional_edges("plan", route_after_plan, {
        "simple": "retrieve",
        "multi_hop": "multi_agent",
        "out_of_scope": "out_of_scope",
    })
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "compress")
    graph.add_edge("compress", "generate")
    graph.add_edge("generate", "output_guard")
    graph.add_edge("multi_agent", "output_guard")
    graph.add_edge("out_of_scope", "save_memory")
    graph.add_edge("output_guard", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()
