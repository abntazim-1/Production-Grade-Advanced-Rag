"""
Multi-agent system.
Splits a complex query into sub-questions, runs them in parallel
across N agents (each with its own retrieval context), then merges.

Used by the LangGraph planner for multi_hop routes.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

from config import get_settings
from models import RetrievedChunk
from retrieval.hybrid_retriever import HybridRetriever
from reranking.reranker import Reranker, ContextCompressor

settings = get_settings()


class SubAgent:
    """One agent handles one sub-question."""

    def __init__(self, agent_id: int, retriever: HybridRetriever, reranker: Reranker):
        self.agent_id = agent_id
        self.retriever = retriever
        self.reranker = reranker
        self.compressor = ContextCompressor()
        self.llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.llm_model,
            temperature=0,
        )

    def run(self, sub_question: str) -> dict:
        """Retrieve → rerank → generate answer for one sub-question."""
        candidates = self.retriever.retrieve(sub_question, use_query_rewriting=False)
        reranked = self.reranker.rerank(sub_question, candidates, top_k=4)
        compressed = self.compressor.compress_all(sub_question, reranked)

        context = "\n\n".join(
            f"[{i+1}] {rc.chunk.content}" for i, rc in enumerate(compressed)
        )

        try:
            resp = self.llm.invoke([
                SystemMessage(content="Answer concisely using only the provided context."),
                HumanMessage(content=f"Context:\n{context}\n\nQuestion: {sub_question}"),
            ])
            answer = resp.content.strip()
        except Exception as e:
            answer = f"Agent {self.agent_id} error: {e}"

        return {
            "agent_id": self.agent_id,
            "sub_question": sub_question,
            "answer": answer,
            "sources": [rc.chunk.metadata.get("source", "") for rc in compressed],
        }


class MultiAgentExecutor:
    """
    Decomposes a complex query into sub-questions,
    runs each on a separate SubAgent in parallel,
    then merges results with a final synthesis LLM call.
    """

    def __init__(self, retriever: HybridRetriever, reranker: Reranker, n_agents: int = 3):
        self.retriever = retriever
        self.reranker = reranker
        self.n_agents = n_agents
        self.agents = [SubAgent(i + 1, retriever, reranker) for i in range(n_agents)]
        self.decomposer = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.llm_model,
            temperature=0.2,
        )
        self.synthesizer = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.llm_model,
            temperature=0,
        )

    def _decompose(self, query: str) -> list[str]:
        """Break a complex query into at most N sub-questions."""
        prompt = f"""Break this query into {self.n_agents} specific sub-questions that together answer it.
Return ONLY a numbered list. No preamble.

Query: {query}"""
        try:
            resp = self.decomposer.invoke([HumanMessage(content=prompt)])
            lines = resp.content.strip().split("\n")
            sub_qs = []
            for line in lines:
                line = line.strip()
                if line and line[0].isdigit():
                    # Strip leading "1. " etc.
                    q = line.split(".", 1)[-1].strip()
                    if q:
                        sub_qs.append(q)
            return sub_qs[:self.n_agents] if sub_qs else [query]
        except Exception:
            return [query]

    def _synthesize(self, original_query: str, agent_results: list[dict]) -> str:
        """Merge sub-answers into one coherent final answer."""
        sub_answers = "\n\n".join(
            f"Sub-question: {r['sub_question']}\nAnswer: {r['answer']}"
            for r in agent_results
        )
        prompt = f"""Using these sub-answers, write one coherent answer to the original question.
Be concise. Cite facts from sub-answers.

Original question: {original_query}

Sub-answers:
{sub_answers}

Final answer:"""
        try:
            resp = self.synthesizer.invoke([HumanMessage(content=prompt)])
            return resp.content.strip()
        except Exception as e:
            # Fallback: concatenate
            return "\n\n".join(r["answer"] for r in agent_results)

    def run(self, query: str) -> dict:
        """Full multi-agent pipeline."""
        sub_questions = self._decompose(query)
        print(f"[MultiAgent] Decomposed into {len(sub_questions)} sub-questions")

        # Run agents in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.n_agents) as pool:
            futures = [
                pool.submit(agent.run, sub_q)
                for agent, sub_q in zip(self.agents, sub_questions)
            ]
            results = [f.result() for f in futures]

        final_answer = self._synthesize(query, results)
        all_sources = list({s for r in results for s in r["sources"] if s})

        return {
            "answer": final_answer,
            "sub_results": results,
            "sources": all_sources,
        }
