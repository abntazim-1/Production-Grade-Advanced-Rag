"""
Query rewriting — port from mini_rag.py.

Uses the fast metadata_model (llama-3.1-8b-instant) for rewriting,
not the heavy main LLM — matches mini_rag.py's rewrite_query() exactly.

On first-turn queries (no history) rewriting is skipped entirely.
"""
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from config import get_settings

settings = get_settings()


class QueryRewriter:
    def __init__(self):
        # Use the fast metadata model — matches mini_rag.py
        self.llm = ChatGroq(
            model=settings.metadata_model,
            temperature=0,
            max_tokens=settings.metadata_max_tokens,
        )

    def rewrite(self, query: str, history: str = "") -> str:
        """
        Rewrites query into a standalone, search-optimized form.

        - If history is empty → returns query as-is (no LLM call).
        - Mirrors rewrite_query() in mini_rag.py exactly.
        """
        if not history.strip():
            return query

        prompt = (
            "Given the chat history and the latest user query, rewrite the query to be "
            "a standalone, search-optimized query. If it is already optimal, return it as is. "
            "ONLY output the rewritten query.\n\n"
            f"History:\n{history}\n\nQuery: {query}\n\nRewritten:"
        )
        try:
            return self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception as e:
            print(f"[QueryRewriter] Error: {e} — using original query")
            return query
