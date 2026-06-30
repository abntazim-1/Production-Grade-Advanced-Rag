"""
Query rewriting.
Produces 3 query variants before retrieval:
  1. Cleaned paraphrase
  2. HyDE — hypothetical document embedding (what would the answer look like?)
  3. Entity-expanded version
Each variant is embedded and retrieved separately; results are merged.
"""
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from config import get_settings

settings = get_settings()


class QueryRewriter:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.llm_model,
            temperature=0.3,
        )

    def rewrite(self, query: str) -> dict[str, str]:
        """
        Returns dict with keys: 'original', 'paraphrase', 'hyde', 'expanded'
        Falls back to original on any error.
        """
        prompt = f"""Given this user query, produce 3 variants. Respond ONLY in this exact format:

PARAPHRASE: <reworded version, same meaning>
HYDE: <a short hypothetical passage that would answer this query, 2-3 sentences>
EXPANDED: <query with relevant entities and synonyms added>

Query: {query}"""

        variants = {
            "original": query,
            "paraphrase": query,
            "hyde": query,
            "expanded": query,
        }

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            text = response.content.strip()
            for key in ["paraphrase", "hyde", "expanded"]:
                tag = key.upper() + ":"
                if tag in text:
                    after = text.split(tag, 1)[1]
                    value = after.split("\n")[0].strip()
                    if value:
                        variants[key] = value
        except Exception as e:
            print(f"[QueryRewriter] Error: {e} — using original query")

        return variants
