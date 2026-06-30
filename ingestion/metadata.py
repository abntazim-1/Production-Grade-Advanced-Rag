"""
Metadata creation for each chunk.
Generates: summary, keywords, synthetic questions.
These are embedded separately for multi-vector retrieval.
"""
import re
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from config import get_settings
from models import Chunk

settings = get_settings()


class MetadataEnricher:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.llm_model,
            temperature=0,
        )

    def enrich(self, chunk: Chunk) -> Chunk:
        """Add summary, keywords, and synthetic questions to a chunk."""
        # Skip very short chunks
        if len(chunk.content) < 80:
            return chunk

        prompt = f"""Given this text chunk, respond ONLY in this exact format:

SUMMARY: <one sentence summary>
KEYWORDS: <comma-separated keywords, max 8>
QUESTIONS: <3 questions this chunk answers, separated by |>

Chunk:
{chunk.content}"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            text = response.content.strip()

            chunk.summary = self._extract(text, "SUMMARY")
            keywords_raw = self._extract(text, "KEYWORDS")
            chunk.keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
            questions_raw = self._extract(text, "QUESTIONS")
            chunk.questions = [q.strip() for q in questions_raw.split("|") if q.strip()]
        except Exception as e:
            print(f"[MetadataEnricher] Failed for chunk {chunk.id}: {e}")

        return chunk

    def _extract(self, text: str, key: str) -> str:
        pattern = rf"{key}:\s*(.+?)(?=\n[A-Z]+:|$)"
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else ""

    def enrich_batch(self, chunks: list[Chunk]) -> list[Chunk]:
        return [self.enrich(c) for c in chunks]
