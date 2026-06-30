"""
Metadata enrichment — port from mini_rag.py.

Key changes:
  - Uses metadata_model (llama-3.1-8b-instant, fast/cheap) not the main LLM
  - Parallel batch enrichment via ThreadPoolExecutor (matches mini_rag.py's
    process_document() ThreadPoolExecutor approach)
  - JSON-based output format (matching mini_rag.py's generate_metadata_for_chunk)
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from config import get_settings
from models import Chunk

settings = get_settings()


class MetadataEnricher:
    def __init__(self):
        # Fast metadata model — matches mini_rag.py's metadata_llm
        self.llm = ChatGroq(
            model=settings.metadata_model,
            temperature=0,
            max_tokens=settings.metadata_max_tokens,
        )

    def enrich(self, chunk: Chunk) -> Chunk:
        """
        Adds summary, keywords, and synthetic questions to a chunk.
        Uses JSON output format from mini_rag.py's generate_metadata_for_chunk().
        """
        if len(chunk.content) < 30:
            return chunk

        prompt = (
            f"Analyze this text chunk (Heading: {chunk.heading or 'none'}):\n"
            f"{chunk.content}\n\n"
            "Return ONLY a JSON object with this exact structure:\n"
            '{"summary": "A 1-sentence summary", '
            '"keywords": ["keyword1", "keyword2"], '
            '"questions": ["Question 1?", "Question 2?"]}'
        )
        try:
            res = self.llm.invoke([HumanMessage(content=prompt)]).content
            start = res.find("{")
            end   = res.rfind("}") + 1
            if start != -1 and end != 0:
                data = json.loads(res[start:end])
                chunk.summary   = data.get("summary", "")
                chunk.keywords  = data.get("keywords", [])
                chunk.questions = data.get("questions", [])
        except Exception as e:
            print(f"[MetadataEnricher] Failed for chunk '{chunk.heading}': {e}")
        return chunk

    def enrich_batch(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Parallel enrichment using ThreadPoolExecutor.
        Matches mini_rag.py's process_document() ThreadPoolExecutor approach.
        Respects settings.metadata_workers and settings.generate_metadata.
        """
        if not settings.generate_metadata:
            print(
                f"[MetadataEnricher] Skipping LLM metadata — fast ingestion mode. "
                f"({len(chunks)} chunks, set GENERATE_METADATA=True to enable enrichment)"
            )
            return chunks

        print(f"[MetadataEnricher] Generating metadata for {len(chunks)} chunks "
              f"using {settings.metadata_workers} workers...")
        with ThreadPoolExecutor(max_workers=settings.metadata_workers) as executor:
            enriched = list(executor.map(self.enrich, chunks))
        return enriched
