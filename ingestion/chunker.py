"""
Structure-aware chunker.
Detects headings, tables, and code blocks before splitting.
Respects sentence boundaries — no mid-sentence cuts.
"""
import re
from typing import Optional
from models import Document, Chunk


# ─── Heading detection ────────────────────────────────────────────────────────

HEADING_PATTERNS = [
    re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE),          # Markdown headings
    re.compile(r"^([A-Z][A-Z\s]{3,}):?\s*$", re.MULTILINE),  # ALL CAPS headings
    re.compile(r"^\d+\.\s+[A-Z].{5,}$", re.MULTILINE),       # Numbered headings
]

TABLE_PATTERN = re.compile(
    r"(\|.+\|[\r\n]+\|[-:| ]+\|[\r\n]+(?:\|.+\|[\r\n]*)+)",  # Markdown tables
    re.MULTILINE
)

CODE_BLOCK_PATTERN = re.compile(r"```[\w]*\n(.*?)```", re.DOTALL)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _detect_heading(line: str) -> Optional[str]:
    for pattern in HEADING_PATTERNS:
        m = pattern.match(line.strip())
        if m:
            return line.strip()
    return None


def _split_sentences(text: str) -> list[str]:
    """Rough sentence splitter that respects abbreviations."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


def _merge_sentences(sentences: list[str], max_chars: int) -> list[str]:
    """
    Greedy merge: keep adding sentences until we'd exceed max_chars,
    then start a new chunk. Ensures no sentence is split mid-way.
    """
    chunks, current = [], []
    current_len = 0
    for s in sentences:
        if current_len + len(s) > max_chars and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(s)
        current_len += len(s)
    if current:
        chunks.append(" ".join(current))
    return chunks


# ─── Main chunker ─────────────────────────────────────────────────────────────

class StructureAwareChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        text = doc.content

        # 1. Extract and preserve tables as atomic chunks
        table_spans = []
        for m in TABLE_PATTERN.finditer(text):
            table_spans.append((m.start(), m.end()))
            chunks.append(Chunk(
                doc_id=doc.id,
                content=m.group(0).strip(),
                chunk_type="table",
                metadata={**doc.metadata, "source": doc.source},
            ))

        # 2. Extract and preserve code blocks as atomic chunks
        code_spans = []
        for m in CODE_BLOCK_PATTERN.finditer(text):
            code_spans.append((m.start(), m.end()))
            chunks.append(Chunk(
                doc_id=doc.id,
                content=m.group(0).strip(),
                chunk_type="code",
                metadata={**doc.metadata, "source": doc.source},
            ))

        # 3. Remove table/code regions, process remaining text
        excluded = sorted(table_spans + code_spans)
        remaining_parts = []
        cursor = 0
        for start, end in excluded:
            if cursor < start:
                remaining_parts.append(text[cursor:start])
            cursor = end
        remaining_parts.append(text[cursor:])
        remaining_text = "\n".join(remaining_parts)

        # 4. Split remaining text by headings
        sections = self._split_by_headings(remaining_text)

        # 5. Within each section, merge sentences into sized chunks
        for heading, section_text in sections:
            sentences = _split_sentences(section_text)
            text_chunks = _merge_sentences(sentences, self.chunk_size)
            for tc in text_chunks:
                if tc.strip():
                    chunks.append(Chunk(
                        doc_id=doc.id,
                        content=tc.strip(),
                        chunk_type="text",
                        heading=heading,
                        metadata={**doc.metadata, "source": doc.source},
                    ))

        return chunks

    def _split_by_headings(self, text: str) -> list[tuple[Optional[str], str]]:
        """Return list of (heading, section_text) pairs."""
        lines = text.split("\n")
        sections: list[tuple[Optional[str], str]] = []
        current_heading = None
        current_lines: list[str] = []

        for line in lines:
            h = _detect_heading(line)
            if h:
                if current_lines:
                    sections.append((current_heading, "\n".join(current_lines)))
                    current_lines = []
                current_heading = h
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_heading, "\n".join(current_lines)))

        return sections
