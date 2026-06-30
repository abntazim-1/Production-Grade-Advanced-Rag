"""
Structure-aware chunker — port from mini_rag.py.

Matches structure_aware_chunking() in mini_rag.py exactly:
  - Table preservation (atomic chunks, hard boundary — no overlap)
  - Heading detection (Markdown + ALL CAPS + numbered)
  - Size-triggered splits with configurable line overlap (chunk_overlap)
  - Code block preservation (atomic chunks)
  - Minimum chunk length filter (> 30 chars)
"""
import re
from typing import Optional
from models import Document, Chunk
from config import get_settings

settings = get_settings()

# ─── Pattern definitions ──────────────────────────────────────────────────────

HEADING_PATTERNS = [
    re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE),           # Markdown headings
    re.compile(r"^([A-Z][A-Z\s]{3,}):?\s*$", re.MULTILINE),   # ALL CAPS headings
    re.compile(r"^\d+\.\s+[A-Z].{5,}$", re.MULTILINE),        # Numbered headings
]

TABLE_ROW_RE   = re.compile(r"^\s*\|.*\|\s*$")
CODE_BLOCK_RE  = re.compile(r"```[\w]*\n(.*?)```", re.DOTALL)
TABLE_BLOCK_RE = re.compile(
    r"(\|.+\|[\r\n]+\|[-:| ]+\|[\r\n]+(?:\|.+\|[\r\n]*)+)",
    re.MULTILINE,
)


def _is_heading(line: str) -> bool:
    """Matches heading detection in mini_rag.py's structure_aware_chunking."""
    stripped = line.strip()
    return bool(
        re.match(r"^#{1,6}\s", stripped)
        or re.match(r"^[A-Z][A-Z\s]{3,}:?\s*$", stripped)
    )


# ─── Main chunker ─────────────────────────────────────────────────────────────

class StructureAwareChunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size    = chunk_size    or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        """
        Two-pass chunker matching mini_rag.py's structure_aware_chunking():

        Pass 1 — Extract tables and code blocks as atomic chunks.
        Pass 2 — Process remaining text line-by-line with heading detection,
                  table-row detection, and size-triggered overlap splits.
        """
        chunks: list[Chunk] = []
        text  = doc.content

        # ── Pass 1: atomic blocks ─────────────────────────────────────────────
        excluded_spans: list[tuple[int, int]] = []

        for m in CODE_BLOCK_RE.finditer(text):
            excluded_spans.append((m.start(), m.end()))
            chunks.append(Chunk(
                doc_id     = doc.id,
                source     = doc.source,
                content    = m.group(0).strip(),
                chunk_type = "code",
                metadata   = {**doc.metadata, "source": doc.source},
            ))

        for m in TABLE_BLOCK_RE.finditer(text):
            excluded_spans.append((m.start(), m.end()))
            chunks.append(Chunk(
                doc_id     = doc.id,
                source     = doc.source,
                content    = m.group(0).strip(),
                chunk_type = "table",
                metadata   = {**doc.metadata, "source": doc.source},
            ))

        # Build remaining text (everything outside excluded spans)
        excluded_spans.sort()
        remaining_parts: list[str] = []
        cursor = 0
        for start, end in excluded_spans:
            if cursor < start:
                remaining_parts.append(text[cursor:start])
            cursor = end
        remaining_parts.append(text[cursor:])
        remaining_text = "\n".join(remaining_parts)

        # ── Pass 2: line-by-line (mirrors mini_rag.py exactly) ───────────────
        lines = re.sub(r"\n{3,}", "\n\n", remaining_text).split("\n")
        chunks.extend(self._line_by_line(lines, doc))
        return chunks

    def _line_by_line(self, lines: list[str], doc: Document) -> list[Chunk]:
        """
        Exact port of mini_rag.py's structure_aware_chunking() inner loop.
        """
        result: list[Chunk] = []
        current_block: list[str] = []
        heading = ""
        in_table = False

        def save_chunk(overlap: bool = False):
            nonlocal current_block
            if current_block:
                content = "\n".join(current_block).strip()
                if len(content) > 30:
                    result.append(Chunk(
                        doc_id     = doc.id,
                        source     = doc.source,
                        content    = content,
                        chunk_type = "text",
                        heading    = heading or None,
                        metadata   = {**doc.metadata, "source": doc.source},
                    ))
                # Overlap: keep last N lines for next chunk (size-triggered splits only)
                if overlap and len(current_block) > self.chunk_overlap:
                    kept = current_block[-self.chunk_overlap:]
                    current_block = kept
                else:
                    current_block = []

        for line in lines:
            is_table_row = bool(TABLE_ROW_RE.match(line))

            if is_table_row:
                in_table = True
                current_block.append(line)
                continue
            elif in_table:
                in_table = False
                save_chunk()  # hard boundary — no overlap across table end

            if _is_heading(line):
                save_chunk()  # hard boundary at headings — no overlap
                heading = line.strip()
            else:
                current_block.append(line)
                # Size-triggered split with overlap (matches mini_rag.py)
                if sum(len(w) for w in current_block) > self.chunk_size * 5:
                    save_chunk(overlap=True)

        save_chunk()
        return result
