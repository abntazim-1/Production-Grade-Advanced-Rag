from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid


class Document(BaseModel):
    """Raw document before chunking."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    source: str                        # filename / URL
    doc_type: str = "text"             # text | code | table | pdf
    metadata: dict = {}


class Chunk(BaseModel):
    """A single processed chunk ready for embedding.

    `source` is kept as a top-level field (matching mini_rag.py) so that
    every layer (retriever, graph, API) can access it without digging into
    `metadata`. It is also stored in `metadata["source"]` for backward compat.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""                   # parent document id (empty for manual ingest)
    content: str
    source: str = ""                   # filename / URL — top-level for easy access
    chunk_type: str = "text"           # text | table | heading | code
    heading: Optional[str] = None      # nearest parent heading
    summary: Optional[str] = None      # LLM-generated summary
    keywords: list[str] = []           # extracted keywords
    questions: list[str] = []          # synthetic Q&A pairs
    metadata: dict = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RetrievedChunk(BaseModel):
    """Chunk returned from retrieval with scores."""
    chunk: Chunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


class QueryRequest(BaseModel):
    query: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    top_k: int = 8
    use_reranking: bool = True
    use_query_rewriting: bool = True
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str
    rewritten_query: str = ""
    timings: dict = {}
    faithfulness_score: Optional[float] = None
    answer_relevancy_score: Optional[float] = None
