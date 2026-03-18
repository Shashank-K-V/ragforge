"""
models.py — All Pydantic request / response schemas for RAGForge.

Design decisions
----------------
* Every public API surface is typed end-to-end with Pydantic v2 so FastAPI
  can auto-generate the OpenAPI spec and validate inputs for free.
* We keep request and response models separate (clear names: *Request vs
  *Response / *Info) so callers always know what they're sending vs
  receiving.
* Optional fields default to None / sensible values so clients can omit
  them without breaking the contract.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ======================================================================= #
#  Shared / primitive types                                                 #
# ======================================================================= #


class DocumentType(str, Enum):
    """Supported ingested document types."""

    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """
    Heuristic confidence bucket derived from the retrieval similarity scores.

    HIGH   → mean similarity ≥ 0.75
    MEDIUM → mean similarity ≥ 0.50
    LOW    → mean similarity < 0.50
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ======================================================================= #
#  Document ingestion                                                       #
# ======================================================================= #


class DocumentUploadResponse(BaseModel):
    """Returned after a successful /documents/upload call."""

    document_id: str = Field(
        ...,
        description="Stable UUID assigned to this document at upload time.",
        examples=["3f2504e0-4f89-11d3-9a0c-0305e82c3301"],
    )
    filename: str = Field(
        ...,
        description="Original filename as sent by the client.",
        examples=["annual_report_2024.pdf"],
    )
    document_type: DocumentType = Field(
        ...,
        description="Detected document type based on file extension.",
    )
    chunk_count: int = Field(
        ...,
        ge=0,
        description="Number of text chunks stored in the vector store.",
        examples=[42],
    )
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of ingestion completion.",
    )
    message: str = Field(
        default="Document ingested successfully.",
        description="Human-readable status message.",
    )


class DocumentInfo(BaseModel):
    """Metadata record for a single ingested document (used in list view)."""

    document_id: str
    filename: str
    document_type: DocumentType
    chunk_count: int
    ingested_at: datetime

    class Config:
        # Allow construction from ORM-like dicts / ChromaDB metadata dicts
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Returned by GET /documents."""

    documents: list[DocumentInfo] = Field(default_factory=list)
    total: int = Field(
        ...,
        ge=0,
        description="Total number of ingested documents.",
    )


# ======================================================================= #
#  Query / RAG pipeline                                                     #
# ======================================================================= #


class QueryRequest(BaseModel):
    """Body for POST /query."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural-language question to answer from the document corpus.",
        examples=["What were the total revenues in Q3 2024?"],
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="How many chunks to retrieve before synthesising the answer.",
    )
    document_id: str | None = Field(
        default=None,
        description=(
            "Optional: restrict retrieval to a single document. "
            "If omitted, the entire corpus is searched."
        ),
    )

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank or whitespace-only")
        return v.strip()


class SourceChunk(BaseModel):
    """A single retrieved text chunk returned alongside the answer."""

    content: str = Field(
        ...,
        description="The raw text of the retrieved chunk.",
    )
    document_id: str = Field(
        ...,
        description="ID of the source document this chunk belongs to.",
    )
    filename: str = Field(
        ...,
        description="Original filename of the source document.",
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Zero-based position of this chunk within the document.",
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score between the query and this chunk (0–1).",
    )
    page_number: int | None = Field(
        default=None,
        description="Page number this chunk came from (PDFs only).",
    )


class QueryResponse(BaseModel):
    """Returned by POST /query."""

    question: str = Field(..., description="Echo of the original question.")
    answer: str = Field(
        ...,
        description="LLM-synthesised answer grounded in retrieved chunks.",
    )
    source_documents: list[SourceChunk] = Field(
        default_factory=list,
        description="The top-k chunks used to generate the answer.",
    )
    confidence: ConfidenceLevel = Field(
        ...,
        description="Heuristic confidence level based on similarity scores.",
    )
    retrieval_latency_ms: float = Field(
        ...,
        ge=0,
        description="Wall-clock time spent on vector search, in milliseconds.",
    )
    synthesis_latency_ms: float = Field(
        ...,
        ge=0,
        description="Wall-clock time spent calling the LLM, in milliseconds.",
    )
    total_latency_ms: float = Field(
        ...,
        ge=0,
        description="End-to-end request latency, in milliseconds.",
    )


# ======================================================================= #
#  Evaluation                                                               #
# ======================================================================= #


class EvalTestCase(BaseModel):
    """A single ground-truth question/answer pair used in the eval suite."""

    question: str
    expected_answer_keywords: list[str] = Field(
        ...,
        description=(
            "Keywords that MUST appear in the synthesised answer for the "
            "test case to be considered correct."
        ),
    )
    expected_source_document_id: str | None = Field(
        default=None,
        description="If set, at least one retrieved chunk must come from this doc.",
    )


class EvalCaseResult(BaseModel):
    """Result for one test case in the evaluation suite."""

    question: str
    retrieved_chunks: int = Field(..., description="Number of chunks retrieved.")
    retrieval_hit: bool = Field(
        ...,
        description=(
            "True if the expected source document was found in the top-k results "
            "(only checked when expected_source_document_id is provided)."
        ),
    )
    answer_relevance: bool = Field(
        ...,
        description="True if all expected_answer_keywords appear in the answer.",
    )
    answer_snippet: str = Field(
        ...,
        description="First 200 chars of the synthesised answer (for inspection).",
    )
    latency_ms: float


class EvaluationResponse(BaseModel):
    """Returned by GET /evaluate."""

    total_cases: int
    retrieval_hit_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of cases where the expected document was retrieved.",
    )
    answer_relevance_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of cases where all expected keywords were in the answer.",
    )
    mean_latency_ms: float
    results: list[EvalCaseResult]
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


# ======================================================================= #
#  Health check                                                             #
# ======================================================================= #


class ComponentStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


class HealthResponse(BaseModel):
    """Returned by GET /health."""

    status: ComponentStatus
    version: str
    components: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-component health details (vector store, LLM, disk, etc.).",
    )
    uptime_seconds: float


# ======================================================================= #
#  Generic error                                                            #
# ======================================================================= #


class ErrorResponse(BaseModel):
    """Standard error envelope — returned for all 4xx / 5xx responses."""

    detail: str
    error_code: str | None = None
