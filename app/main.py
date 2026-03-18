"""
main.py — FastAPI application entrypoint and all HTTP route handlers.

Routes
------
POST /documents/upload   — ingest a PDF or TXT file
GET  /documents          — list all ingested documents
POST /query              — ask a question via the RAG pipeline
GET  /evaluate           — run the evaluation suite
GET  /health             — liveness + component health check

Design notes
------------
* lifespan context manager warms up the embedding model and vector store
  at startup so the first real request isn't slow.
* File uploads are validated for content-type and size before any
  expensive processing begins.
* All errors are caught and returned as structured ErrorResponse JSON
  (never raw Python tracebacks).
* Logging is configured via LOG_LEVEL env var (default: INFO).
"""

from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.evaluation import run_evaluation
from app.ingestion import ingest_file
from app.models import (
    ComponentStatus,
    DocumentListResponse,
    DocumentUploadResponse,
    DocumentType,
    ErrorResponse,
    EvaluationResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from app.pipeline import run_rag_pipeline
from app.retrieval import (
    check_vector_store_health,
    embed_and_store,
    get_embedding_model,
    get_vector_store,
    list_documents,
    register_document,
)

# ======================================================================= #
#  Logging setup                                                            #
# ======================================================================= #

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ======================================================================= #
#  App startup / shutdown                                                   #
# ======================================================================= #

_startup_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Warm up expensive singletons at startup so the first real request
    doesn't pay the initialisation cost.
    """
    global _startup_time  # noqa: PLW0603
    _startup_time = time.time()

    logger.info("RAGForge starting up…")
    logger.info("LLM provider: %s", settings.LLM_PROVIDER)

    # Force embedding model download / cache load
    try:
        get_embedding_model()
        get_vector_store()
        logger.info("Vector store and embedding model ready.")
    except Exception as exc:  # noqa: BLE001
        # Non-fatal: the app still starts, endpoints will fail gracefully
        logger.warning("Warm-up failed (will retry on first request): %s", exc)

    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    yield  # ← application runs here

    logger.info("RAGForge shutting down.")


# ======================================================================= #
#  FastAPI app                                                              #
# ======================================================================= #

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — open for development; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================================= #
#  Allowed MIME types / extensions                                          #
# ======================================================================= #

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/octet-stream",  # some browsers send this for .pdf
}

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".text", ".docx"}


def _detect_doc_type_from_filename(filename: str) -> DocumentType:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mapping = {
        ".pdf": DocumentType.PDF,
        ".txt": DocumentType.TXT,
        ".text": DocumentType.TXT,
        ".docx": DocumentType.DOCX,
    }
    return mapping.get(ext, DocumentType.UNKNOWN)


# ======================================================================= #
#  Routes                                                                   #
# ======================================================================= #


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """
    Returns overall application health and per-component status.

    Used by Docker HEALTHCHECK, load balancers, and monitoring.
    """
    vs_health = check_vector_store_health()
    vector_store_status = (
        ComponentStatus.OK if vs_health.get("status") == "ok" else ComponentStatus.DOWN
    )

    # LLM is considered healthy if provider credentials are configured
    llm_configured = bool(
        settings.OPENAI_API_KEY
        if settings.LLM_PROVIDER == "openai"
        else settings.HUGGINGFACE_API_KEY
    )
    llm_status = ComponentStatus.OK if llm_configured else ComponentStatus.DEGRADED

    overall = (
        ComponentStatus.OK
        if (vector_store_status == ComponentStatus.OK and llm_status == ComponentStatus.OK)
        else ComponentStatus.DEGRADED
    )

    uptime = time.time() - _startup_time if _startup_time else 0.0

    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        components={
            "vector_store": {**vs_health, "status": vector_store_status.value},
            "llm": {
                "status": llm_status.value,
                "provider": settings.LLM_PROVIDER,
                "model": (
                    settings.OPENAI_MODEL
                    if settings.LLM_PROVIDER == "openai"
                    else settings.HF_MODEL_ID
                ),
            },
            "embedding": {
                "status": "ok",
                "model": settings.EMBEDDING_MODEL,
            },
        },
        uptime_seconds=round(uptime, 2),
    )


@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
    tags=["Documents"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or size"},
        422: {"model": ErrorResponse, "description": "No text could be extracted"},
        500: {"model": ErrorResponse, "description": "Internal ingestion error"},
    },
)
async def upload_document(
    file: UploadFile = File(..., description="PDF, TXT, or DOCX file to ingest"),
) -> DocumentUploadResponse:
    """
    Upload a document and trigger the full ingestion pipeline.

    1. Validates file type and size.
    2. Extracts text (PDF page-by-page, DOCX paragraph-by-paragraph, TXT as-is).
    3. Splits into overlapping chunks (CHUNK_SIZE chars, CHUNK_OVERLAP overlap).
    4. Embeds chunks with sentence-transformers and stores in ChromaDB.
    5. Registers the document in the JSON registry for GET /documents.

    Returns the assigned document_id and chunk_count.
    """
    filename = file.filename or "upload"

    # --- Validate extension ---
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # --- Read file bytes ---
    file_bytes = await file.read()

    # --- Validate size ---
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File size {len(file_bytes) / 1_048_576:.1f} MB exceeds the "
                f"{settings.MAX_FILE_SIZE_MB} MB limit."
            ),
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # --- Ingest ---
    try:
        document_id, chunks = ingest_file(filename=filename, file_bytes=file_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during ingestion of '%s'", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text content could be extracted from the uploaded file.",
        )

    # --- Embed and store ---
    try:
        embed_and_store(chunks)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error embedding document '%s'", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store error: {exc}",
        ) from exc

    # --- Register in document registry ---
    doc_type = _detect_doc_type_from_filename(filename)
    register_document(
        document_id=document_id,
        filename=filename,
        document_type=doc_type,
        chunk_count=len(chunks),
    )

    logger.info(
        "Uploaded and ingested '%s' → id=%s, %d chunks.",
        filename,
        document_id,
        len(chunks),
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        document_type=doc_type,
        chunk_count=len(chunks),
    )


@app.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
    tags=["Documents"],
)
async def get_documents() -> DocumentListResponse:
    """
    Return metadata for all previously ingested documents.

    Documents are returned newest-first.  The document_id from this
    response can be used in POST /query to scope retrieval to a single doc.
    """
    docs = list_documents()
    return DocumentListResponse(documents=docs, total=len(docs))


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against the document corpus",
    tags=["RAG"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid question"},
        503: {"model": ErrorResponse, "description": "No documents ingested yet"},
        500: {"model": ErrorResponse, "description": "Pipeline error"},
    },
)
async def query_documents(body: QueryRequest) -> QueryResponse:
    """
    Run the full RAG pipeline for the given question.

    1. Embeds the question.
    2. Retrieves the top-k most similar chunks (optionally scoped to one doc).
    3. Sends the question + context to the LLM for answer synthesis.
    4. Returns the answer, source chunks, confidence level, and latency breakdown.
    """
    # Guard: at least one document must be ingested
    docs = list_documents()
    if not docs:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No documents have been ingested yet. "
                "Upload at least one document via POST /documents/upload first."
            ),
        )

    try:
        response = run_rag_pipeline(
            question=body.question,
            top_k=body.top_k,
            document_id=body.document_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG pipeline error for question='%s'", body.question[:80])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        ) from exc

    return response


@app.get(
    "/evaluate",
    response_model=EvaluationResponse,
    summary="Run the retrieval & answer-quality evaluation suite",
    tags=["Evaluation"],
    responses={
        503: {"model": ErrorResponse, "description": "No documents ingested yet"},
        500: {"model": ErrorResponse, "description": "Evaluation error"},
    },
)
async def evaluate(
    max_cases: int = Query(
        default=5,
        ge=1,
        le=20,
        description="Number of test cases to run (defaults to full suite).",
    ),
) -> EvaluationResponse:
    """
    Run a subset of the hardcoded evaluation suite and return metrics.

    Metrics returned:
    * **retrieval_hit_rate** — fraction of cases where the expected document
      was in the top-k results.
    * **answer_relevance_rate** — fraction of cases where all expected keywords
      appeared in the synthesised answer.
    * **mean_latency_ms** — average end-to-end latency per case.

    Note: this endpoint calls the LLM for every test case.  With 5 cases and
    a remote LLM, expect ~10–30 seconds total.
    """
    docs = list_documents()
    if not docs:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No documents ingested. Upload documents before running evaluation.",
        )

    try:
        from app.evaluation import TEST_CASES, run_evaluation  # noqa: PLC0415

        cases = TEST_CASES[:max_cases]
        result = run_evaluation(test_cases=cases)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Evaluation error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {exc}",
        ) from exc

    return result


# ======================================================================= #
#  Global exception handler                                                 #
# ======================================================================= #


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """Catch-all: return structured JSON for any unhandled exception."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail="An unexpected internal error occurred.",
            error_code="INTERNAL_ERROR",
        ).model_dump(),
    )


# ======================================================================= #
#  Entrypoint                                                               #
# ======================================================================= #

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
