"""
retrieval.py — Vector store management and similarity search.

Responsibilities
----------------
1. Initialise a persistent ChromaDB collection on first use.
2. Embed and store LangChain Document chunks (from ingestion.py).
3. Persist a lightweight document-registry JSON file so GET /documents
   can list uploaded files without querying Chroma.
4. Perform top-k cosine-similarity search and return SourceChunk objects.
5. Expose a per-document filter for scoped retrieval.

Why ChromaDB?
-------------
* Runs fully in-process — no external service, no Docker dependency.
* Persists to a local directory (CHROMA_PERSIST_DIR), survives restarts.
* First-class LangChain integration via Chroma wrapper.
* Free; no API keys; works on HF Spaces disk storage.

Embedding model
---------------
sentence-transformers/all-MiniLM-L6-v2 (via HuggingFaceEmbeddings):
* 80 MB download (cached after first run).
* 384-dimensional embeddings.
* Runs on CPU — no GPU required.
* Strong retrieval quality for English documents.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.models import DocumentInfo, DocumentType, SourceChunk

logger = logging.getLogger(__name__)

# Path to the simple JSON document registry
_REGISTRY_PATH = Path(settings.CHROMA_PERSIST_DIR) / "document_registry.json"


# ======================================================================= #
#  Embedding model (singleton)                                              #
# ======================================================================= #


_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return the singleton embedding model, initialising on first call.

    HuggingFaceEmbeddings downloads the model weights on first use and
    caches them in ~/.cache/huggingface.  Subsequent calls are instant.
    """
    global _embedding_model  # noqa: PLW0603
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # cosine sim works best
        )
        logger.info("Embedding model loaded.")
    return _embedding_model


# ======================================================================= #
#  ChromaDB vector store (singleton)                                        #
# ======================================================================= #


_vector_store: Chroma | None = None


def get_vector_store() -> Chroma:
    """
    Return the singleton Chroma vector store, creating the persisted
    collection if it doesn't already exist.
    """
    global _vector_store  # noqa: PLW0603
    if _vector_store is None:
        persist_dir = settings.CHROMA_PERSIST_DIR
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "Connecting to ChromaDB at '%s', collection '%s'",
            persist_dir,
            settings.CHROMA_COLLECTION_NAME,
        )
        _vector_store = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=get_embedding_model(),
            persist_directory=persist_dir,
        )
    return _vector_store


# ======================================================================= #
#  Document registry (lightweight JSON index)                               #
# ======================================================================= #


def _load_registry() -> dict[str, Any]:
    """Load the document registry from disk, returning an empty dict on miss."""
    if _REGISTRY_PATH.exists():
        try:
            return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read document registry: %s", exc)
    return {}


def _save_registry(registry: dict[str, Any]) -> None:
    """Atomically write the document registry to disk."""
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
    tmp.replace(_REGISTRY_PATH)


def register_document(
    document_id: str,
    filename: str,
    document_type: DocumentType,
    chunk_count: int,
) -> None:
    """Upsert a document record into the JSON registry."""
    registry = _load_registry()
    registry[document_id] = {
        "document_id": document_id,
        "filename": filename,
        "document_type": document_type.value,
        "chunk_count": chunk_count,
        "ingested_at": datetime.utcnow().isoformat(),
    }
    _save_registry(registry)
    logger.debug("Registered document %s (%s)", document_id, filename)


def list_documents() -> list[DocumentInfo]:
    """Return all registered documents as DocumentInfo objects."""
    registry = _load_registry()
    docs: list[DocumentInfo] = []
    for record in registry.values():
        docs.append(
            DocumentInfo(
                document_id=record["document_id"],
                filename=record["filename"],
                document_type=DocumentType(record.get("document_type", "unknown")),
                chunk_count=record["chunk_count"],
                ingested_at=datetime.fromisoformat(record["ingested_at"]),
            )
        )
    # Newest first
    return sorted(docs, key=lambda d: d.ingested_at, reverse=True)


# ======================================================================= #
#  Embedding and storage                                                    #
# ======================================================================= #


def embed_and_store(chunks: list[Document]) -> None:
    """
    Embed a list of Document chunks and upsert them into ChromaDB.

    We use add_documents (not from_documents) so we can call this
    incrementally for each uploaded file without rebuilding the store.

    ChromaDB deduplicates by ID so re-ingesting the same document_id +
    chunk_index combination is safe (it overwrites, not duplicates).
    """
    if not chunks:
        logger.warning("embed_and_store called with empty chunk list — skipping.")
        return

    vs = get_vector_store()

    # Assign stable IDs so upsert is idempotent:
    #   "{document_id}_{chunk_index}"
    ids = [
        f"{chunk.metadata['document_id']}_{chunk.metadata['chunk_index']}"
        for chunk in chunks
    ]

    logger.info("Embedding %d chunks…", len(chunks))
    t0 = time.perf_counter()
    vs.add_documents(documents=chunks, ids=ids)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("Stored %d chunks in ChromaDB (%.0f ms).", len(chunks), elapsed)


# ======================================================================= #
#  Similarity search                                                        #
# ======================================================================= #


def similarity_search(
    query: str,
    top_k: int | None = None,
    document_id: str | None = None,
) -> list[SourceChunk]:
    """
    Embed *query* and return the top-k most similar chunks.

    Parameters
    ----------
    query       : Natural-language question.
    top_k       : Number of results.  Defaults to settings.RETRIEVAL_TOP_K.
    document_id : If provided, restrict results to this document's chunks.

    Returns
    -------
    List of SourceChunk ordered by descending similarity score.
    """
    top_k = top_k or settings.RETRIEVAL_TOP_K
    vs = get_vector_store()

    # Build optional metadata filter for per-document scoping
    where_filter: dict[str, Any] | None = None
    if document_id:
        where_filter = {"document_id": {"$eq": document_id}}

    # similarity_search_with_relevance_scores returns (Document, score) pairs.
    # Scores are cosine similarities in [0, 1] when embeddings are normalised.
    results: list[tuple[Document, float]] = vs.similarity_search_with_relevance_scores(
        query=query,
        k=top_k,
        filter=where_filter,
    )

    source_chunks: list[SourceChunk] = []
    for doc, score in results:
        meta = doc.metadata
        source_chunks.append(
            SourceChunk(
                content=doc.page_content,
                document_id=meta.get("document_id", "unknown"),
                filename=meta.get("filename", "unknown"),
                chunk_index=meta.get("chunk_index", 0),
                similarity_score=round(float(score), 4),
                page_number=meta.get("page_number"),
            )
        )

    # Sort descending by score (Chroma usually returns them sorted, but
    # let's be explicit to guard against future version changes).
    source_chunks.sort(key=lambda c: c.similarity_score, reverse=True)
    return source_chunks


# ======================================================================= #
#  Health probe                                                             #
# ======================================================================= #


def check_vector_store_health() -> dict[str, Any]:
    """
    Return a health-check dict for GET /health.

    Intentionally lightweight — just counts documents in the collection.
    """
    try:
        vs = get_vector_store()
        count = vs._collection.count()  # noqa: SLF001  (internal but stable)
        return {"status": "ok", "total_chunks": count}
    except Exception as exc:  # noqa: BLE001
        logger.error("Vector store health check failed: %s", exc)
        return {"status": "down", "error": str(exc)}
