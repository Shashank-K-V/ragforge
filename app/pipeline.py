"""
pipeline.py — Full RAG chain: retrieve → format context → synthesise answer.

Flow
----
  query
    │
    ▼
  retrieval.similarity_search()          ← vector store lookup
    │  top-k SourceChunk objects
    ▼
  _build_context_string()                ← assemble prompt context block
    │  formatted context string
    ▼
  _get_llm_chain()                       ← LangChain LLM + prompt template
    │  LLM answer
    ▼
  _compute_confidence()                  ← heuristic from similarity scores
    │
    ▼
  QueryResponse

LLM provider selection
-----------------------
* LLM_PROVIDER=huggingface (default):
    Uses langchain-huggingface HuggingFaceEndpoint pointing at the free
    HuggingFace Inference API.  Requires HUGGINGFACE_API_KEY.
    Model: mistralai/Mistral-7B-Instruct-v0.2 (configurable via HF_MODEL_ID).

* LLM_PROVIDER=openai:
    Uses langchain-openai ChatOpenAI.  Requires OPENAI_API_KEY.
    Model: gpt-3.5-turbo by default (configurable via OPENAI_MODEL).

Both paths use an identical prompt template so behaviour is consistent.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.config import settings
from app.models import ConfidenceLevel, QueryResponse, SourceChunk
from app.retrieval import similarity_search

logger = logging.getLogger(__name__)


# ======================================================================= #
#  Prompt template                                                          #
# ======================================================================= #

# The prompt is deliberately explicit about:
# 1. Using ONLY the provided context (prevents hallucination).
# 2. Saying "I don't know" when the context is insufficient.
# 3. Citing the source document name inline.

_RAG_PROMPT_TEMPLATE = """You are an expert document analyst.
Answer the user's question using ONLY the information provided in the context below.
If the context does not contain enough information to answer the question, say:
"I don't have enough information in the provided documents to answer that question."
Do NOT make up facts or use prior knowledge.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (be concise, cite the source document name when relevant):"""

_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=_RAG_PROMPT_TEMPLATE,
)


# ======================================================================= #
#  LLM factory (singleton per provider)                                    #
# ======================================================================= #

_llm_instance: BaseLLM | None = None


def _build_llm() -> Any:
    """
    Construct and return the appropriate LangChain LLM based on LLM_PROVIDER.

    Returns a LangChain-compatible LLM / chat model.  Both providers
    expose the same .invoke() interface via LCEL.
    """
    if settings.LLM_PROVIDER == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY to be set.")
        from langchain_openai import ChatOpenAI  # optional dependency

        logger.info("Using OpenAI LLM: %s", settings.OPENAI_MODEL)
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0,  # deterministic for QA
            max_tokens=512,
        )

    # --- Default: HuggingFace Inference API (free tier) ---
    if not settings.HUGGINGFACE_API_KEY:
        raise ValueError(
            "LLM_PROVIDER=huggingface requires HUGGINGFACE_API_KEY to be set. "
            "Get a free token at https://huggingface.co/settings/tokens"
        )

    from langchain_huggingface import HuggingFaceEndpoint  # noqa: PLC0415

    logger.info("Using HuggingFace LLM: %s", settings.HF_MODEL_ID)
    return HuggingFaceEndpoint(
        repo_id=settings.HF_MODEL_ID,
        huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.01,  # near-zero for deterministic QA
        repetition_penalty=1.1,
    )


def get_llm() -> Any:
    """Return the singleton LLM instance, building it on first call."""
    global _llm_instance  # noqa: PLW0603
    if _llm_instance is None:
        _llm_instance = _build_llm()
    return _llm_instance


# ======================================================================= #
#  Context builder                                                          #
# ======================================================================= #


def _build_context_string(chunks: list[SourceChunk]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Each chunk shows its source filename and page number (if known) so the
    LLM can cite the source naturally.
    """
    if not chunks:
        return "No relevant documents were found."

    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source_ref = chunk.filename
        if chunk.page_number:
            source_ref += f" (page {chunk.page_number})"
        parts.append(f"[{i}] Source: {source_ref}\n{chunk.content.strip()}")
    return "\n\n---\n\n".join(parts)


# ======================================================================= #
#  Confidence heuristic                                                     #
# ======================================================================= #


def _compute_confidence(chunks: list[SourceChunk]) -> ConfidenceLevel:
    """
    Bucket mean cosine similarity into a ConfidenceLevel.

    Thresholds are pragmatic and can be tuned via experimentation:
      HIGH   → mean ≥ 0.75  (query closely matches retrieved chunks)
      MEDIUM → mean ≥ 0.50
      LOW    → mean <  0.50  (weak match; answer may be off-topic)
    """
    if not chunks:
        return ConfidenceLevel.LOW

    mean_score = sum(c.similarity_score for c in chunks) / len(chunks)

    if mean_score >= 0.75:
        return ConfidenceLevel.HIGH
    if mean_score >= 0.50:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


# ======================================================================= #
#  Main public API                                                          #
# ======================================================================= #


def run_rag_pipeline(
    question: str,
    top_k: int | None = None,
    document_id: str | None = None,
) -> QueryResponse:
    """
    End-to-end RAG pipeline: retrieve → synthesise → respond.

    Parameters
    ----------
    question    : Natural-language question from the user.
    top_k       : Override for the number of chunks to retrieve.
    document_id : Restrict retrieval to a single document.

    Returns
    -------
    QueryResponse with answer, source_documents, confidence, and latencies.
    """
    top_k = top_k or settings.RETRIEVAL_TOP_K
    total_start = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Step 1: Retrieve relevant chunks                                     #
    # ------------------------------------------------------------------ #
    retrieval_start = time.perf_counter()
    chunks = similarity_search(query=question, top_k=top_k, document_id=document_id)
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

    logger.info(
        "Retrieved %d chunks for question='%s...' (%.0f ms)",
        len(chunks),
        question[:60],
        retrieval_latency_ms,
    )

    # ------------------------------------------------------------------ #
    # Step 2: Build context + synthesise answer                            #
    # ------------------------------------------------------------------ #
    context = _build_context_string(chunks)

    synthesis_start = time.perf_counter()

    llm = get_llm()

    # LangChain Expression Language (LCEL) chain:
    #   {"context": context, "question": question}
    #     → _prompt  (formats into the RAG prompt string)
    #     → llm      (calls the model)
    #     → StrOutputParser  (extracts string from AIMessage)
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | _prompt
        | llm
        | StrOutputParser()
    )

    answer: str = chain.invoke({"context": context, "question": question})
    answer = answer.strip()

    synthesis_latency_ms = (time.perf_counter() - synthesis_start) * 1000
    total_latency_ms = (time.perf_counter() - total_start) * 1000

    logger.info(
        "Synthesis complete (%.0f ms LLM | %.0f ms total)",
        synthesis_latency_ms,
        total_latency_ms,
    )

    # ------------------------------------------------------------------ #
    # Step 3: Assemble response                                            #
    # ------------------------------------------------------------------ #
    return QueryResponse(
        question=question,
        answer=answer,
        source_documents=chunks,
        confidence=_compute_confidence(chunks),
        retrieval_latency_ms=round(retrieval_latency_ms, 2),
        synthesis_latency_ms=round(synthesis_latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
    )
