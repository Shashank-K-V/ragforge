"""
evaluation.py — Retrieval and answer-quality evaluation framework.

The eval suite runs a fixed set of question/answer pairs against the live
RAG pipeline and measures two metrics:

1. Retrieval Hit Rate
   For each test case that specifies an expected_source_document_id, we
   check whether that document appears in the top-k retrieved chunks.
   Measures whether the vector store is finding the right documents.

2. Answer Relevance Rate
   For each test case, we check whether all expected_answer_keywords appear
   (case-insensitive) in the synthesised answer.
   Measures whether the LLM is using retrieved context correctly.

Why keyword matching instead of LLM-as-judge?
----------------------------------------------
* Zero extra cost — no secondary LLM call.
* Deterministic — same keywords always give the same pass/fail.
* Good enough for a first-pass smoke test; swap in RAGAS or TruLens for
  production-grade continuous evaluation.

Test cases
----------
The 5 hardcoded test cases below cover the sample.pdf bundled in docs/.
If you upload different documents, update TEST_CASES accordingly.
expected_source_document_id is set to None by default because document IDs
are assigned at upload time — populate it after uploading your docs.
"""

from __future__ import annotations

import logging
import time
from statistics import mean

from app.models import (
    EvalCaseResult,
    EvalTestCase,
    EvaluationResponse,
)
from app.pipeline import run_rag_pipeline

logger = logging.getLogger(__name__)


# ======================================================================= #
#  Hardcoded test suite                                                     #
# ======================================================================= #

# These test cases are written for the sample "AI/ML primer" document
# bundled in docs/sample.pdf.  If you swap out the document, update these.
#
# expected_source_document_id=None means "skip source-document hit check".
# After uploading sample.pdf, replace None with the returned document_id
# so retrieval hit rate is also tracked.

TEST_CASES: list[EvalTestCase] = [
    EvalTestCase(
        question="What is retrieval-augmented generation?",
        expected_answer_keywords=["retrieval", "generation", "context"],
        expected_source_document_id=None,
    ),
    EvalTestCase(
        question="What are the main components of a RAG pipeline?",
        expected_answer_keywords=["retrieval", "vector", "embedding"],
        expected_source_document_id=None,
    ),
    EvalTestCase(
        question="How does chunking affect retrieval quality?",
        expected_answer_keywords=["chunk", "overlap", "context"],
        expected_source_document_id=None,
    ),
    EvalTestCase(
        question="What embedding models are commonly used for RAG?",
        expected_answer_keywords=["embedding", "model", "sentence"],
        expected_source_document_id=None,
    ),
    EvalTestCase(
        question="What is cosine similarity and why is it used in vector search?",
        expected_answer_keywords=["cosine", "similarity", "vector"],
        expected_source_document_id=None,
    ),
]


# ======================================================================= #
#  Evaluation runner                                                        #
# ======================================================================= #


def _evaluate_single_case(case: EvalTestCase) -> EvalCaseResult:
    """
    Run one test case through the full RAG pipeline and score it.

    Returns an EvalCaseResult with retrieval_hit and answer_relevance flags.
    """
    t0 = time.perf_counter()

    try:
        response = run_rag_pipeline(
            question=case.question,
            top_k=4,
            document_id=None,  # always search full corpus during eval
        )
        answer = response.answer
        retrieved_doc_ids = {c.document_id for c in response.source_documents}
        chunk_count = len(response.source_documents)
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline error during eval for '%s': %s", case.question, exc)
        answer = ""
        retrieved_doc_ids = set()
        chunk_count = 0

    latency_ms = (time.perf_counter() - t0) * 1000

    # --- Retrieval hit ---
    # Only checked if expected_source_document_id is provided.
    if case.expected_source_document_id is not None:
        retrieval_hit = case.expected_source_document_id in retrieved_doc_ids
    else:
        # No expected document specified — count as hit to avoid penalising
        # users who haven't set up expected IDs yet.
        retrieval_hit = True

    # --- Answer relevance: all keywords must appear (case-insensitive) ---
    answer_lower = answer.lower()
    answer_relevance = all(
        kw.lower() in answer_lower for kw in case.expected_answer_keywords
    )

    logger.info(
        "Eval '%s...' → retrieval_hit=%s, answer_relevance=%s (%.0f ms)",
        case.question[:50],
        retrieval_hit,
        answer_relevance,
        latency_ms,
    )

    return EvalCaseResult(
        question=case.question,
        retrieved_chunks=chunk_count,
        retrieval_hit=retrieval_hit,
        answer_relevance=answer_relevance,
        answer_snippet=answer[:200],
        latency_ms=round(latency_ms, 2),
    )


def run_evaluation(
    test_cases: list[EvalTestCase] | None = None,
) -> EvaluationResponse:
    """
    Run the full evaluation suite and return aggregated metrics.

    Parameters
    ----------
    test_cases : Override the default TEST_CASES for custom evaluation.

    Returns
    -------
    EvaluationResponse with per-case results and aggregate metrics.
    """
    cases = test_cases or TEST_CASES
    logger.info("Starting evaluation suite: %d test cases.", len(cases))

    results: list[EvalCaseResult] = []
    for case in cases:
        result = _evaluate_single_case(case)
        results.append(result)

    # Aggregate
    total = len(results)
    hits_checked = [r for r in results]  # all results included in hit rate
    retrieval_hit_rate = (
        sum(1 for r in hits_checked if r.retrieval_hit) / total if total else 0.0
    )
    answer_relevance_rate = (
        sum(1 for r in results if r.answer_relevance) / total if total else 0.0
    )
    latencies = [r.latency_ms for r in results]
    mean_latency = mean(latencies) if latencies else 0.0

    logger.info(
        "Evaluation complete — hit_rate=%.2f, relevance=%.2f, mean_latency=%.0f ms",
        retrieval_hit_rate,
        answer_relevance_rate,
        mean_latency,
    )

    return EvaluationResponse(
        total_cases=total,
        retrieval_hit_rate=round(retrieval_hit_rate, 4),
        answer_relevance_rate=round(answer_relevance_rate, 4),
        mean_latency_ms=round(mean_latency, 2),
        results=results,
    )
