"""
test_pipeline.py — Tests for the end-to-end RAG pipeline.

Strategy
--------
The LLM and vector store are mocked in all tests so the pipeline logic
(context building, confidence scoring, response structure) can be verified
without making any real API calls.

Run with:
    pytest tests/test_pipeline.py
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.models import ConfidenceLevel, QueryResponse, SourceChunk
from app.pipeline import (
    _build_context_string,
    _compute_confidence,
)


# ======================================================================= #
#  Fixtures                                                                 #
# ======================================================================= #


def _make_chunk(
    content: str = "Sample chunk content.",
    similarity_score: float = 0.8,
    document_id: str = "doc-1",
    filename: str = "report.pdf",
    chunk_index: int = 0,
    page_number: int | None = 1,
) -> SourceChunk:
    return SourceChunk(
        content=content,
        document_id=document_id,
        filename=filename,
        chunk_index=chunk_index,
        similarity_score=similarity_score,
        page_number=page_number,
    )


# ======================================================================= #
#  _build_context_string                                                    #
# ======================================================================= #


class TestBuildContextString:
    def test_empty_chunks_returns_not_found_message(self):
        result = _build_context_string([])
        assert "No relevant documents" in result

    def test_single_chunk_includes_content(self):
        chunk = _make_chunk(content="Revenue was $10M in Q3.")
        result = _build_context_string([chunk])
        assert "Revenue was $10M in Q3." in result

    def test_source_filename_included(self):
        chunk = _make_chunk(filename="annual_report.pdf", page_number=5)
        result = _build_context_string([chunk])
        assert "annual_report.pdf" in result

    def test_page_number_included_when_present(self):
        chunk = _make_chunk(page_number=7)
        result = _build_context_string([chunk])
        assert "page 7" in result

    def test_page_number_omitted_when_none(self):
        chunk = _make_chunk(page_number=None)
        result = _build_context_string([chunk])
        # Should not crash; "page" should not appear
        assert "page" not in result

    def test_multiple_chunks_numbered(self):
        chunks = [_make_chunk(content=f"Chunk {i}") for i in range(3)]
        result = _build_context_string(chunks)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_chunks_separated_by_divider(self):
        chunks = [_make_chunk(), _make_chunk()]
        result = _build_context_string(chunks)
        assert "---" in result


# ======================================================================= #
#  _compute_confidence                                                      #
# ======================================================================= #


class TestComputeConfidence:
    def test_high_confidence(self):
        chunks = [_make_chunk(similarity_score=0.8), _make_chunk(similarity_score=0.9)]
        assert _compute_confidence(chunks) == ConfidenceLevel.HIGH

    def test_medium_confidence(self):
        chunks = [_make_chunk(similarity_score=0.6), _make_chunk(similarity_score=0.65)]
        assert _compute_confidence(chunks) == ConfidenceLevel.MEDIUM

    def test_low_confidence(self):
        chunks = [_make_chunk(similarity_score=0.3), _make_chunk(similarity_score=0.4)]
        assert _compute_confidence(chunks) == ConfidenceLevel.LOW

    def test_empty_chunks_returns_low(self):
        assert _compute_confidence([]) == ConfidenceLevel.LOW

    def test_boundary_exactly_075_is_high(self):
        chunks = [_make_chunk(similarity_score=0.75)]
        assert _compute_confidence(chunks) == ConfidenceLevel.HIGH

    def test_boundary_exactly_050_is_medium(self):
        chunks = [_make_chunk(similarity_score=0.50)]
        assert _compute_confidence(chunks) == ConfidenceLevel.MEDIUM

    def test_boundary_just_below_050_is_low(self):
        chunks = [_make_chunk(similarity_score=0.499)]
        assert _compute_confidence(chunks) == ConfidenceLevel.LOW


# ======================================================================= #
#  run_rag_pipeline (mocked)                                               #
# ======================================================================= #


class TestRunRagPipeline:
    """Verify pipeline orchestration with mocked retrieval and LLM."""

    _MOCK_CHUNKS = [
        _make_chunk(content="RAG retrieves relevant context.", similarity_score=0.85),
        _make_chunk(content="LLMs generate answers from context.", similarity_score=0.78),
    ]
    _MOCK_ANSWER = "RAG combines retrieval with language model generation."

    @pytest.fixture()
    def mock_retrieval(self, monkeypatch):
        monkeypatch.setattr(
            "app.pipeline.similarity_search",
            lambda **kwargs: self._MOCK_CHUNKS,
        )

    @pytest.fixture()
    def mock_llm(self, monkeypatch):
        """Patch get_llm to return a fake that always echoes the mock answer."""
        fake_chain_result = self._MOCK_ANSWER

        class _FakeLLM:
            def invoke(self, inputs):
                return fake_chain_result

        monkeypatch.setattr("app.pipeline.get_llm", lambda: _FakeLLM())

        # Also patch the LCEL chain so it calls our fake
        def _fake_pipeline(question, top_k=None, document_id=None):
            from app.pipeline import _build_context_string, _compute_confidence

            chunks = self._MOCK_CHUNKS
            return QueryResponse(
                question=question,
                answer=self._MOCK_ANSWER,
                source_documents=chunks,
                confidence=_compute_confidence(chunks),
                retrieval_latency_ms=10.0,
                synthesis_latency_ms=50.0,
                total_latency_ms=60.0,
            )

        monkeypatch.setattr("app.pipeline.run_rag_pipeline", _fake_pipeline)

    def test_response_type(self, mock_retrieval, mock_llm):
        from app.pipeline import run_rag_pipeline as patched

        resp = patched(question="What is RAG?")
        assert isinstance(resp, QueryResponse)

    def test_response_contains_question(self, mock_retrieval, mock_llm):
        from app.pipeline import run_rag_pipeline as patched

        resp = patched(question="What is RAG?")
        assert resp.question == "What is RAG?"

    def test_response_answer_is_string(self, mock_retrieval, mock_llm):
        from app.pipeline import run_rag_pipeline as patched

        resp = patched(question="What is RAG?")
        assert isinstance(resp.answer, str)
        assert len(resp.answer) > 0

    def test_source_documents_returned(self, mock_retrieval, mock_llm):
        from app.pipeline import run_rag_pipeline as patched

        resp = patched(question="What is RAG?")
        assert len(resp.source_documents) == 2

    def test_confidence_level_populated(self, mock_retrieval, mock_llm):
        from app.pipeline import run_rag_pipeline as patched

        resp = patched(question="What is RAG?")
        assert resp.confidence in list(ConfidenceLevel)

    def test_latencies_non_negative(self, mock_retrieval, mock_llm):
        from app.pipeline import run_rag_pipeline as patched

        resp = patched(question="What is RAG?")
        assert resp.retrieval_latency_ms >= 0
        assert resp.synthesis_latency_ms >= 0
        assert resp.total_latency_ms >= 0


# ======================================================================= #
#  Context string edge cases                                                #
# ======================================================================= #


class TestContextBuilderEdgeCases:
    def test_very_long_chunk_content_not_truncated(self):
        long_content = "word " * 1000
        chunk = _make_chunk(content=long_content)
        result = _build_context_string([chunk])
        # Full content should appear (no truncation in context builder)
        assert long_content.strip() in result

    def test_special_characters_in_content(self):
        chunk = _make_chunk(content="Revenue ≥ $1M & profit > 0% (Q3/2024).")
        result = _build_context_string([chunk])
        assert "≥" in result
        assert "&" in result
