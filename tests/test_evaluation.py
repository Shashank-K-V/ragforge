"""
test_evaluation.py — Tests for the evaluation framework.

All tests mock run_rag_pipeline so no LLM or vector store is needed.
"""

from __future__ import annotations

from unittest.mock import patch

from app.evaluation import TEST_CASES, _evaluate_single_case, run_evaluation
from app.models import (
    ConfidenceLevel,
    EvalTestCase,
    EvaluationResponse,
    QueryResponse,
    SourceChunk,
)


# ======================================================================= #
#  Helpers                                                                  #
# ======================================================================= #


def _make_query_response(answer: str, doc_ids: list[str] | None = None) -> QueryResponse:
    doc_ids = doc_ids or ["default-doc"]
    chunks = [
        SourceChunk(
            content="Some chunk content.",
            document_id=did,
            filename="file.txt",
            chunk_index=i,
            similarity_score=0.8,
            page_number=None,
        )
        for i, did in enumerate(doc_ids)
    ]
    return QueryResponse(
        question="Test question",
        answer=answer,
        source_documents=chunks,
        confidence=ConfidenceLevel.HIGH,
        retrieval_latency_ms=10.0,
        synthesis_latency_ms=40.0,
        total_latency_ms=50.0,
    )


# ======================================================================= #
#  TEST_CASES structure                                                     #
# ======================================================================= #


class TestTestCasesStructure:
    def test_has_five_cases(self):
        assert len(TEST_CASES) == 5

    def test_all_have_questions(self):
        for case in TEST_CASES:
            assert isinstance(case.question, str)
            assert len(case.question) > 0

    def test_all_have_keywords(self):
        for case in TEST_CASES:
            assert len(case.expected_answer_keywords) >= 1

    def test_eval_test_case_is_pydantic_model(self):
        case = EvalTestCase(
            question="What is X?",
            expected_answer_keywords=["x", "y"],
            expected_source_document_id=None,
        )
        assert case.question == "What is X?"


# ======================================================================= #
#  _evaluate_single_case                                                    #
# ======================================================================= #


class TestEvaluateSingleCase:
    def test_answer_relevance_true_when_all_keywords_present(self):
        case = EvalTestCase(
            question="What is RAG?",
            expected_answer_keywords=["retrieval", "generation"],
            expected_source_document_id=None,
        )
        mock_resp = _make_query_response("RAG combines retrieval and generation.")

        with patch("app.evaluation.run_rag_pipeline", return_value=mock_resp):
            result = _evaluate_single_case(case)

        assert result.answer_relevance is True

    def test_answer_relevance_false_when_keyword_missing(self):
        case = EvalTestCase(
            question="What is RAG?",
            expected_answer_keywords=["retrieval", "missing_keyword_xyz"],
            expected_source_document_id=None,
        )
        mock_resp = _make_query_response("RAG uses retrieval.")

        with patch("app.evaluation.run_rag_pipeline", return_value=mock_resp):
            result = _evaluate_single_case(case)

        assert result.answer_relevance is False

    def test_keyword_matching_is_case_insensitive(self):
        case = EvalTestCase(
            question="?",
            expected_answer_keywords=["RETRIEVAL"],
            expected_source_document_id=None,
        )
        mock_resp = _make_query_response("This involves retrieval.")  # lowercase

        with patch("app.evaluation.run_rag_pipeline", return_value=mock_resp):
            result = _evaluate_single_case(case)

        assert result.answer_relevance is True

    def test_retrieval_hit_true_when_doc_found(self):
        case = EvalTestCase(
            question="?",
            expected_answer_keywords=["x"],
            expected_source_document_id="target-doc",
        )
        mock_resp = _make_query_response("x answer", doc_ids=["target-doc", "other"])

        with patch("app.evaluation.run_rag_pipeline", return_value=mock_resp):
            result = _evaluate_single_case(case)

        assert result.retrieval_hit is True

    def test_retrieval_hit_false_when_doc_not_found(self):
        case = EvalTestCase(
            question="?",
            expected_answer_keywords=["x"],
            expected_source_document_id="target-doc",
        )
        mock_resp = _make_query_response("x answer", doc_ids=["some-other-doc"])

        with patch("app.evaluation.run_rag_pipeline", return_value=mock_resp):
            result = _evaluate_single_case(case)

        assert result.retrieval_hit is False

    def test_retrieval_hit_true_when_no_expected_doc(self):
        """When expected_source_document_id is None, hit is always True."""
        case = EvalTestCase(
            question="?",
            expected_answer_keywords=["x"],
            expected_source_document_id=None,
        )
        mock_resp = _make_query_response("x answer", doc_ids=["random-doc"])

        with patch("app.evaluation.run_rag_pipeline", return_value=mock_resp):
            result = _evaluate_single_case(case)

        assert result.retrieval_hit is True

    def test_answer_snippet_truncated_to_200_chars(self):
        case = EvalTestCase(
            question="?",
            expected_answer_keywords=["x"],
            expected_source_document_id=None,
        )
        long_answer = "x " * 200  # 400 chars
        mock_resp = _make_query_response(long_answer)

        with patch("app.evaluation.run_rag_pipeline", return_value=mock_resp):
            result = _evaluate_single_case(case)

        assert len(result.answer_snippet) <= 200

    def test_latency_non_negative(self):
        case = TEST_CASES[0]
        mock_resp = _make_query_response("retrieval generation context")

        with patch("app.evaluation.run_rag_pipeline", return_value=mock_resp):
            result = _evaluate_single_case(case)

        assert result.latency_ms >= 0

    def test_pipeline_error_produces_zero_chunks_and_false_relevance(self):
        case = EvalTestCase(
            question="?",
            expected_answer_keywords=["key"],
            expected_source_document_id=None,
        )

        with patch("app.evaluation.run_rag_pipeline", side_effect=RuntimeError("LLM down")):
            result = _evaluate_single_case(case)

        assert result.retrieved_chunks == 0
        assert result.answer_relevance is False


# ======================================================================= #
#  run_evaluation                                                           #
# ======================================================================= #


class TestRunEvaluation:
    def _make_perfect_mock(self, keywords: list[str]) -> QueryResponse:
        """Returns a mock response that passes all keyword checks."""
        answer = " ".join(keywords)
        return _make_query_response(answer)

    def test_returns_evaluation_response(self):
        cases = [TEST_CASES[0]]
        mock_resp = _make_query_response("retrieval generation context")

        with patch("app.evaluation.run_rag_pipeline", return_value=mock_resp):
            result = run_evaluation(test_cases=cases)

        assert isinstance(result, EvaluationResponse)

    def test_total_cases_matches_input(self):
        cases = TEST_CASES[:3]
        mock_resp = _make_query_response("retrieval generation context embedding model sentence cosine similarity vector")

        with patch("app.evaluation.run_rag_pipeline", return_value=mock_resp):
            result = run_evaluation(test_cases=cases)

        assert result.total_cases == 3

    def test_rates_between_zero_and_one(self):
        mock_resp = _make_query_response("retrieval generation context")

        with patch("app.evaluation.run_rag_pipeline", return_value=mock_resp):
            result = run_evaluation(test_cases=TEST_CASES[:2])

        assert 0.0 <= result.retrieval_hit_rate <= 1.0
        assert 0.0 <= result.answer_relevance_rate <= 1.0

    def test_perfect_score_when_all_keywords_present(self):
        # Build an answer containing all keywords from all test cases
        all_keywords = [
            kw for case in TEST_CASES for kw in case.expected_answer_keywords
        ]
        answer = " ".join(all_keywords)
        mock_resp = _make_query_response(answer)

        with patch("app.evaluation.run_rag_pipeline", return_value=mock_resp):
            result = run_evaluation(test_cases=TEST_CASES)

        assert result.answer_relevance_rate == 1.0

    def test_zero_score_when_no_keywords_present(self):
        mock_resp = _make_query_response("The answer is forty-two.")

        with patch("app.evaluation.run_rag_pipeline", return_value=mock_resp):
            result = run_evaluation(test_cases=TEST_CASES)

        assert result.answer_relevance_rate == 0.0

    def test_mean_latency_positive(self):
        mock_resp = _make_query_response("retrieval generation context")

        with patch("app.evaluation.run_rag_pipeline", return_value=mock_resp):
            result = run_evaluation(test_cases=TEST_CASES[:2])

        assert result.mean_latency_ms >= 0

    def test_results_list_length_matches_total(self):
        mock_resp = _make_query_response("x")

        with patch("app.evaluation.run_rag_pipeline", return_value=mock_resp):
            result = run_evaluation(test_cases=TEST_CASES)

        assert len(result.results) == result.total_cases

    def test_default_test_cases_used_when_none_passed(self):
        mock_resp = _make_query_response("x")

        with patch("app.evaluation.run_rag_pipeline", return_value=mock_resp):
            result = run_evaluation()

        assert result.total_cases == len(TEST_CASES)
