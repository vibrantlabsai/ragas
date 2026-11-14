"""Tests for SummaryScore metric (collections implementation)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import SummaryScore


class MockInstructorLLM(InstructorBaseRagasLLM):
    """Mock implementation of InstructorBaseRagasLLM for testing."""

    def __init__(self):
        self.agenerate = AsyncMock()
        self.generate = MagicMock()

    def generate(self, prompt, response_model):
        return self.generate(prompt, response_model)

    async def agenerate(self, prompt, response_model):
        return await self.agenerate(prompt, response_model)


@pytest.fixture
def mock_llm():
    """Fixture providing a mock LLM."""
    return MockInstructorLLM()


class TestSummaryScoreCollectionsQaScore:
    """``_compute_qa_score`` needs no LLM — it's pure arithmetic over the
    already-generated answers list."""

    def test_all_correct_answers_scores_one(self, mock_llm):
        metric = SummaryScore(llm=mock_llm)
        assert metric._compute_qa_score(["1", "1", "1"]) == 1.0

    def test_no_correct_answers_scores_zero(self, mock_llm):
        metric = SummaryScore(llm=mock_llm)
        assert metric._compute_qa_score(["0", "0"]) == 0.0

    def test_mixed_answers_computes_ratio(self, mock_llm):
        metric = SummaryScore(llm=mock_llm)
        assert metric._compute_qa_score(["1", "0", "1", "0"]) == 0.5

    def test_empty_answers_scores_zero_instead_of_raising(self, mock_llm):
        """Regression: keyphrase or question extraction can legitimately
        return an empty list (already handled by logging an error and
        continuing), which used to cascade into `answers == []` and a
        ZeroDivisionError here instead of a score."""
        metric = SummaryScore(llm=mock_llm)
        assert metric._compute_qa_score([]) == 0.0
