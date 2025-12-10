"""Tests for DomainSpecificRubrics metric (collections implementation)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections.domain_specific_rubrics import (
    DomainSpecificRubrics,
    RubricsScoreWithoutReference,
    RubricsScoreWithReference,
)
from ragas.metrics.collections.domain_specific_rubrics.util import (
    DEFAULT_REFERENCE_FREE_RUBRICS,
    DEFAULT_WITH_REFERENCE_RUBRICS,
    RubricScoreOutput,
)


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


class TestDomainSpecificRubricsCollections:
    """Test cases for DomainSpecificRubrics metric from collections."""

    @pytest.mark.asyncio
    async def test_perfect_score(self, mock_llm):
        """Test case where LLM returns perfect score."""
        mock_llm.agenerate.return_value = RubricScoreOutput(
            feedback="The response is completely accurate and thorough.",
            score=5,
        )

        metric = DomainSpecificRubrics(llm=mock_llm)
        result = await metric.ascore(
            user_input="What is the capital of France?",
            response="The capital of France is Paris.",
        )

        assert result.value == 5.0
        assert "accurate" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_low_score(self, mock_llm):
        """Test case where LLM returns low score."""
        mock_llm.agenerate.return_value = RubricScoreOutput(
            feedback="The response is entirely incorrect.",
            score=1,
        )

        metric = DomainSpecificRubrics(llm=mock_llm)
        result = await metric.ascore(
            user_input="What is the capital of France?",
            response="The capital of France is London.",
        )

        assert result.value == 1.0
        assert "incorrect" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_medium_score(self, mock_llm):
        """Test case with medium score."""
        mock_llm.agenerate.return_value = RubricScoreOutput(
            feedback="The response is mostly accurate but lacks detail.",
            score=3,
        )

        metric = DomainSpecificRubrics(llm=mock_llm)
        result = await metric.ascore(
            user_input="Explain photosynthesis.",
            response="Photosynthesis is when plants make food.",
        )

        assert result.value == 3.0

    @pytest.mark.asyncio
    async def test_with_reference(self, mock_llm):
        """Test reference-based evaluation."""
        mock_llm.agenerate.return_value = RubricScoreOutput(
            feedback="The response aligns well with the reference.",
            score=4,
        )

        metric = DomainSpecificRubrics(llm=mock_llm, with_reference=True)
        result = await metric.ascore(
            user_input="What is the capital of France?",
            response="The capital of France is Paris.",
            reference="Paris is the capital and largest city of France.",
        )

        assert result.value == 4.0

    @pytest.mark.asyncio
    async def test_with_contexts(self, mock_llm):
        """Test with retrieved and reference contexts."""
        mock_llm.agenerate.return_value = RubricScoreOutput(
            feedback="The response uses context appropriately.",
            score=5,
        )

        metric = DomainSpecificRubrics(llm=mock_llm)
        result = await metric.ascore(
            user_input="What is the capital of France?",
            response="Based on the context, Paris is the capital of France.",
            retrieved_contexts=["Paris is the capital of France."],
            reference_contexts=["France's capital is Paris."],
        )

        assert result.value == 5.0

    @pytest.mark.asyncio
    async def test_custom_rubrics(self, mock_llm):
        """Test with custom rubrics."""
        custom_rubrics = {
            "score1_description": "Completely wrong",
            "score2_description": "Mostly wrong",
            "score3_description": "Partially correct",
            "score4_description": "Mostly correct",
            "score5_description": "Fully correct",
        }

        mock_llm.agenerate.return_value = RubricScoreOutput(
            feedback="The answer is fully correct.",
            score=5,
        )

        metric = DomainSpecificRubrics(llm=mock_llm, rubrics=custom_rubrics)
        result = await metric.ascore(
            user_input="What is 2+2?",
            response="4",
        )

        assert result.value == 5.0
        # Verify the prompt contains custom rubrics
        call_args = mock_llm.agenerate.call_args
        prompt_str = call_args[0][0]
        assert "Fully correct" in prompt_str

    @pytest.mark.asyncio
    async def test_rubrics_score_without_reference_class(self, mock_llm):
        """Test RubricsScoreWithoutReference convenience class."""
        mock_llm.agenerate.return_value = RubricScoreOutput(
            feedback="Good response.",
            score=4,
        )

        metric = RubricsScoreWithoutReference(llm=mock_llm)
        assert metric.name == "rubrics_score_without_reference"
        assert metric.with_reference is False

        result = await metric.ascore(
            user_input="Test question",
            response="Test response",
        )

        assert result.value == 4.0

    @pytest.mark.asyncio
    async def test_rubrics_score_with_reference_class(self, mock_llm):
        """Test RubricsScoreWithReference convenience class."""
        mock_llm.agenerate.return_value = RubricScoreOutput(
            feedback="Matches reference well.",
            score=5,
        )

        metric = RubricsScoreWithReference(llm=mock_llm)
        assert metric.name == "rubrics_score_with_reference"
        assert metric.with_reference is True

        result = await metric.ascore(
            user_input="Test question",
            response="Test response",
            reference="Reference answer",
        )

        assert result.value == 5.0

    def test_default_rubrics_without_reference(self, mock_llm):
        """Test that default rubrics are set correctly for reference-free mode."""
        metric = DomainSpecificRubrics(llm=mock_llm, with_reference=False)
        assert metric.rubrics == DEFAULT_REFERENCE_FREE_RUBRICS

    def test_default_rubrics_with_reference(self, mock_llm):
        """Test that default rubrics are set correctly for reference-based mode."""
        metric = DomainSpecificRubrics(llm=mock_llm, with_reference=True)
        assert metric.rubrics == DEFAULT_WITH_REFERENCE_RUBRICS

    def test_rubrics_in_prompt(self, mock_llm):
        """Test that rubrics are included in the prompt instruction."""
        metric = DomainSpecificRubrics(llm=mock_llm)
        assert "Scoring Rubrics:" in metric.scoring_prompt.instruction
        assert "score1_description" in metric.scoring_prompt.instruction

    def test_custom_name(self, mock_llm):
        """Test setting a custom metric name."""
        metric = DomainSpecificRubrics(llm=mock_llm, name="my_custom_rubric")
        assert metric.name == "my_custom_rubric"

    @pytest.mark.asyncio
    async def test_all_optional_inputs(self, mock_llm):
        """Test that all inputs are optional."""
        mock_llm.agenerate.return_value = RubricScoreOutput(
            feedback="Cannot evaluate without inputs.",
            score=1,
        )

        metric = DomainSpecificRubrics(llm=mock_llm)
        # This should not raise even with minimal inputs
        result = await metric.ascore(response="Just a response")

        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_feedback_in_result_reason(self, mock_llm):
        """Test that feedback is returned in result.reason."""
        expected_feedback = "This is detailed feedback about the response quality."
        mock_llm.agenerate.return_value = RubricScoreOutput(
            feedback=expected_feedback,
            score=4,
        )

        metric = DomainSpecificRubrics(llm=mock_llm)
        result = await metric.ascore(
            user_input="Question",
            response="Answer",
        )

        assert result.reason == expected_feedback

    def test_allowed_values_range(self, mock_llm):
        """Test that allowed values are set to 1-5 range."""
        metric = DomainSpecificRubrics(llm=mock_llm)
        assert metric.allowed_values == (1.0, 5.0)
