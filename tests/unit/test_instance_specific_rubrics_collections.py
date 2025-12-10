"""Tests for InstanceSpecificRubrics metric (collections implementation)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections.instance_specific_rubrics import InstanceSpecificRubrics
from ragas.metrics.collections.instance_specific_rubrics.util import (
    InstanceRubricScoreOutput,
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


@pytest.fixture
def sample_rubrics():
    """Fixture providing sample rubrics."""
    return {
        "score1_description": "The response is completely incorrect",
        "score2_description": "The response has major errors",
        "score3_description": "The response is partially correct",
        "score4_description": "The response is mostly correct",
        "score5_description": "The response is fully correct",
    }


class TestInstanceSpecificRubricsCollections:
    """Test cases for InstanceSpecificRubrics metric from collections."""

    @pytest.mark.asyncio
    async def test_perfect_score(self, mock_llm, sample_rubrics):
        """Test case where LLM returns perfect score."""
        mock_llm.agenerate.return_value = InstanceRubricScoreOutput(
            feedback="The response is fully correct and comprehensive.",
            score=5,
        )

        metric = InstanceSpecificRubrics(llm=mock_llm)
        result = await metric.ascore(
            user_input="What is 2+2?",
            response="4",
            rubrics=sample_rubrics,
        )

        assert result.value == 5.0
        assert "correct" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_low_score(self, mock_llm, sample_rubrics):
        """Test case where LLM returns low score."""
        mock_llm.agenerate.return_value = InstanceRubricScoreOutput(
            feedback="The response is completely incorrect.",
            score=1,
        )

        metric = InstanceSpecificRubrics(llm=mock_llm)
        result = await metric.ascore(
            user_input="What is 2+2?",
            response="10",
            rubrics=sample_rubrics,
        )

        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_medium_score(self, mock_llm, sample_rubrics):
        """Test case with medium score."""
        mock_llm.agenerate.return_value = InstanceRubricScoreOutput(
            feedback="The response is partially correct but lacks detail.",
            score=3,
        )

        metric = InstanceSpecificRubrics(llm=mock_llm)
        result = await metric.ascore(
            user_input="Explain photosynthesis.",
            response="Plants make food from sunlight.",
            rubrics=sample_rubrics,
        )

        assert result.value == 3.0

    @pytest.mark.asyncio
    async def test_with_reference(self, mock_llm, sample_rubrics):
        """Test evaluation with reference answer."""
        mock_llm.agenerate.return_value = InstanceRubricScoreOutput(
            feedback="The response aligns well with the reference.",
            score=4,
        )

        metric = InstanceSpecificRubrics(llm=mock_llm)
        result = await metric.ascore(
            user_input="What is the capital of France?",
            response="The capital of France is Paris.",
            reference="Paris is the capital city of France.",
            rubrics=sample_rubrics,
        )

        assert result.value == 4.0

    @pytest.mark.asyncio
    async def test_with_contexts(self, mock_llm, sample_rubrics):
        """Test with retrieved and reference contexts."""
        mock_llm.agenerate.return_value = InstanceRubricScoreOutput(
            feedback="The response uses context appropriately.",
            score=5,
        )

        metric = InstanceSpecificRubrics(llm=mock_llm)
        result = await metric.ascore(
            user_input="What is the capital of France?",
            response="Based on the context, Paris is the capital of France.",
            retrieved_contexts=["Paris is the capital of France."],
            reference_contexts=["France's capital is Paris."],
            rubrics=sample_rubrics,
        )

        assert result.value == 5.0

    @pytest.mark.asyncio
    async def test_different_rubrics_per_sample(self, mock_llm):
        """Test that different rubrics can be used for different samples."""
        mock_llm.agenerate.return_value = InstanceRubricScoreOutput(
            feedback="The email is highly professional.",
            score=5,
        )

        metric = InstanceSpecificRubrics(llm=mock_llm)

        # First sample with email rubrics
        email_rubrics = {
            "score1_description": "Unprofessional email",
            "score2_description": "Lacks proper formatting",
            "score3_description": "Acceptable but could be better",
            "score4_description": "Professional with minor issues",
            "score5_description": "Highly professional email",
        }

        result1 = await metric.ascore(
            user_input="Write a professional email",
            response="Dear Sir/Madam...",
            rubrics=email_rubrics,
        )

        # Second sample with code rubrics
        code_rubrics = {
            "score1_description": "Code doesn't work",
            "score2_description": "Code has bugs",
            "score3_description": "Code works but inefficient",
            "score4_description": "Good code with minor issues",
            "score5_description": "Excellent, clean code",
        }

        mock_llm.agenerate.return_value = InstanceRubricScoreOutput(
            feedback="The code is excellent and clean.",
            score=5,
        )

        result2 = await metric.ascore(
            user_input="Write a sorting function",
            response="def sort(arr): return sorted(arr)",
            rubrics=code_rubrics,
        )

        assert result1.value == 5.0
        assert result2.value == 5.0
        # Verify different rubrics were passed in prompts
        assert mock_llm.agenerate.call_count == 2

    @pytest.mark.asyncio
    async def test_rubrics_required(self, mock_llm):
        """Test that rubrics parameter is required."""
        metric = InstanceSpecificRubrics(llm=mock_llm)

        with pytest.raises(ValueError, match="rubrics must be provided"):
            await metric.ascore(
                user_input="Test question",
                response="Test response",
                rubrics={},
            )

    @pytest.mark.asyncio
    async def test_rubrics_in_prompt(self, mock_llm, sample_rubrics):
        """Test that rubrics are included in the prompt."""
        mock_llm.agenerate.return_value = InstanceRubricScoreOutput(
            feedback="Good response.",
            score=4,
        )

        metric = InstanceSpecificRubrics(llm=mock_llm)
        await metric.ascore(
            user_input="Test",
            response="Test response",
            rubrics=sample_rubrics,
        )

        # Verify the prompt contains rubrics
        call_args = mock_llm.agenerate.call_args
        prompt_str = call_args[0][0]
        assert "score1_description" in prompt_str
        assert "completely incorrect" in prompt_str

    def test_custom_name(self, mock_llm):
        """Test setting a custom metric name."""
        metric = InstanceSpecificRubrics(llm=mock_llm, name="my_instance_rubric")
        assert metric.name == "my_instance_rubric"

    def test_default_name(self, mock_llm):
        """Test default metric name."""
        metric = InstanceSpecificRubrics(llm=mock_llm)
        assert metric.name == "instance_specific_rubrics"

    @pytest.mark.asyncio
    async def test_feedback_in_result_reason(self, mock_llm, sample_rubrics):
        """Test that feedback is returned in result.reason."""
        expected_feedback = "This is detailed feedback about the response quality."
        mock_llm.agenerate.return_value = InstanceRubricScoreOutput(
            feedback=expected_feedback,
            score=4,
        )

        metric = InstanceSpecificRubrics(llm=mock_llm)
        result = await metric.ascore(
            user_input="Question",
            response="Answer",
            rubrics=sample_rubrics,
        )

        assert result.reason == expected_feedback

    def test_allowed_values_range(self, mock_llm):
        """Test that allowed values are set to 1-5 range."""
        metric = InstanceSpecificRubrics(llm=mock_llm)
        assert metric.allowed_values == (1.0, 5.0)

    @pytest.mark.asyncio
    async def test_minimal_inputs(self, mock_llm, sample_rubrics):
        """Test with only required rubrics and response."""
        mock_llm.agenerate.return_value = InstanceRubricScoreOutput(
            feedback="Evaluated response.",
            score=3,
        )

        metric = InstanceSpecificRubrics(llm=mock_llm)
        result = await metric.ascore(
            response="Just a response",
            rubrics=sample_rubrics,
        )

        assert result.value == 3.0

    @pytest.mark.asyncio
    async def test_custom_score_range_rubrics(self, mock_llm):
        """Test with rubrics using different score range (1-3)."""
        custom_rubrics = {
            "score1_description": "Poor",
            "score2_description": "Average",
            "score3_description": "Excellent",
        }

        mock_llm.agenerate.return_value = InstanceRubricScoreOutput(
            feedback="Excellent work.",
            score=3,
        )

        metric = InstanceSpecificRubrics(llm=mock_llm)
        result = await metric.ascore(
            user_input="Test",
            response="Test response",
            rubrics=custom_rubrics,
        )

        assert result.value == 3.0
