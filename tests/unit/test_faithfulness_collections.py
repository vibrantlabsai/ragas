"""Tests for Faithfulness metric (collections implementation)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.metrics.collections.faithfulness.metric import (
    VERDICT_FAITHFUL,
    VERDICT_NOT_FAITHFUL,
)
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
    StatementGeneratorOutput,
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


SAMPLE_USER_INPUT = "What is the capital of France?"
SAMPLE_RESPONSE = "The capital of France is Paris. It is also known as the City of Light."
SAMPLE_CONTEXTS = ["Paris is the capital and most populous city of France."]


class TestFaithfulnessCollections:
    """Test cases for Faithfulness metric from collections."""

    @pytest.mark.asyncio
    async def test_perfect_faithfulness_returns_reason(self, mock_llm):
        """All statements faithful: score=1.0, reason includes all verdicts."""
        statements = [
            "The capital of France is Paris.",
            "Paris is known as the City of Light.",
        ]
        mock_llm.agenerate = AsyncMock(
            side_effect=[
                StatementGeneratorOutput(statements=statements),
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement=statements[0],
                            reason="The context states Paris is the capital of France.",
                            verdict=1,
                        ),
                        StatementFaithfulnessAnswer(
                            statement=statements[1],
                            reason="The context supports this claim.",
                            verdict=1,
                        ),
                    ]
                ),
            ]
        )

        metric = Faithfulness(llm=mock_llm)
        result = await metric.ascore(
            user_input=SAMPLE_USER_INPUT,
            response=SAMPLE_RESPONSE,
            retrieved_contexts=SAMPLE_CONTEXTS,
        )

        assert result.value == 1.0
        assert result.reason is not None
        assert "2 out of 2" in result.reason
        assert VERDICT_FAITHFUL in result.reason

    @pytest.mark.asyncio
    async def test_partial_faithfulness_returns_reason(self, mock_llm):
        """Mixed verdicts: score=0.5, reason shows faithful and unfaithful."""
        statements = [
            "The capital of France is Paris.",
            "Paris has a population of 10 million.",
        ]
        mock_llm.agenerate = AsyncMock(
            side_effect=[
                StatementGeneratorOutput(statements=statements),
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement=statements[0],
                            reason="The context confirms Paris is the capital.",
                            verdict=1,
                        ),
                        StatementFaithfulnessAnswer(
                            statement=statements[1],
                            reason="The context does not mention population numbers.",
                            verdict=0,
                        ),
                    ]
                ),
            ]
        )

        metric = Faithfulness(llm=mock_llm)
        result = await metric.ascore(
            user_input=SAMPLE_USER_INPUT,
            response=SAMPLE_RESPONSE,
            retrieved_contexts=SAMPLE_CONTEXTS,
        )

        assert result.value == 0.5
        assert result.reason is not None
        assert "1 out of 2" in result.reason
        assert VERDICT_FAITHFUL in result.reason
        assert VERDICT_NOT_FAITHFUL in result.reason

    @pytest.mark.asyncio
    async def test_zero_faithfulness_returns_reason(self, mock_llm):
        """No statements faithful: score=0.0, reason reflects all unfaithful."""
        statements = ["Paris has a population of 10 million."]
        mock_llm.agenerate = AsyncMock(
            side_effect=[
                StatementGeneratorOutput(statements=statements),
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement=statements[0],
                            reason="The context does not mention population.",
                            verdict=0,
                        ),
                    ]
                ),
            ]
        )

        metric = Faithfulness(llm=mock_llm)
        result = await metric.ascore(
            user_input=SAMPLE_USER_INPUT,
            response=SAMPLE_RESPONSE,
            retrieved_contexts=SAMPLE_CONTEXTS,
        )

        assert result.value == 0.0
        assert result.reason is not None
        assert "0 out of 1" in result.reason
        assert VERDICT_NOT_FAITHFUL in result.reason

    @pytest.mark.asyncio
    async def test_empty_statements_returns_nan_no_reason(self, mock_llm):
        """No statements generated: score=NaN, no reason field."""
        mock_llm.agenerate = AsyncMock(
            return_value=StatementGeneratorOutput(statements=[])
        )

        metric = Faithfulness(llm=mock_llm)
        result = await metric.ascore(
            user_input=SAMPLE_USER_INPUT,
            response=SAMPLE_RESPONSE,
            retrieved_contexts=SAMPLE_CONTEXTS,
        )

        import math

        assert math.isnan(result.value)
        assert result.reason is None

    @pytest.mark.asyncio
    async def test_reason_contains_individual_statement_reasons(self, mock_llm):
        """Verify each statement's reason text appears in the result reason."""
        statement_text = "The capital of France is Paris."
        reason_text = "The context explicitly states this fact."
        mock_llm.agenerate = AsyncMock(
            side_effect=[
                StatementGeneratorOutput(statements=[statement_text]),
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement=statement_text,
                            reason=reason_text,
                            verdict=1,
                        ),
                    ]
                ),
            ]
        )

        metric = Faithfulness(llm=mock_llm)
        result = await metric.ascore(
            user_input=SAMPLE_USER_INPUT,
            response=SAMPLE_RESPONSE,
            retrieved_contexts=SAMPLE_CONTEXTS,
        )

        assert statement_text in result.reason
        assert reason_text in result.reason

    @pytest.mark.asyncio
    async def test_reason_format_numbered_list(self, mock_llm):
        """Verify reason is formatted as a numbered list of verdicts."""
        statements = ["Statement A.", "Statement B.", "Statement C."]
        mock_llm.agenerate = AsyncMock(
            side_effect=[
                StatementGeneratorOutput(statements=statements),
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement=s,
                            reason=f"Reason for {s}",
                            verdict=1 if i % 2 == 0 else 0,
                        )
                        for i, s in enumerate(statements)
                    ]
                ),
            ]
        )

        metric = Faithfulness(llm=mock_llm)
        result = await metric.ascore(
            user_input=SAMPLE_USER_INPUT,
            response=SAMPLE_RESPONSE,
            retrieved_contexts=SAMPLE_CONTEXTS,
        )

        assert "1. [" in result.reason
        assert "2. [" in result.reason
        assert "3. [" in result.reason

    @pytest.mark.asyncio
    async def test_input_validation_missing_response(self, mock_llm):
        """Missing response raises ValueError."""
        metric = Faithfulness(llm=mock_llm)
        with pytest.raises(ValueError, match="response is missing"):
            await metric.ascore(
                user_input=SAMPLE_USER_INPUT,
                response="",
                retrieved_contexts=SAMPLE_CONTEXTS,
            )

    @pytest.mark.asyncio
    async def test_input_validation_missing_user_input(self, mock_llm):
        """Missing user_input raises ValueError."""
        metric = Faithfulness(llm=mock_llm)
        with pytest.raises(ValueError, match="user_input is missing"):
            await metric.ascore(
                user_input="",
                response=SAMPLE_RESPONSE,
                retrieved_contexts=SAMPLE_CONTEXTS,
            )

    @pytest.mark.asyncio
    async def test_input_validation_missing_contexts(self, mock_llm):
        """Missing retrieved_contexts raises ValueError."""
        metric = Faithfulness(llm=mock_llm)
        with pytest.raises(ValueError, match="retrieved_contexts is missing"):
            await metric.ascore(
                user_input=SAMPLE_USER_INPUT,
                response=SAMPLE_RESPONSE,
                retrieved_contexts=[],
            )

    def test_metric_name_default(self, mock_llm):
        """Default metric name is 'faithfulness'."""
        metric = Faithfulness(llm=mock_llm)
        assert metric.name == "faithfulness"

    def test_metric_name_custom(self, mock_llm):
        """Custom metric name is set correctly."""
        metric = Faithfulness(llm=mock_llm, name="custom_faithfulness")
        assert metric.name == "custom_faithfulness"


class TestBuildReason:
    """Test _build_reason method directly."""

    def _make_metric(self):
        """Create a Faithfulness metric with mock LLM."""
        mock_llm = MockInstructorLLM()
        return Faithfulness(llm=mock_llm)

    def test_empty_verdicts(self):
        """Empty verdicts list returns descriptive message."""
        metric = self._make_metric()
        verdicts = NLIStatementOutput(statements=[])
        reason = metric._build_reason(verdicts, float("nan"))
        assert "No statements" in reason

    def test_single_faithful(self):
        """Single faithful statement formats correctly."""
        metric = self._make_metric()
        verdicts = NLIStatementOutput(
            statements=[
                StatementFaithfulnessAnswer(
                    statement="Paris is the capital.",
                    reason="Supported by context.",
                    verdict=1,
                )
            ]
        )
        reason = metric._build_reason(verdicts, 1.0)
        assert "1 out of 1" in reason
        assert VERDICT_FAITHFUL in reason
        assert "Paris is the capital." in reason
        assert "Supported by context." in reason

    def test_mixed_verdicts_format(self):
        """Mixed verdicts show both faithful and not faithful labels."""
        metric = self._make_metric()
        verdicts = NLIStatementOutput(
            statements=[
                StatementFaithfulnessAnswer(
                    statement="Correct fact.",
                    reason="Matches context.",
                    verdict=1,
                ),
                StatementFaithfulnessAnswer(
                    statement="Wrong claim.",
                    reason="Not in context.",
                    verdict=0,
                ),
            ]
        )
        reason = metric._build_reason(verdicts, 0.5)
        assert "1 out of 2" in reason
        assert f"[{VERDICT_FAITHFUL}]" in reason
        assert f"[{VERDICT_NOT_FAITHFUL}]" in reason
