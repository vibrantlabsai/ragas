"""Tests for FactualCorrectness metric (collections implementation).

Focus: empty-claims handling. When claim decomposition yields no claims on
either side (e.g. a correct abstention like "Not found in provided context."
compared against the same abstention reference), the metric returns NaN,
consistent with Faithfulness, instead of a hard 0.0. Mixed cases keep their
meaningful zeros. See issue #2806.
"""

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections.factual_correctness import FactualCorrectness
from ragas.metrics.collections.factual_correctness.util import (
    ClaimDecompositionOutput,
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
)

ABSTENTION = "Not found in provided context."


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


def nli(*verdicts):
    """NLI output with one statement per verdict."""
    return NLIStatementOutput(
        statements=[
            StatementFaithfulnessAnswer(
                statement=f"claim {i}", reason="mock", verdict=v
            )
            for i, v in enumerate(verdicts)
        ]
    )


class TestFactualCorrectnessEmptyClaims:
    """Empty-claims (abstention) handling; call order per mode is deterministic:
    decompose(response) [-> NLI vs reference], then for recall/f1
    decompose(reference) [-> NLI vs response]."""

    @pytest.mark.asyncio
    async def test_both_texts_yield_no_claims_returns_nan_f1(self, mock_llm):
        """Correct abstention (response == reference): NaN, not total failure."""
        mock_llm.agenerate.side_effect = [
            ClaimDecompositionOutput(claims=[]),  # decompose response
            ClaimDecompositionOutput(claims=[]),  # decompose reference
        ]
        metric = FactualCorrectness(llm=mock_llm)
        result = await metric.ascore(response=ABSTENTION, reference=ABSTENTION)
        assert math.isnan(result.value)

    @pytest.mark.asyncio
    async def test_both_texts_yield_no_claims_returns_nan_precision(self, mock_llm):
        mock_llm.agenerate.side_effect = [
            ClaimDecompositionOutput(claims=[]),  # decompose response only
        ]
        metric = FactualCorrectness(llm=mock_llm, mode="precision")
        result = await metric.ascore(response=ABSTENTION, reference=ABSTENTION)
        assert math.isnan(result.value)

    @pytest.mark.asyncio
    async def test_both_texts_yield_no_claims_returns_nan_recall(self, mock_llm):
        mock_llm.agenerate.side_effect = [
            ClaimDecompositionOutput(claims=[]),  # decompose response
            ClaimDecompositionOutput(claims=[]),  # decompose reference
        ]
        metric = FactualCorrectness(llm=mock_llm, mode="recall")
        result = await metric.ascore(response=ABSTENTION, reference=ABSTENTION)
        assert math.isnan(result.value)

    @pytest.mark.asyncio
    async def test_response_abstains_reference_factual_scores_zero(self, mock_llm):
        """fn > 0 keeps its meaningful 0.0: response covered none of the reference."""
        mock_llm.agenerate.side_effect = [
            ClaimDecompositionOutput(claims=[]),  # response -> no claims, NLI skipped
            ClaimDecompositionOutput(claims=["The Pro plan costs 40 dollars"]),
            nli(False),  # reference claim not supported by the abstention
        ]
        metric = FactualCorrectness(llm=mock_llm)
        result = await metric.ascore(
            response=ABSTENTION, reference="The Pro plan costs 40 dollars."
        )
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_response_factual_reference_abstains_scores_zero(self, mock_llm):
        """fp > 0 keeps its meaningful 0.0: hallucination where abstention was required."""
        mock_llm.agenerate.side_effect = [
            ClaimDecompositionOutput(claims=["Uptime is 99.9 percent"]),
            nli(False),  # response claim unsupported by the abstention reference
            ClaimDecompositionOutput(claims=[]),  # reference -> no claims
        ]
        metric = FactualCorrectness(llm=mock_llm)
        result = await metric.ascore(
            response="Uptime is 99.9 percent.", reference=ABSTENTION
        )
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_supported_claims_still_score_one(self, mock_llm):
        """Regression guard: the normal scoring path is unchanged."""
        mock_llm.agenerate.side_effect = [
            ClaimDecompositionOutput(claims=["Paris is the capital of France"]),
            nli(True),
            ClaimDecompositionOutput(claims=["France's capital is Paris"]),
            nli(True),
        ]
        metric = FactualCorrectness(llm=mock_llm)
        result = await metric.ascore(
            response="Paris is the capital of France.",
            reference="France's capital is Paris.",
        )
        assert result.value == 1.0
