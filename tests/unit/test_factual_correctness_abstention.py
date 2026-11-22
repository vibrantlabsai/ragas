"""Regression tests for FactualCorrectness NaN-on-abstention behavior.

See: https://github.com/vibrantlabsai/ragas/issues/2806

FactualCorrectness used to return a hard 0.0 whenever claim decomposition
produced no claims on either side (tp == fp == fn == 0) -- for example when
`response` and `reference` are the same correct abstention, such as
"Not found in provided context.". That silently scored a correct abstention
as total failure and dragged aggregate means down with no signal that the
metric had nothing to measure.

Faithfulness already handles the equivalent "no statements generated" case
by returning NaN. These tests pin FactualCorrectness (legacy and v2
collections implementations) to the same NaN-on-nothing-measured behavior,
while leaving the meaningful zero cases (one side abstains, the other
doesn't) unchanged.
"""

import math
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics._factual_correctness import (
    FactualCorrectness as LegacyFactualCorrectness,
)
from ragas.metrics.collections import FactualCorrectness
from ragas.metrics.collections.factual_correctness.util import (
    ClaimDecompositionOutput,
)


class MockInstructorLLM(InstructorBaseRagasLLM):
    """Mock InstructorBaseRagasLLM whose claim decomposition yields no claims."""

    def __init__(self):
        self.agenerate = AsyncMock(return_value=ClaimDecompositionOutput(claims=[]))
        self.generate = MagicMock()

    def generate(self, prompt, response_model):
        return self.generate(prompt, response_model)

    async def agenerate(self, prompt, response_model):
        return await self.agenerate(prompt, response_model)


@pytest.fixture
def empty_claims_llm():
    return MockInstructorLLM()


class TestFactualCorrectnessCollectionsAbstention:
    """v2 (ragas.metrics.collections) FactualCorrectness."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["precision", "recall", "f1"])
    async def test_both_sides_empty_returns_nan(self, empty_claims_llm, mode):
        """Identical abstentions decompose to zero claims on both sides -> NaN."""
        metric = FactualCorrectness(llm=empty_claims_llm, mode=mode)
        abstention = "Not found in provided context."

        result = await metric.ascore(response=abstention, reference=abstention)

        assert math.isnan(result.value), (
            f"mode={mode}: expected NaN when no claims were decomposable, "
            f"got {result.value}"
        )

    @pytest.mark.asyncio
    async def test_response_abstains_reference_has_claims_stays_zero(self):
        """Response yields no claims but reference does (fn > 0) -> still 0.0, not NaN."""
        from ragas.metrics.collections.factual_correctness.util import (
            NLIStatementOutput,
            StatementFaithfulnessAnswer,
        )

        llm = MockInstructorLLM()
        # 1st call: decompose response's claims -> none (precision direction skipped
        #    verification since there are no claims to verify).
        # 2nd call: decompose reference's claims -> one claim (recall direction).
        # 3rd call: verify that reference claim against the response -> unsupported.
        llm.agenerate = AsyncMock(
            side_effect=[
                ClaimDecompositionOutput(claims=[]),
                ClaimDecompositionOutput(claims=["Paris is the capital of France."]),
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement="Paris is the capital of France.",
                            reason="The response is an abstention and does not "
                            "mention this claim.",
                            verdict=0,
                        )
                    ]
                ),
            ]
        )
        metric = FactualCorrectness(llm=llm, mode="recall")

        result = await metric.ascore(
            response="Not found in provided context.",
            reference="Paris is the capital of France.",
        )

        # fn > 0 here, so this is a *meaningful* zero and must not become NaN.
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_normal_supported_claim_is_unaffected(self):
        """Sanity check: the guard must not affect the ordinary scoring path."""
        from ragas.metrics.collections.factual_correctness.util import (
            NLIStatementOutput,
            StatementFaithfulnessAnswer,
        )

        llm = MockInstructorLLM()
        llm.agenerate = AsyncMock(
            side_effect=[
                ClaimDecompositionOutput(claims=["Paris is the capital of France."]),
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement="Paris is the capital of France.",
                            reason="Directly supported by the reference.",
                            verdict=1,
                        )
                    ]
                ),
            ]
        )
        metric = FactualCorrectness(llm=llm, mode="precision")

        result = await metric.ascore(
            response="Paris is the capital of France.",
            reference="Paris is the capital of France.",
        )

        assert result.value == 1.0


class TestFactualCorrectnessLegacyAbstention:
    """Legacy (ragas.metrics._factual_correctness) FactualCorrectness."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["precision", "recall", "f1"])
    async def test_both_sides_empty_returns_nan(self, monkeypatch, mode):
        metric = LegacyFactualCorrectness(llm=object(), mode=mode)

        async def fake_decompose_and_verify_claims(reference, response, callbacks):
            return np.array([], dtype=bool)

        monkeypatch.setattr(
            metric,
            "decompose_and_verify_claims",
            fake_decompose_and_verify_claims,
        )

        sample = SingleTurnSample(
            response="Not found in provided context.",
            reference="Not found in provided context.",
        )
        score = await metric._single_turn_ascore(sample, None)

        assert math.isnan(score), (
            f"mode={mode}: expected NaN when no claims were decomposable, got {score}"
        )

    @pytest.mark.asyncio
    async def test_meaningful_zero_is_unaffected(self, monkeypatch):
        """fn > 0 (one side has claims, the other doesn't) must remain a real 0.0."""
        metric = LegacyFactualCorrectness(llm=object(), mode="recall")

        calls = {"n": 0}

        async def fake_decompose_and_verify_claims(reference, response, callbacks):
            # First call (precision direction): response -> reference, no claims.
            # Second call (recall direction): reference -> response, one unsupported claim.
            calls["n"] += 1
            if calls["n"] == 1:
                return np.array([], dtype=bool)
            return np.array([False], dtype=bool)

        monkeypatch.setattr(
            metric,
            "decompose_and_verify_claims",
            fake_decompose_and_verify_claims,
        )

        sample = SingleTurnSample(
            response="Not found in provided context.",
            reference="Paris is the capital of France.",
        )
        score = await metric._single_turn_ascore(sample, None)

        assert score == 0.0
