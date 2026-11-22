"""Tests for FactualCorrectness metric (collections implementation)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import FactualCorrectness


class MockInstructorLLM(InstructorBaseRagasLLM):
    """Mock InstructorBaseRagasLLM for tests that bypass LLM calls."""

    def __init__(self):
        self.generate_mock = MagicMock()
        self.agenerate_mock = AsyncMock()

    def generate(self, prompt, response_model):
        return self.generate_mock(prompt, response_model)

    async def agenerate(self, prompt, response_model):
        return await self.agenerate_mock(prompt, response_model)


class TestFactualCorrectnessCollections:
    """Test cases for FactualCorrectness metric from collections."""

    def test_recall_uses_reference_claim_coverage(self):
        """Recall should measure reference claims covered by the response."""
        metric = FactualCorrectness(llm=MockInstructorLLM(), mode="recall")
        metric._decompose_and_verify_claims = AsyncMock(
            side_effect=[
                np.array([True, True]),
                np.array([True, False]),
            ]
        )

        result = asyncio.run(metric.ascore(response="response", reference="reference"))

        assert result.value == 0.5
