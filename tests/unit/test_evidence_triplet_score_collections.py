"""Tests for EvidenceTripletScore metric (collections implementation)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import EvidenceTripletScore, NeutrosophicEvidenceTriplet
from ragas.metrics.collections.evidence_triplet_score.util import (
    EvidenceTripletOutput,
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


class TestEvidenceTripletScoreCollections:
    """Test cases for EvidenceTripletScore metric from collections."""

    def test_init_default_name(self, mock_llm):
        metric = EvidenceTripletScore(llm=mock_llm)
        assert metric.name == "evidence_triplet_score"

    def test_neutrosophic_alias_default_name(self, mock_llm):
        metric = NeutrosophicEvidenceTriplet(llm=mock_llm)
        assert metric.name == "neutrosophic_evidence_triplet"

    @pytest.mark.asyncio
    async def test_supported_response_has_high_truth(self, mock_llm):
        mock_llm.agenerate.return_value = EvidenceTripletOutput(
            truth=0.95,
            indeterminacy=0.05,
            falsity=0.0,
            reason="The context directly supports the response.",
        )

        metric = EvidenceTripletScore(llm=mock_llm)
        result = await metric.ascore(
            user_input="Where was Einstein born?",
            response="Einstein was born in Ulm, Germany.",
            retrieved_contexts=["Albert Einstein was born in Ulm, Germany."],
        )

        assert result.value == {
            "truth": 0.95,
            "indeterminacy": 0.05,
            "falsity": 0.0,
        }
        assert result.reason is not None
        assert "supports" in result.reason

    @pytest.mark.asyncio
    async def test_insufficient_context_has_high_indeterminacy(self, mock_llm):
        mock_llm.agenerate.return_value = EvidenceTripletOutput(
            truth=0.2,
            indeterminacy=0.85,
            falsity=0.1,
            reason="The retrieved context is insufficient to verify the answer.",
        )

        metric = EvidenceTripletScore(llm=mock_llm)
        result = await metric.ascore(
            user_input="What caused the outage?",
            response="The outage was caused by a database failover.",
            retrieved_contexts=["Root cause analysis is still pending."],
        )

        assert result.value["indeterminacy"] == 0.85
        assert result.value["truth"] == 0.2
        assert result.reason is not None
        assert "insufficient" in result.reason

    @pytest.mark.asyncio
    async def test_contradictory_context_has_high_falsity(self, mock_llm):
        mock_llm.agenerate.return_value = EvidenceTripletOutput(
            truth=0.1,
            indeterminacy=0.25,
            falsity=0.9,
            reason="The retrieved context contradicts the response.",
        )

        metric = EvidenceTripletScore(llm=mock_llm)
        result = await metric.ascore(
            user_input="What caused the outage?",
            response="The outage was caused by a database failover.",
            retrieved_contexts=[
                "The database failover completed successfully.",
                "The outage was caused by network routing errors.",
            ],
        )

        assert result.value["falsity"] == 0.9
        assert result.value["truth"] == 0.1
        assert result.reason is not None
        assert "contradicts" in result.reason

    @pytest.mark.asyncio
    async def test_mixed_evidence_preserves_independent_scores(self, mock_llm):
        mock_llm.agenerate.return_value = EvidenceTripletOutput(
            truth=0.7,
            indeterminacy=0.6,
            falsity=0.4,
            reason="Some claims are supported, while other claims remain unclear.",
        )

        metric = EvidenceTripletScore(llm=mock_llm)
        result = await metric.ascore(
            user_input="Summarize the incident.",
            response="The incident involved network errors and a database failover.",
            retrieved_contexts=[
                "Network routing errors were detected.",
                "Database status is not yet confirmed.",
            ],
        )

        assert result.value == {
            "truth": 0.7,
            "indeterminacy": 0.6,
            "falsity": 0.4,
        }

    @pytest.mark.asyncio
    async def test_missing_user_input_raises_value_error(self, mock_llm):
        metric = EvidenceTripletScore(llm=mock_llm)

        with pytest.raises(ValueError, match="user_input is missing"):
            await metric.ascore(
                user_input="",
                response="A response.",
                retrieved_contexts=["Context."],
            )

    @pytest.mark.asyncio
    async def test_missing_response_raises_value_error(self, mock_llm):
        metric = EvidenceTripletScore(llm=mock_llm)

        with pytest.raises(ValueError, match="response is missing"):
            await metric.ascore(
                user_input="Question?",
                response="",
                retrieved_contexts=["Context."],
            )

    @pytest.mark.asyncio
    async def test_missing_contexts_raises_value_error(self, mock_llm):
        metric = EvidenceTripletScore(llm=mock_llm)

        with pytest.raises(ValueError, match="retrieved_contexts is missing"):
            await metric.ascore(
                user_input="Question?",
                response="A response.",
                retrieved_contexts=[],
            )

    @pytest.mark.asyncio
    async def test_empty_contexts_raises_value_error(self, mock_llm):
        metric = EvidenceTripletScore(llm=mock_llm)

        with pytest.raises(ValueError, match="non-empty context"):
            await metric.ascore(
                user_input="Question?",
                response="A response.",
                retrieved_contexts=["  "],
            )

    @pytest.mark.asyncio
    async def test_invalid_output_reason_raises_value_error(self, mock_llm):
        mock_llm.agenerate.return_value = EvidenceTripletOutput(
            truth=0.4,
            indeterminacy=0.5,
            falsity=0.6,
            reason=" ",
        )

        metric = EvidenceTripletScore(llm=mock_llm)

        with pytest.raises(ValueError, match="reason is missing"):
            await metric.ascore(
                user_input="Question?",
                response="A response.",
                retrieved_contexts=["Context."],
            )
