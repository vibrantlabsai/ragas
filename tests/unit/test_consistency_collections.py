"""Tests for AnswerConsistency and RetrievalConsistency (collections implementation)."""

import itertools
from unittest.mock import AsyncMock, MagicMock

import pytest

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import AnswerConsistency, RetrievalConsistency
from ragas.metrics.collections.consistency.util import (
    AnswerAgreementOutput,
    jaccard,
    sample_run_pairs,
    word_ngrams,
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


def agreement(**overrides) -> AnswerAgreementOutput:
    scores = dict(core_facts=5, specifics=5, supporting_detail=5, stance=5, format=5)
    scores.update(overrides)
    return AnswerAgreementOutput(reason="judge reason", **scores)


@pytest.fixture
def mock_llm():
    return MockInstructorLLM()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class TestSampleRunPairs:
    @pytest.mark.parametrize("n", [3, 4, 5, 8, 11])
    def test_default_is_n_pairs_covering_every_run_twice(self, n):
        pairs = sample_run_pairs(n)
        assert len(pairs) == len(set(pairs)) == min(n, n * (n - 1) // 2)
        for run in range(n):
            assert sum(run in p for p in pairs) >= 2

    def test_budget_at_or_above_all_pairs_returns_every_pair(self):
        assert sample_run_pairs(4, 6) == list(itertools.combinations(range(4), 2))
        assert sample_run_pairs(4, 100) == sample_run_pairs(4, 6)

    @pytest.mark.parametrize("n", [5, 6, 7, 9])
    def test_small_budget_covers_as_many_runs_as_possible(self, n):
        for budget in range(1, n):
            covered = {r for p in sample_run_pairs(n, budget) for r in p}
            assert len(covered) == min(n, 2 * budget)

    def test_deterministic_and_edge_cases(self):
        assert sample_run_pairs(9, seed=3) == sample_run_pairs(9, seed=3)
        assert sample_run_pairs(1) == [] and sample_run_pairs(0) == []
        with pytest.raises(ValueError):
            sample_run_pairs(5, 0)


def test_jaccard_and_word_ngrams():
    assert jaccard({1, 2, 3}, {2, 3, 4}) == 0.5
    assert jaccard([], []) == 1.0
    assert word_ngrams(["The quick brown fox"], 2) == {
        ("the", "quick"),
        ("quick", "brown"),
        ("brown", "fox"),
    }
    assert word_ngrams(["Hi"], 3) == {("hi",)}
    assert word_ngrams(["", "  "], 3) == set()


# --------------------------------------------------------------------------- #
# AnswerConsistency
# --------------------------------------------------------------------------- #


class TestAnswerConsistency:
    def test_init(self, mock_llm):
        metric = AnswerConsistency(llm=mock_llm)
        assert metric.name == "answer_consistency"
        assert metric.max_comparisons is None

    def test_rejects_legacy_llm(self):
        with pytest.raises(ValueError):
            AnswerConsistency(llm=MagicMock())

    @pytest.mark.asyncio
    async def test_identical_runs_score_one_without_llm_calls(self, mock_llm):
        metric = AnswerConsistency(llm=mock_llm)
        result = await metric.ascore(
            user_input="How long are backups kept?",
            responses=["30 days.", "30 days.", "30  Days."],
        )
        assert result.value == 1.0
        assert "agree" in result.reason
        mock_llm.agenerate.assert_not_called()

    @pytest.mark.asyncio
    async def test_weighted_score_and_reason(self, mock_llm):
        mock_llm.agenerate.return_value = agreement(format=1)
        metric = AnswerConsistency(llm=mock_llm, max_comparisons=1)
        result = await metric.ascore(
            user_input="How long are backups kept?",
            responses=["30 days.", "- 30 days\n- all sites"],
        )
        assert result.value == pytest.approx(0.9)  # format has weight 0.10
        assert "format 0.00" in result.reason
        assert "core_facts 1.00" in result.reason
        assert "judge reason" in result.reason

    @pytest.mark.asyncio
    async def test_core_fact_disagreement_costs_more_than_format(self, mock_llm):
        metric = AnswerConsistency(llm=mock_llm, max_comparisons=1)
        mock_llm.agenerate.return_value = agreement(core_facts=1)
        facts = await metric.ascore(responses=["a", "b"])
        mock_llm.agenerate.return_value = agreement(format=1)
        form = await metric.ascore(responses=["a", "b"])
        assert facts.value < form.value

    @pytest.mark.asyncio
    async def test_prompt_contains_input_and_both_responses(self, mock_llm):
        mock_llm.agenerate.return_value = agreement()
        metric = AnswerConsistency(llm=mock_llm, max_comparisons=1)
        await metric.ascore(user_input="QUESTION?", responses=["ALPHA", "BETA"])
        prompt, response_model = mock_llm.agenerate.call_args.args
        assert "QUESTION?" in prompt and "ALPHA" in prompt and "BETA" in prompt
        assert response_model is AnswerAgreementOutput

    @pytest.mark.asyncio
    async def test_default_budget_is_one_llm_call_per_run(self, mock_llm):
        mock_llm.agenerate.return_value = agreement()
        metric = AnswerConsistency(llm=mock_llm)
        await metric.ascore(responses=["a", "b", "c", "d", "e"])
        assert mock_llm.agenerate.call_count == 5  # not 10

    @pytest.mark.asyncio
    async def test_out_of_range_ratings_are_clamped(self, mock_llm):
        mock_llm.agenerate.return_value = agreement(core_facts=9, format=0)
        metric = AnswerConsistency(llm=mock_llm, max_comparisons=1)
        result = await metric.ascore(responses=["a", "b"])
        assert result.value == pytest.approx(0.9)
        assert 0.0 <= result.value <= 1.0

    @pytest.mark.asyncio
    async def test_needs_at_least_two_runs(self, mock_llm):
        metric = AnswerConsistency(llm=mock_llm)
        with pytest.raises(ValueError, match="at least two runs"):
            await metric.ascore(responses=["only one"])

    @pytest.mark.asyncio
    async def test_batch_score(self, mock_llm):
        mock_llm.agenerate.return_value = agreement(specifics=1)
        metric = AnswerConsistency(llm=mock_llm)
        results = await metric.abatch_score(
            [
                {"user_input": "q1", "responses": ["same", "same"]},
                {"user_input": "q2", "responses": ["30 days", "90 days"]},
            ]
        )
        assert results[0].value == 1.0
        assert results[1].value == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# RetrievalConsistency
# --------------------------------------------------------------------------- #


class TestRetrievalConsistency:
    def test_init(self):
        metric = RetrievalConsistency()
        assert metric.name == "retrieval_consistency"
        assert metric.ngram_size == 3
        with pytest.raises(ValueError):
            RetrievalConsistency(ngram_size=0)

    @pytest.mark.asyncio
    async def test_ids_jaccard_ignores_order(self):
        metric = RetrievalConsistency()
        result = await metric.ascore(
            retrieved_context_ids_runs=[
                ["doc-12", "doc-7", "doc-3"],
                ["doc-7", "doc-12", "doc-3"],
                ["doc-12", "doc-7", "doc-41"],
            ]
        )
        # pairs: (1, 2) identical = 1.0; each pair with run 3 shares 2 of 4 IDs = 0.5
        assert result.value == pytest.approx((1.0 + 0.5 + 0.5) / 3)
        assert "chunk IDs" in result.reason

    @pytest.mark.asyncio
    async def test_texts_survive_rechunking_where_ids_would_not(self):
        text = (
            "backup jobs run nightly and retention is thirty days "
            "for all gold tier policies"
        )
        words = text.split()
        rechunked = [" ".join(words[:7]), " ".join(words[6:])]
        metric = RetrievalConsistency(ngram_size=2)
        result = await metric.ascore(retrieved_contexts_runs=[[text], rechunked])
        assert result.value > 0.9

    @pytest.mark.asyncio
    async def test_disjoint_and_identical_runs(self):
        metric = RetrievalConsistency()
        same = await metric.ascore(retrieved_context_ids_runs=[[1, 2], [2, 1]])
        disjoint = await metric.ascore(retrieved_context_ids_runs=[[1], [2]])
        assert same.value == 1.0 and disjoint.value == 0.0

    @pytest.mark.asyncio
    async def test_input_validation(self):
        metric = RetrievalConsistency()
        with pytest.raises(ValueError, match="exactly one"):
            await metric.ascore()
        with pytest.raises(ValueError, match="exactly one"):
            await metric.ascore(
                retrieved_contexts_runs=[["a"], ["b"]],
                retrieved_context_ids_runs=[["a"], ["b"]],
            )
        with pytest.raises(ValueError, match="at least two runs"):
            await metric.ascore(retrieved_context_ids_runs=[["a"]])

    def test_sync_score(self):
        metric = RetrievalConsistency()
        result = metric.score(retrieved_context_ids_runs=[["a", "b"], ["a", "b"]])
        assert result.value == 1.0
