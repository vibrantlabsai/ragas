import pytest
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._ir_ranking import (
    ContextHitRate,
    ContextMAP,
    ContextMRR,
    ContextNDCG,
)


@pytest.mark.asyncio
async def test_context_ndcg_perfect_ranking():
    metric = ContextNDCG(k=3)
    sample = SingleTurnSample(
        user_input="What is vector search?",
        retrieved_contexts=[
            "Vector search uses approximate nearest neighbors.",
            "BM25 is a keyword search algorithm.",
            "Relational databases use B-Trees.",
        ],
        reference_contexts=[
            "Vector search uses approximate nearest neighbors.",
        ],
    )
    score = await metric._single_turn_ascore(sample, callbacks=None)
    # When the only relevant item is at rank 1, NDCG@K is 1.0
    assert pytest.approx(score, 0.001) == 1.0


@pytest.mark.asyncio
async def test_context_ndcg_suboptimal_ranking():
    metric = ContextNDCG(k=3)
    sample = SingleTurnSample(
        user_input="What is vector search?",
        retrieved_contexts=[
            "Relational databases use B-Trees.",
            "Vector search uses approximate nearest neighbors.",
            "BM25 is a keyword search algorithm.",
        ],
        reference_contexts=[
            "Vector search uses approximate nearest neighbors.",
        ],
    )
    score = await metric._single_turn_ascore(sample, callbacks=None)
    # Rank 2 vs Ideal Rank 1: (1 / log2(3)) / (1 / log2(2)) = log2(2) / log2(3) = 1 / 1.58496 ~ 0.6309
    assert 0.62 < score < 0.64


@pytest.mark.asyncio
async def test_context_mrr_first_rank():
    metric = ContextMRR(k=5)
    sample = SingleTurnSample(
        retrieved_contexts=["First relevant passage", "Irrelevant", "Another irrelevant"],
        reference_contexts=["First relevant passage"],
    )
    score = await metric._single_turn_ascore(sample, callbacks=None)
    assert score == 1.0


@pytest.mark.asyncio
async def test_context_mrr_second_rank():
    metric = ContextMRR(k=5)
    sample = SingleTurnSample(
        retrieved_contexts=["Irrelevant context", "First relevant passage", "Another irrelevant"],
        reference_contexts=["First relevant passage"],
    )
    score = await metric._single_turn_ascore(sample, callbacks=None)
    assert score == 0.5


@pytest.mark.asyncio
async def test_context_mrr_no_match():
    metric = ContextMRR(k=5)
    sample = SingleTurnSample(
        retrieved_contexts=["Irrelevant 1", "Irrelevant 2"],
        reference_contexts=["Ground truth reference"],
    )
    score = await metric._single_turn_ascore(sample, callbacks=None)
    assert score == 0.0


@pytest.mark.asyncio
async def test_context_map():
    metric = ContextMAP(k=3)
    sample = SingleTurnSample(
        retrieved_contexts=["Relevant Doc 1", "Irrelevant Doc", "Relevant Doc 2"],
        reference_contexts=["Relevant Doc 1", "Relevant Doc 2"],
    )
    # Precision at rank 1 = 1/1 = 1.0
    # Precision at rank 3 = 2/3 ~ 0.6667
    # MAP = (1.0 + 0.6667) / 2 = 1.6667 / 2 ~ 0.8333
    score = await metric._single_turn_ascore(sample, callbacks=None)
    assert pytest.approx(score, 0.01) == 0.8333


@pytest.mark.asyncio
async def test_context_hit_rate():
    metric = ContextHitRate(k=2)
    sample_hit = SingleTurnSample(
        retrieved_contexts=["Irrelevant", "Relevant Context", "Extra Doc"],
        reference_contexts=["Relevant Context"],
    )
    score_hit = await metric._single_turn_ascore(sample_hit, callbacks=None)
    assert score_hit == 1.0

    sample_miss = SingleTurnSample(
        retrieved_contexts=["Irrelevant 1", "Irrelevant 2", "Relevant Context"],
        reference_contexts=["Relevant Context"],
    )
    # Cutoff k=2 excludes rank 3
    score_miss = await metric._single_turn_ascore(sample_miss, callbacks=None)
    assert score_miss == 0.0


@pytest.mark.asyncio
async def test_empty_contexts():
    metric = ContextNDCG(k=5)
    sample_empty_retrieved = SingleTurnSample(
        retrieved_contexts=[],
        reference_contexts=["Reference"],
    )
    assert await metric._single_turn_ascore(sample_empty_retrieved, callbacks=None) == 0.0

    sample_empty_reference = SingleTurnSample(
        retrieved_contexts=["Retrieved"],
        reference_contexts=[],
    )
    assert await metric._single_turn_ascore(sample_empty_reference, callbacks=None) == 0.0
