import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._context_precision import NonLLMContextPrecisionWithReference
from ragas.metrics._context_recall import NonLLMContextRecall


@pytest.mark.asyncio
async def test_non_llm_context_recall_with_no_retrieved_contexts():
    """An empty retrieval scores 0.0 instead of raising ValueError from max()."""
    metric = NonLLMContextRecall()
    sample = SingleTurnSample(
        user_input="q",
        retrieved_contexts=[],
        reference_contexts=["a reference context"],
    )

    assert await metric.single_turn_ascore(sample) == 0.0


@pytest.mark.asyncio
async def test_non_llm_context_precision_with_no_reference_contexts():
    """No reference contexts scores 0.0 instead of raising ValueError from max()."""
    metric = NonLLMContextPrecisionWithReference()
    sample = SingleTurnSample(
        user_input="q",
        retrieved_contexts=["a retrieved context"],
        reference_contexts=[],
    )

    assert await metric.single_turn_ascore(sample) == 0.0


@pytest.mark.asyncio
async def test_non_llm_context_metrics_with_matching_contexts():
    """Non-empty inputs keep their previous behaviour."""
    recall = NonLLMContextRecall()
    precision = NonLLMContextPrecisionWithReference()
    sample = SingleTurnSample(
        user_input="q",
        retrieved_contexts=["the same context"],
        reference_contexts=["the same context"],
    )

    assert await recall.single_turn_ascore(sample) == 1.0
    assert await precision.single_turn_ascore(sample) > 0.99
