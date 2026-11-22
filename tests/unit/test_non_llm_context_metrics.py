"""Unit tests for the non-LLM context precision and recall metrics.

Both metrics convert a per-context string-similarity score into a binary
relevance judgement using a shared ``threshold`` default and then aggregate it.
The two sibling metrics must apply that threshold consistently: a context whose
similarity is exactly at the threshold should count as relevant for both
``NonLLMContextRecall`` and ``NonLLMContextPrecisionWithReference``.
"""

import numpy as np
import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._context_precision import NonLLMContextPrecisionWithReference
from ragas.metrics._context_recall import NonLLMContextRecall


class _ConstantSimilarity:
    """Stand-in distance measure that returns a fixed similarity score.

    Lets us drive the metrics' threshold logic deterministically without
    depending on the exact value a real string-similarity backend produces.
    """

    def __init__(self, value: float):
        self.value = value

    async def single_turn_ascore(self, sample, callbacks=None) -> float:
        return self.value


def test_recall_counts_similarity_exactly_at_threshold():
    """A reference context matched exactly at the threshold must count as recalled."""
    metric = NonLLMContextRecall()  # default threshold = 0.5
    assert metric._compute_score([0.5]) == 1.0
    assert metric._compute_score([0.49]) == 0.0
    assert metric._compute_score([0.5, 0.6, 0.4]) == pytest.approx(2 / 3)


def test_recall_empty_verdicts_is_nan():
    """No reference contexts -> undefined recall (nan), not a crash."""
    assert np.isnan(NonLLMContextRecall()._compute_score([]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "similarity, expected",
    [
        (0.4, 0.0),  # below threshold -> not relevant
        (0.5, 1.0),  # exactly at threshold -> relevant
        (0.6, 1.0),  # above threshold -> relevant
    ],
)
async def test_precision_and_recall_agree_at_threshold(similarity, expected):
    """Precision and recall must make the same relevance call for the same similarity.

    With a single retrieved/reference context, both metrics reduce to the binary
    relevance decision, so their scores must be identical. This regresses the
    inconsistency where recall used ``>`` while precision used ``>=``.
    """
    sample = SingleTurnSample(
        retrieved_contexts=["retrieved"],
        reference_contexts=["reference"],
    )

    recall = NonLLMContextRecall()
    recall._distance_measure = _ConstantSimilarity(similarity)
    precision = NonLLMContextPrecisionWithReference(
        distance_measure=_ConstantSimilarity(similarity)
    )

    recall_score = await recall._single_turn_ascore(sample, callbacks=None)
    precision_score = await precision._single_turn_ascore(sample, callbacks=None)

    # NonLLMContextPrecision aggregates via average precision, which adds a 1e-10
    # smoothing term to the denominator, so a perfect score is ~0.9999999999 rather
    # than exactly 1.0 -- compare with approx.
    assert recall_score == pytest.approx(expected)
    assert precision_score == pytest.approx(expected)
    assert recall_score == pytest.approx(precision_score)


@pytest.mark.asyncio
async def test_non_llm_context_metrics_real_similarity():
    """End-to-end sanity check with the real string-similarity backend."""
    identical = SingleTurnSample(
        retrieved_contexts=["abc"],
        reference_contexts=["abc"],
    )
    disjoint = SingleTurnSample(
        retrieved_contexts=["abc"],
        reference_contexts=["xyz"],
    )

    assert await NonLLMContextRecall()._single_turn_ascore(
        identical, None
    ) == pytest.approx(1.0)
    assert await NonLLMContextPrecisionWithReference()._single_turn_ascore(
        identical, None
    ) == pytest.approx(1.0)
    assert await NonLLMContextRecall()._single_turn_ascore(
        disjoint, None
    ) == pytest.approx(0.0)
    assert await NonLLMContextPrecisionWithReference()._single_turn_ascore(
        disjoint, None
    ) == pytest.approx(0.0)
