"""Unit tests for `ResponseRelevancy` score post-processing.

Regression tests for #2088: cosine similarity is mathematically in [-1, 1],
so near-orthogonal embeddings can drive the metric below zero. The metric is
documented to live in [0, 1], so we clip on the way out.
"""

from __future__ import annotations

import numpy as np
import pytest

from ragas.metrics._answer_relevance import (
    ResponseRelevanceOutput,
    ResponseRelevancy,
)


def _make_metric(similarity_values: list[float]) -> ResponseRelevancy:
    """Build a metric whose `calculate_similarity` returns a fixed array."""

    metric = ResponseRelevancy()
    metric.calculate_similarity = lambda question, generated_questions: np.asarray(
        similarity_values, dtype=float
    )
    return metric


@pytest.mark.parametrize(
    ("similarities", "expected"),
    [
        ([-0.001, -0.002, -0.003], 0.0),
        ([-0.5, 0.5], 0.0),
        ([0.3, 0.5, 0.7], 0.5),
        ([0.9, 0.9, 0.9], 0.9),
    ],
)
def test_score_is_clipped_to_unit_interval(similarities, expected):
    metric = _make_metric(similarities)
    answers = [
        ResponseRelevanceOutput(question=f"q{i}", noncommittal=0)
        for i in range(len(similarities))
    ]
    row = {"user_input": "original"}

    score = metric._calculate_score(answers, row)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(expected)


def test_noncommittal_answers_force_zero_regardless_of_similarity():
    metric = _make_metric([0.9, 0.9, 0.9])
    answers = [
        ResponseRelevanceOutput(question=f"q{i}", noncommittal=1) for i in range(3)
    ]
    row = {"user_input": "original"}

    assert metric._calculate_score(answers, row) == 0.0


def test_empty_generated_questions_return_nan():
    metric = ResponseRelevancy()
    answers = [ResponseRelevanceOutput(question="", noncommittal=0) for _ in range(3)]
    row = {"user_input": "original"}

    score = metric._calculate_score(answers, row)
    assert np.isnan(score)
