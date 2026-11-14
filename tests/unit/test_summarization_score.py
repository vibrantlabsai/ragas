"""Unit tests for ``SummarizationScore``."""

import math

import numpy as np

from ragas.metrics import SummarizationScore


def test_compute_qa_score_returns_nan_for_empty_answers():
    """Regression: empty ``answers`` list must not raise ZeroDivisionError.

    When upstream keyphrase extraction or question generation returns ``[]``
    (the LLM refused, was rate-limited, or the input had no extractable
    content), ``_get_answers`` is called with no questions and returns an empty
    list of answers. ``_compute_qa_score`` must not divide by zero in that case;
    it should signal "no score" by returning NaN, matching the convention used
    by other ragas metrics (e.g. context_recall).
    """
    metric = SummarizationScore()
    assert math.isnan(metric._compute_qa_score([]))


def test_compute_qa_score_normal_path():
    metric = SummarizationScore()
    # 3 of 4 answers correct
    assert metric._compute_qa_score(["1", "0", "1", "1"]) == 0.75


def test_compute_score_propagates_nan_when_qa_fails():
    """If qa_score is NaN, the overall score should also be NaN."""
    metric = SummarizationScore()
    nan_qa = metric._compute_qa_score([])
    overall = metric._compute_score({"qa_score": nan_qa, "conciseness_score": 1.0})
    assert np.isnan(overall)
