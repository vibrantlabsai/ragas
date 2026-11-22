"""Threshold boundary behavior for the non-LLM context metrics (#2777)."""

from ragas.metrics import NonLLMContextRecall


def test_non_llm_context_recall_score_at_threshold_counts_as_relevant():
    """A similarity score exactly equal to the threshold is relevant, consistent
    with NonLLMContextPrecisionWithReference's `>=` comparison."""
    metric = NonLLMContextRecall()
    assert metric.threshold == 0.5
    assert metric._compute_score([0.5]) == 1.0


def test_non_llm_context_recall_scores_below_threshold_not_relevant():
    metric = NonLLMContextRecall()
    assert metric._compute_score([0.49, 0.51]) == 0.5
    assert metric._compute_score([0.1, 0.2]) == 0.0
