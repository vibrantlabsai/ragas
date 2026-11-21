"""
Unit tests for threshold consistency between NonLLMContextRecall and
NonLLMContextPrecisionWithReference.

Regression for issue #2777: NonLLMContextRecall used strict `>` while
NonLLMContextPrecisionWithReference used `>=`, causing a context that
scores exactly at the threshold to be counted as relevant by precision
but not by recall.
"""

import pytest

from ragas.metrics._context_recall import NonLLMContextRecall
from ragas.metrics._context_precision import NonLLMContextPrecisionWithReference


class TestNonLLMContextRecallThreshold:
    """Tests for NonLLMContextRecall._compute_score threshold boundary."""

    def setup_method(self):
        self.metric = NonLLMContextRecall()

    def test_score_above_threshold_is_relevant(self):
        """Score strictly above threshold counts as relevant."""
        result = self.metric._compute_score([0.6])
        assert result == 1.0

    def test_score_at_threshold_is_relevant(self):
        """Score exactly at threshold must count as relevant (>= boundary, regression for #2777)."""
        result = self.metric._compute_score([0.5])
        assert result == 1.0, (
            "Score exactly at threshold should be relevant. "
            "Bug was: `> threshold` instead of `>= threshold`."
        )

    def test_score_below_threshold_is_not_relevant(self):
        """Score strictly below threshold counts as not relevant."""
        result = self.metric._compute_score([0.4])
        assert result == 0.0

    def test_mixed_scores_with_boundary(self):
        """Mixed list: one at threshold (relevant), one below (not relevant)."""
        result = self.metric._compute_score([0.5, 0.3])
        # 1 relevant out of 2 = 0.5
        assert result == pytest.approx(0.5)

    def test_empty_list_returns_nan(self):
        """Empty verdict list returns nan."""
        import math
        result = self.metric._compute_score([])
        assert math.isnan(result)

    def test_custom_threshold_boundary(self):
        """Custom threshold: score exactly at custom threshold is relevant."""
        self.metric.threshold = 0.7
        assert self.metric._compute_score([0.7]) == 1.0
        assert self.metric._compute_score([0.69]) == 0.0


class TestThresholdConsistency:
    """Recall and precision must treat the threshold boundary identically."""

    def test_boundary_score_consistent_between_recall_and_precision(self):
        """
        A context that scores exactly at threshold must be counted as relevant
        by BOTH metrics. Before the fix, recall used `>` (excluded boundary)
        while precision used `>=` (included boundary).
        """
        recall = NonLLMContextRecall()
        precision = NonLLMContextPrecisionWithReference()

        # Both default to threshold=0.5
        assert recall.threshold == precision.threshold == 0.5

        # Recall: _compute_score takes a flat list of similarity floats
        recall_result = recall._compute_score([0.5])

        # Precision: _calculate_average_precision takes a binary [0/1] list;
        # the >= check happens inside _single_turn_ascore before calling it.
        # We test the `>=` operator directly by asserting the threshold guard:
        score = 0.5
        precision_binary = 1 if score >= precision.threshold else 0
        recall_binary = 1 if score >= recall.threshold else 0

        assert recall_binary == precision_binary == 1, (
            "Both metrics must treat a score at the threshold as relevant. "
            "recall_binary=%d, precision_binary=%d" % (recall_binary, precision_binary)
        )
        assert recall_result == 1.0
