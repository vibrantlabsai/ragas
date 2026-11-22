"""Unit tests for NonLLM context metric threshold consistency (issue #2777).

`NonLLMContextRecall` and `NonLLMContextPrecisionWithReference` share the same
default ``threshold`` (0.5) and the same ``NonLLMStringSimilarity`` distance
measure, so a per-context similarity score that lands exactly on the threshold
must be classified as relevant by both metrics. Previously recall used a strict
``>`` while precision used ``>=``, so a boundary score was relevant for
precision but not for recall.
"""

import sys

import numpy as np

from ragas.metrics._context_precision import NonLLMContextPrecisionWithReference
from ragas.metrics._context_recall import NonLLMContextRecall
from ragas.metrics.utils import meets_threshold


class TestNonLLMThresholdConsistency:
    def test_recall_counts_score_at_threshold_as_relevant(self):
        """A similarity exactly at the threshold is relevant (>=), not dropped."""
        recall = NonLLMContextRecall(threshold=0.5)
        # Single context whose similarity equals the threshold.
        score = recall._compute_score([0.5])
        assert score == 1.0

    def test_recall_below_threshold_is_irrelevant(self):
        recall = NonLLMContextRecall(threshold=0.5)
        assert recall._compute_score([0.49]) == 0.0

    def test_recall_and_precision_agree_at_threshold(self):
        """Both metrics must classify a boundary score the same way."""
        threshold = 0.5
        recall = NonLLMContextRecall(threshold=threshold)
        precision = NonLLMContextPrecisionWithReference(threshold=threshold)

        boundary = [threshold]
        # recall: numerator/denom -> 1/1 == 1.0 when the boundary counts.
        recall_relevant = recall._compute_score(boundary) > 0
        # precision: average precision over a single relevant item == 1.0.
        precision_relevant = (
            precision._calculate_average_precision(
                [1 if s >= threshold else 0 for s in boundary]
            )
            > 0
        )

        assert recall_relevant == precision_relevant == True  # noqa: E712

    def test_recall_empty_returns_nan(self):
        recall = NonLLMContextRecall(threshold=0.5)
        assert np.isnan(recall._compute_score([]))


class TestSharedThresholdHelper:
    """The two metrics agree because they share one rule, not by coincidence."""

    def test_helper_is_inclusive_at_the_boundary(self):
        assert meets_threshold(0.5, 0.5) is True
        assert meets_threshold(0.51, 0.5) is True
        assert meets_threshold(0.49, 0.5) is False

    def test_both_metrics_bind_the_same_rule(self):
        """Both modules must resolve to the one definition, not a private copy.

        Equal outputs only show the two agree today. Identity shows they
        cannot diverge: a future edit to either metric's boundary has to go
        through this function.
        """
        # Fetched from sys.modules: `ragas.metrics._context_recall` resolves
        # to the re-exported class, not the module that defines it.
        recall_module = sys.modules["ragas.metrics._context_recall"]
        precision_module = sys.modules["ragas.metrics._context_precision"]

        assert recall_module.meets_threshold is meets_threshold
        assert precision_module.meets_threshold is meets_threshold

    def test_recall_follows_the_shared_rule(self, monkeypatch):
        """Recall reads the rule at call time rather than inlining it."""
        recall_module = sys.modules["ragas.metrics._context_recall"]

        monkeypatch.setattr(
            recall_module, "meets_threshold", lambda score, threshold: False
        )
        # Nothing clears an always-false rule, so the boundary score that
        # normally counts must now be dropped.
        assert NonLLMContextRecall(threshold=0.5)._compute_score([0.5]) == 0.0

    def test_no_metric_reimplements_the_boundary(self):
        """Neither NonLLM metric may compare against self.threshold directly."""
        from pathlib import Path

        for name in (
            "ragas.metrics._context_recall",
            "ragas.metrics._context_precision",
        ):
            module = sys.modules[name]
            source = Path(module.__file__).read_text(encoding="utf-8")
            assert ">= self.threshold" not in source
            assert "> self.threshold" not in source
