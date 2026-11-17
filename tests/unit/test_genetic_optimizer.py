from types import SimpleNamespace

import pandas as pd

from ragas.losses import BinaryMetricLoss, MSELoss
from ragas.metrics.base import MetricOutputType
from ragas.optimizers import GeneticOptimizer
from ragas.optimizers.genetic import MIN_ANNOTATIONS


class _FakeResult:
    def __init__(self, values):
        self.values = values

    def to_pandas(self):
        return pd.DataFrame({"fake_metric": self.values})


def _optimizer(monkeypatch, output_type, y_true, predictions):
    """GeneticOptimizer whose LLM stages are stubbed out.

    Only the fitness evaluation and the final pick of the best candidate run.
    ``predictions`` maps each candidate prompt to the scores the metric
    returns when it uses that prompt.
    """
    metric = SimpleNamespace(
        name="fake_metric", output_type=output_type, get_prompts=lambda: {}
    )
    optimizer = GeneticOptimizer(metric=metric, llm=object())
    candidates = [{"instruction": name} for name in predictions]

    monkeypatch.setattr(optimizer, "initialize_population", lambda **kw: [])
    monkeypatch.setattr(
        optimizer, "feedback_mutation", lambda population, *a, **kw: candidates
    )
    monkeypatch.setattr(optimizer, "cross_over_mutation", lambda **kw: kw["candidates"])
    monkeypatch.setattr(
        optimizer, "_get_evaluation_dataset", lambda dataset: (None, y_true)
    )
    monkeypatch.setattr(
        optimizer,
        "evaluate_candidate",
        lambda **kw: _FakeResult(predictions[kw["candidate"]["instruction"]]),
    )
    return optimizer


def test_genetic_optimizer_picks_lowest_mse(monkeypatch):
    """MSELoss is an error, so the best prompt has the lowest loss."""
    y_true = [1.0, 0.0, 1.0, 0.0]
    predictions = {
        "good": [1.0, 0.0, 1.0, 0.0],  # MSE 0.0
        "ok": [1.0, 0.0, 0.5, 0.5],  # MSE 0.125
        "bad": [0.0, 1.0, 0.0, 1.0],  # MSE 1.0
    }
    optimizer = _optimizer(
        monkeypatch, MetricOutputType.CONTINUOUS, y_true, predictions
    )

    best = optimizer.optimize(list(range(MIN_ANNOTATIONS)), MSELoss(), {})

    assert best == {"instruction": "good"}


def test_genetic_optimizer_picks_highest_accuracy(monkeypatch):
    """BinaryMetricLoss returns accuracy, so the best prompt has the highest value."""
    y_true = [1, 0, 1, 0]
    predictions = {
        "bad": [0, 1, 0, 1],  # accuracy 0.0
        "good": [1, 0, 1, 0],  # accuracy 1.0
        "ok": [1, 0, 0, 0],  # accuracy 0.75
    }
    optimizer = _optimizer(monkeypatch, MetricOutputType.BINARY, y_true, predictions)

    best = optimizer.optimize(list(range(MIN_ANNOTATIONS)), BinaryMetricLoss(), {})

    assert best == {"instruction": "good"}
