from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel

from ragas.metrics import MetricResult
from ragas.metrics.base import SimpleLLMMetric

if TYPE_CHECKING:
    from ragas.dataset import Dataset
    from ragas.llms import BaseRagasLLM


class MetricResponseModel(BaseModel):
    value: int


@dataclass
class CustomMetric(SimpleLLMMetric):
    def __post_init__(self):
        super().__post_init__()
        self._response_model = MetricResponseModel

    def get_correlation(self, gold_labels: list[str], predictions: list[str]) -> float:
        return 0.0


def test_validate_alignment_filters_missing_scores_as_pairs():
    """Missing labels must not shift predictions onto a different example."""

    class FakeDataset:
        rows = [
            {"input": "first", "test_metric": "pass"},
            {"input": "second", "test_metric": None},
            {"input": "third", "test_metric": "fail"},
            {"input": "fourth", "test_metric": "pass"},
        ]

        def reload(self):
            pass

        def __iter__(self):
            return iter(self.rows)

        def get_row_value(self, row, key):
            return row.get(key)

        def to_pandas(self):
            return {}

    metric = CustomMetric(name="test_metric", prompt="Evaluate {input}")
    predictions = iter(["pass", "fail", None, "fail"])
    metric.score = lambda **kwargs: MetricResult(  # type: ignore[method-assign]
        value=next(predictions)
    )

    compared_scores = None

    def get_correlation(gold_scores, pred_scores):
        nonlocal compared_scores
        compared_scores = (gold_scores, pred_scores)
        return 0.25

    metric.get_correlation = get_correlation  # type: ignore[method-assign]

    result = metric.validate_alignment(
        llm=cast("BaseRagasLLM", object()),
        test_dataset=cast("Dataset", FakeDataset()),
    )

    assert compared_scores == (["pass", "pass"], ["pass", "fail"])
    assert result["correlation"] == 0.25
    assert result["agreement_rate"] == 0.5
    assert result["df"]["test_metric_pred"] == ["pass", "fail", None, "fail"]
