import asyncio
import math
import warnings

import pytest

from ragas import aevaluate, evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics.base import MetricType, SingleTurnMetric


class FailingRowMetric(SingleTurnMetric):
    def init(self, run_config):
        pass

    async def _single_turn_ascore(self, sample, callbacks):
        if sample.response == "fail":
            raise RuntimeError("row failed")
        return 1.0


@pytest.mark.parametrize("entrypoint", ["sync", "sync_no_nest", "async"])
@pytest.mark.parametrize("warn_on_missing_scores", [None, True, False])
def test_evaluation_coverage_after_executor_failure(
    entrypoint, warn_on_missing_scores, monkeypatch, caplog
):
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    dataset = EvaluationDataset(
        samples=[SingleTurnSample(response=value) for value in ["ok", "fail"]]
    )
    metric = FailingRowMetric(
        name="failing_row",
        _required_columns={MetricType.SINGLE_TURN: {"response"}},
    )
    kwargs = dict(
        dataset=dataset,
        metrics=[metric],
        raise_exceptions=False,
        show_progress=False,
    )
    if warn_on_missing_scores is not None:
        kwargs["warn_on_missing_scores"] = warn_on_missing_scores

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if entrypoint == "async":
            result = asyncio.run(aevaluate(**kwargs))
        else:
            result = evaluate(**kwargs, allow_nest_asyncio=entrypoint == "sync")

    coverage_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    coverage_logs = [
        record for record in caplog.records if record.name == "ragas.dataset_schema"
    ]
    expected_warnings = int(warn_on_missing_scores is not False)
    assert len(coverage_warnings) == expected_warnings
    assert len(coverage_logs) == expected_warnings
    if expected_warnings:
        assert "failing_row: 1/2" in str(coverage_warnings[0].message)

    assert result["failing_row"][0] == 1.0
    assert math.isnan(result["failing_row"][1])
    assert result.nan_counts == {"failing_row": 1}
    assert result.total_counts == {"failing_row": 2}
    assert result.coverage == {"failing_row": 0.5}
    assert result.summary() == {
        "failing_row": {"mean": 1.0, "n_scored": 1, "n_total": 2, "coverage": 0.5}
    }
    assert repr(result) == "{'failing_row': 1.0000 (1/2)}"
