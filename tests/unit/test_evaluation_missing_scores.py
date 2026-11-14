import typing as t

import numpy as np
import pytest

from ragas import aevaluate, evaluate
from ragas.callbacks import ChainRun
from ragas.dataset_schema import EvaluationDataset, EvaluationResult, SingleTurnSample
from ragas.metrics.base import SingleTurnMetric


class ControlledMetric(SingleTurnMetric):
    """Exercise real evaluation and executor paths without calling an LLM."""

    def init(self, run_config):
        pass

    async def _single_turn_ascore(self, sample, callbacks):
        if sample.response == "error":
            raise ValueError("Scoring failed")
        if sample.response == "undefined":
            return np.nan
        return float(sample.response)


def make_dataset(responses):
    return EvaluationDataset(
        samples=[SingleTurnSample(response=response) for response in responses]
    )


def make_result(scores, **kwargs):
    # Supply a valid root trace, as an evaluation run would.
    root = ChainRun(
        run_id="root", parent_run_id=None, name="evaluation", inputs={}, metadata={}
    )
    return EvaluationResult(
        scores=scores,
        dataset=make_dataset(["1"] * len(scores)),
        ragas_traces={"root": root},
        **kwargs,
    )


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([1.0, np.nan], "0.5000"),
        ([np.nan, np.nan], "0.0000"),
        ([1.0, 0.0], "0.5000"),
        ([0.0, np.nan], "0.0000"),
        ([0.2, np.nan, 0.4], "0.2000"),
        ([float("inf"), np.nan], "inf"),
    ],
)
def test_zero_policy_uses_all_rows_without_changing_raw_scores(scores, expected):
    result = make_result(
        [{"metric": score} for score in scores], missing_score_policy="zero"
    )

    assert repr(result) == f"{{'metric': {expected}}}"
    np.testing.assert_equal(result["metric"], scores)
    np.testing.assert_equal([row["metric"] for row in result.scores], scores)
    np.testing.assert_equal(result.to_pandas()["metric"].to_numpy(), scores)


@pytest.mark.parametrize("policy_kwargs", [{}, {"missing_score_policy": "exclude"}])
@pytest.mark.parametrize(
    ("scores", "expected"),
    [([1.0, np.nan], "1.0000"), ([np.nan, np.nan], "nan"), ([1.0, 0.0], "0.5000")],
)
def test_default_and_explicit_exclusion_preserve_existing_means(
    policy_kwargs, scores, expected
):
    result = make_result([{"metric": score} for score in scores], **policy_kwargs)

    assert repr(result) == f"{{'metric': {expected}}}"


def test_zero_policy_aggregates_each_metric_independently():
    result = make_result(
        [
            {"continuous": 1.0, "binary": np.nan},
            {"continuous": np.nan, "binary": 0.0},
            {"continuous": np.nan, "binary": 1.0},
        ],
        binary_columns=["binary"],
        missing_score_policy="zero",
    )

    assert repr(result) == "{'continuous': 0.3333, 'binary': 0.3333}"


def test_result_rejects_invalid_missing_score_policy():
    with pytest.raises(ValueError, match="missing_score_policy"):
        make_result([{"metric": 1.0}], missing_score_policy="invalid")


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_response", ["error", "undefined"])
@pytest.mark.parametrize("policy", ["exclude", "zero"])
async def test_aevaluate_missing_rows_do_not_inflate_zero_policy(
    missing_response, policy
):
    baseline = await aevaluate(
        make_dataset(["1", "0"]),
        metrics=[ControlledMetric(name="metric")],
        show_progress=False,
        missing_score_policy=policy,
    )
    degraded = await aevaluate(
        make_dataset(["1", missing_response]),
        metrics=[ControlledMetric(name="metric")],
        show_progress=False,
        missing_score_policy=policy,
    )

    assert isinstance(baseline, EvaluationResult)
    assert isinstance(degraded, EvaluationResult)
    assert repr(baseline) == "{'metric': 0.5000}"
    assert repr(degraded) == (
        "{'metric': 0.5000}" if policy == "zero" else "{'metric': 1.0000}"
    )
    np.testing.assert_equal(degraded["metric"], [1.0, np.nan])


@pytest.mark.parametrize("allow_nest_asyncio", [False, True])
def test_evaluate_forwards_zero_policy(allow_nest_asyncio):
    result = evaluate(
        make_dataset(["1", "error"]),
        metrics=[ControlledMetric(name="metric")],
        show_progress=False,
        allow_nest_asyncio=allow_nest_asyncio,
        missing_score_policy="zero",
    )

    assert isinstance(result, EvaluationResult)
    assert repr(result) == "{'metric': 0.5000}"
    np.testing.assert_equal(result["metric"], [1.0, np.nan])


@pytest.mark.asyncio
async def test_default_evaluation_still_excludes_missing_scores():
    result = await aevaluate(
        make_dataset(["1", "error"]),
        metrics=[ControlledMetric(name="metric")],
        show_progress=False,
    )

    assert repr(result) == "{'metric': 1.0000}"


@pytest.mark.asyncio
async def test_zero_policy_does_not_suppress_exceptions():
    with pytest.raises(ValueError, match="Scoring failed"):
        await aevaluate(
            make_dataset(["1", "error"]),
            metrics=[ControlledMetric(name="metric")],
            show_progress=False,
            missing_score_policy="zero",
            raise_exceptions=True,
        )


@pytest.mark.asyncio
async def test_invalid_policy_is_rejected_before_metric_initialization(monkeypatch):
    metric = ControlledMetric(name="metric")

    def unexpected_init(run_config):
        pytest.fail("Invalid policies should fail before metric initialization")

    monkeypatch.setattr(metric, "init", unexpected_init)
    with pytest.raises(ValueError, match="missing_score_policy"):
        await aevaluate(
            make_dataset(["1"]),
            metrics=[metric],
            show_progress=False,
            missing_score_policy=t.cast(t.Any, "invalid"),
        )


@pytest.mark.asyncio
async def test_return_executor_preserves_raw_missing_scores():
    from ragas.executor import Executor

    executor = await aevaluate(
        make_dataset(["1", "error"]),
        metrics=[ControlledMetric(name="metric")],
        show_progress=False,
        missing_score_policy="zero",
        return_executor=True,
    )

    assert isinstance(executor, Executor)
    np.testing.assert_equal(await executor.aresults(), [1.0, np.nan])
