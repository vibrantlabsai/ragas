"""R7: goals, floors (user-set only), scorer and the DSPy metric bridge."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ragas.optimizers.prompt_optimization import metrics as metrics_module
from ragas.optimizers.prompt_optimization.config import OptimizationGoal
from ragas.optimizers.prompt_optimization.metrics import (
    RagasFaithfulnessMetric,
    Scorer,
    aggregate,
    build_dspy_metric,
    check_floors,
    default_metric_registry,
    run_in_thread_loop,
    run_without_loop,
)


def test_goal_metrics_are_target_plus_floored_only():
    goal = OptimizationGoal(
        "token_f1", 0.8, floors={"token_recall": "baseline", "token_f1": 0.1}
    )
    assert goal.metrics == ["token_f1", "token_recall"]


def test_goal_validation(tmp_path):
    with pytest.raises(ValueError):
        OptimizationGoal(target_value=1.5)
    with pytest.raises(ValueError):
        OptimizationGoal(floors={"token_recall": "high"})  # type: ignore[dict-item]
    path = tmp_path / "goal.json"
    path.write_text(
        '{"target_metric": "token_f0_5", "floors": {"exact_match": 0.2}}',
        encoding="utf-8",
    )
    goal = OptimizationGoal.from_json(path)
    assert goal.target_metric == "token_f0_5" and goal.floors == {"exact_match": 0.2}


def test_resolve_floors_absolute_and_baseline():
    goal = OptimizationGoal(
        floors={"token_recall": "baseline", "exact_match": 0.5}, floor_tolerance=0.02
    )
    resolved = goal.resolve_floors({"token_recall": 0.8, "exact_match": 0.1})
    assert resolved == {"token_recall": pytest.approx(0.78), "exact_match": 0.5}


def test_resolve_baseline_floor_needs_baseline():
    goal = OptimizationGoal(floors={"token_recall": "baseline"})
    with pytest.raises(ValueError, match="baseline"):
        goal.resolve_floors(None)


def test_no_floors_means_no_constraint():
    assert OptimizationGoal().resolve_floors({"token_recall": 0.0}) == {}
    assert check_floors({"token_f1": 0.0}, {}).passed


def test_check_floors_reports_each_metric():
    check = check_floors(
        {"token_recall": 0.7, "exact_match": 0.5},
        {"token_recall": 0.75, "exact_match": 0.5},
    )
    assert not check.passed
    assert check.failed_metrics == ["token_recall"]
    with pytest.raises(KeyError):
        check_floors({}, {"token_recall": 0.5})


def test_scorer_and_aggregate(golden_rows):
    scorer = Scorer(["token_f1", "exact_match", "token_f1"])
    assert scorer.metric_names == ["token_f1", "exact_match"]
    row = golden_rows[0]
    assert scorer.score(row, row.reference) == {"token_f1": 1.0, "exact_match": 1.0}
    assert aggregate([{"m": 1.0}, {"m": 0.0}]) == {"m": 0.5}
    assert aggregate([]) == {}


def test_scorer_rejects_unknown_metric():
    with pytest.raises(KeyError, match="faithfulness"):
        Scorer(["faithfulness"])


def test_registry_names():
    assert set(default_metric_registry()) == {
        "token_precision",
        "token_recall",
        "token_f1",
        "token_f0_5",
        "exact_match",
    }


def _async_metric(value):
    async def ascore(**kwargs):
        await asyncio.sleep(0)  # really suspends, like an LLM call
        return SimpleNamespace(value=value)

    return SimpleNamespace(ascore=ascore)


def test_faithfulness_adapter_treats_nan_as_faithful(golden_rows):
    assert (
        RagasFaithfulnessMetric(_async_metric(float("nan")))(
            golden_rows[0], "I don't know."
        )
        == 1.0
    )
    assert RagasFaithfulnessMetric(_async_metric(0.5))(golden_rows[0], "x") == 0.5


def test_run_without_loop_and_thread_loop():
    async def pure():
        return 42

    assert run_without_loop(pure()) == 42

    async def suspends():
        await asyncio.sleep(0)
        return 1

    with pytest.raises(RuntimeError, match="event loop"):
        run_without_loop(suspends())
    assert run_in_thread_loop(suspends()) == 1
    first = metrics_module._thread_state.loop
    assert run_in_thread_loop(suspends()) == 1
    assert metrics_module._thread_state.loop is first, "one loop per thread, reused"


def test_scorer_creates_no_event_loops(monkeypatch, golden_rows):
    """Regression: a new event loop per score() deadlocked threads on Windows."""

    def forbidden(*args, **kwargs):
        raise AssertionError("scoring must not create an event loop")

    monkeypatch.setattr(asyncio, "new_event_loop", forbidden)
    monkeypatch.setattr(asyncio, "run", forbidden)
    scorer = Scorer(list(default_metric_registry()))
    rows = golden_rows * 25

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda r: scorer.score(r, r.reference), rows))
    assert all(r["token_f1"] == 1.0 and r["exact_match"] == 1.0 for r in results)


def test_dspy_metric_bridge(golden_rows):
    goal = OptimizationGoal("token_f1")
    metric = build_dspy_metric(
        goal, Scorer(["token_f1"]), {r.id: r for r in golden_rows}
    )
    row = golden_rows[0]
    gold = SimpleNamespace(row_id=row.id)
    assert metric(gold, SimpleNamespace(answer=row.reference)) == 1.0
    assert metric(gold, SimpleNamespace(answer=None)) == 0.0
    assert (
        metric(gold, SimpleNamespace(), trace=None, pred_name=None, pred_trace=None)
        == 0.0
    )
