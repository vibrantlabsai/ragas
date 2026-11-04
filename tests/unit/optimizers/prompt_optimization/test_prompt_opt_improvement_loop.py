"""R9: the improvement loop accepts only significant, floor-respecting holdout gains."""

import json

import pytest

from ragas.optimizers.prompt_optimization.budget import BudgetExceeded, BudgetTracker
from ragas.optimizers.prompt_optimization.config import (
    BudgetConfig,
    LoopConfig,
    OptimizationGoal,
)
from ragas.optimizers.prompt_optimization.improvement_loop import (
    FLOOR_VIOLATION,
    IMPROVED,
    NO_CHANGE,
    REGRESSED,
    UNCHANGED_PROMPT,
    paired_bootstrap_ci,
    run_improvement_loop,
)
from ragas.optimizers.prompt_optimization.metrics import Scorer

SCORER = Scorer(["token_f1", "token_recall"])
LOOP = LoopConfig(max_rounds=3, min_gain=0.02, n_bootstrap=500)


def _run(baseline, optimize_fn, holdout, answer_fn, goal=None, **kw):
    return run_improvement_loop(
        baseline,
        optimize_fn,
        holdout,
        answer_fn,
        SCORER,
        goal or OptimizationGoal("token_f1", target_value=1.0),
        kw.pop("config", LOOP),
        num_threads=1,
        **kw,
    )


def test_bootstrap_ci_properties():
    assert paired_bootstrap_ci([]) == (0.0, 0.0)
    assert paired_bootstrap_ci([0.5] * 10) == (0.5, 0.5)
    low, high = paired_bootstrap_ci([0.0, 1.0] * 10, n_bootstrap=500)
    assert low < 0.5 < high
    assert paired_bootstrap_ci([0.1, 0.2, 0.3], seed=4) == paired_bootstrap_ci(
        [0.1, 0.2, 0.3], seed=4
    )


def test_real_gain_is_accepted(splits, baseline_bundle, stub_answerer):
    concise = baseline_bundle.with_instruction(
        "Be concise. Say I don't know if missing.", "better"
    )
    report = _run(
        baseline_bundle, lambda cur, n: concise, splits["holdout"], stub_answerer
    )
    assert report.rounds[0].verdict == IMPROVED
    assert report.improved and report.final_bundle.source == "better"
    assert report.final.aggregates["token_f1"] > report.baseline.aggregates["token_f1"]
    assert report.stop_reason.startswith("target")


def test_no_gain_is_not_accepted(splits, baseline_bundle):
    # Negative control: answers ignore the prompt, so nothing can improve.
    def insensitive(bundle, row):
        return f"Great question! {row.reference}. More words here."

    other = baseline_bundle.with_instruction("Be concise.", "other")
    report = _run(baseline_bundle, lambda cur, n: other, splits["holdout"], insensitive)
    assert report.rounds[0].verdict == NO_CHANGE
    assert not report.improved
    assert report.final_bundle is baseline_bundle
    assert report.cumulative_delta == 0.0


def test_regression_detected(splits, baseline_bundle, stub_answerer):
    def worse_for_candidate(bundle, row):
        # Baseline is decent but not perfect (so the target is not already met).
        return (
            "nothing useful"
            if bundle.source == "worse"
            else f"{row.reference} and more"
        )

    worse = baseline_bundle.with_instruction("x", "worse")
    report = _run(
        baseline_bundle, lambda cur, n: worse, splits["holdout"], worse_for_candidate
    )
    assert report.rounds[0].verdict == REGRESSED
    assert report.final_bundle is baseline_bundle


def test_floor_violation_rejected(splits, baseline_bundle):
    # Floor control: the candidate raises F1 but loses recall.
    def answer(bundle, row):
        if bundle.source == "short":
            return row.reference.split()[0]
        return f"{row.reference} " + "padding " * 30

    goal = OptimizationGoal(
        "token_f1", target_value=1.0, floors={"token_recall": "baseline"}
    )
    short = baseline_bundle.with_instruction("short", "short")
    report = _run(
        baseline_bundle, lambda cur, n: short, splits["holdout"], answer, goal=goal
    )
    assert (
        report.rounds[0].after.aggregates["token_f1"]
        > report.baseline.aggregates["token_f1"]
    )
    assert report.rounds[0].verdict == FLOOR_VIOLATION
    assert report.final_bundle is baseline_bundle


def test_unchanged_prompt_stops(splits, baseline_bundle, stub_answerer):
    report = _run(baseline_bundle, lambda cur, n: cur, splits["holdout"], stub_answerer)
    assert report.rounds[0].verdict == UNCHANGED_PROMPT
    assert len(report.rounds) == 1


def test_target_already_reached(splits, baseline_bundle):
    report = _run(
        baseline_bundle,
        lambda cur, n: pytest.fail("must not optimize"),
        splits["holdout"],
        lambda b, r: r.reference,
    )
    assert report.rounds == [] and "already" in report.stop_reason


def test_multiple_rounds_until_no_gain(splits, baseline_bundle, stub_answerer):
    steps = [
        baseline_bundle.with_instruction("Be concise.", "r1"),
        baseline_bundle.with_instruction(
            "Be concise. Say I don't know if missing.", "r2"
        ),
        baseline_bundle.with_instruction(
            "Be concise. Say I don't know if missing!", "r3"
        ),
    ]
    goal = OptimizationGoal("token_f1", target_value=1.0)
    report = _run(
        baseline_bundle,
        lambda cur, n: steps[n - 1],
        splits["holdout"],
        stub_answerer,
        goal=goal,
        config=LoopConfig(max_rounds=3, min_gain=0.02, n_bootstrap=300),
    )
    assert [r.verdict for r in report.rounds][0] == IMPROVED
    assert report.cumulative_ci[0] > 0


def test_budget_exhaustion_stops_cleanly(splits, baseline_bundle, stub_answerer):
    budget = BudgetTracker(BudgetConfig(max_usd=1.0))

    def optimize(cur, n):
        raise BudgetExceeded("cap reached")

    report = _run(
        baseline_bundle, optimize, splits["holdout"], stub_answerer, budget=budget
    )
    assert report.stop_reason.startswith("budget exhausted")
    assert report.final_bundle is baseline_bundle


def test_report_save(tmp_path, splits, baseline_bundle, stub_answerer):
    better = baseline_bundle.with_instruction("Be concise.", "better")
    seen = []
    report = _run(
        baseline_bundle,
        lambda c, n: better,
        splits["holdout"],
        stub_answerer,
        on_round=seen.append,
    )
    paths = report.save(tmp_path)
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["rounds"][0]["verdict"] == report.rounds[0].verdict
    assert "| Round | Verdict |" in paths["markdown"].read_text(encoding="utf-8")
    assert len(seen) == len(report.rounds)
