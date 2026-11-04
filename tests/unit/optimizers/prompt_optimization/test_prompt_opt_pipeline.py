"""R9 integration: full tool flow on the shipped 80-row dataset, simulated models.

ragas eval (before) -> Layer 1 -> Layer 2 (real MIPROv2) -> test benchmark ->
holdout improvement loop -> ragas eval (after) -> comparison.

Three model behaviours prove the loop's verdicts are trustworthy:
- sensitive: a real gain must be detected and accepted;
- insensitive (negative control): no improvement may be reported;
- recall_dropping (floor control): candidates below the recall floor are rejected.
"""

import json

import pytest

pytest.importorskip("dspy")
pytest.importorskip("optuna")

from prompt_opt_testdata import BASELINE_PROMPT, GOAL_FILE, GOLDEN_DATASET  # noqa: E402

from ragas.optimizers.prompt_optimization import (  # noqa: E402
    InstructionSearchConfig,
    LoopConfig,
    OptimizationGoal,
    run_pipeline,
)


def _run(tmp_path, behavior):
    return run_pipeline(
        dataset=GOLDEN_DATASET,
        system_prompt=BASELINE_PROMPT.read_text(encoding="utf-8"),
        goal=OptimizationGoal.from_json(GOAL_FILE),
        out_dir=tmp_path / behavior,
        mode="simulated",
        simulated_behavior=behavior,
        instruction_config=InstructionSearchConfig(
            num_candidates=4, num_trials=6, minibatch_size=10
        ),
        loop_config=LoopConfig(max_rounds=2, n_bootstrap=500),
        num_threads=4,
    )


def test_sensitive_model_improves(tmp_path):
    result = _run(tmp_path, "sensitive")
    target = result.comparison.target_metric
    assert result.loop is not None and result.loop.improved
    assert result.after.aggregates[target] > result.before.aggregates[target]
    assert result.comparison.target_ci[0] > 0, "gain must be significant"
    floors = OptimizationGoal.from_json(GOAL_FILE).resolve_floors(
        result.before.aggregates
    )
    for metric, floor in floors.items():
        assert result.after.aggregates[metric] >= floor
    out = result.out_dir
    for name in (
        "comparison.md",
        "comparison.json",
        "loop_report.md",
        "loop_report.json",
        "spend.json",
        "run_info.json",
        "final/final_prompt.md",
        "final/anthropic_request_template.json",
        "ragas_eval/experiments/before.csv",
        "ragas_eval/experiments/after.csv",
    ):
        assert (out / name).exists(), name
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    assert info["mode"] == "simulated:sensitive" and info["proposer_calls"] > 0
    # Exported demos are verbatim golden train rows.
    golden = {
        json.loads(line)["id"]: json.loads(line)
        for line in GOLDEN_DATASET.read_text(encoding="utf-8").splitlines()
    }
    for shot in result.final_bundle.few_shots:
        assert golden[shot.row_id]["split"] == "train"
        assert shot.answer == golden[shot.row_id]["reference"]


def test_insensitive_model_reports_no_improvement(tmp_path):
    result = _run(tmp_path, "insensitive")
    assert not result.loop.improved
    assert result.final_bundle.source == "baseline"
    assert result.after.aggregates == result.before.aggregates


def test_recall_dropping_candidates_fail_the_floor(tmp_path):
    result = _run(tmp_path, "recall_dropping")
    summary = json.loads(
        (result.out_dir / "round_1" / "round_summary.json").read_text(encoding="utf-8")
    )
    leaderboard = summary["benchmark"]["leaderboard"]
    rejected = [e for e in leaderboard if "token_recall" in e["failed_floors"]]
    assert rejected, (
        "a concise, fact-dropping candidate must be rejected by the recall floor"
    )
    assert summary["benchmark"]["winner"]["floors_passed"]
    floors = OptimizationGoal.from_json(GOAL_FILE).resolve_floors(
        result.before.aggregates
    )
    assert result.after.aggregates["token_recall"] >= floors["token_recall"]
