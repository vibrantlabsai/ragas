"""R14 (resume), R16 (install hint) and one optimization round with stubs."""

import builtins
import json

import pytest

from ragas.optimizers.prompt_optimization.config import (
    FewShotSearchConfig,
    OptimizationGoal,
)
from ragas.optimizers.prompt_optimization.metrics import Scorer
from ragas.optimizers.prompt_optimization.optimizer import RAGPromptOptimizer


class CountingProposer:
    """Returns two fixed instruction variants, demos untouched."""

    def __init__(self):
        self.calls = 0

    def propose(self, bundle, train_rows, eval_rows, metric):
        self.calls += 1
        return [
            bundle.with_instruction(bundle.instruction, "layer2:original"),
            bundle.with_instruction(
                "Be concise. Say I don't know if the documents lack it.",
                "layer2:instruction=1",
            ),
        ]


def _optimizer(golden_rows, answerer, proposer):
    return RAGPromptOptimizer(
        answerer,
        Scorer(["token_f1", "token_recall"]),
        OptimizationGoal("token_f1", floors={"token_recall": "baseline"}),
        golden_rows,
        proposer=proposer,
        few_shot_config=FewShotSearchConfig(num_candidate_sets=3, max_demos=2),
        num_threads=1,
    )


def test_round_produces_winner_and_files(
    tmp_path, golden_rows, splits, baseline_bundle, stub_answerer
):
    proposer = CountingProposer()
    result = _optimizer(golden_rows, stub_answerer, proposer).optimize(
        baseline_bundle, splits, out_dir=tmp_path
    )
    assert result.winner.source.startswith(("layer1", "layer2"))
    assert result.benchmark.winner.floor_check.passed
    round_dir = tmp_path / "round_1"
    for name in (
        "layer1_state.json",
        "layer2_state.json",
        "round_summary.json",
        "final_prompt.md",
        "final_prompt.json",
        "anthropic_request_template.json",
    ):
        assert (round_dir / name).exists(), name
    summary = json.loads((round_dir / "round_summary.json").read_text(encoding="utf-8"))
    assert summary["benchmark"]["selection_biased"] is True
    assert {c["source"] for c in summary["layer2"]} == {
        "layer2:original",
        "layer2:instruction=1",
    }


def test_resume_skips_completed_layers(
    tmp_path, golden_rows, splits, baseline_bundle, stub_answerer
):
    proposer = CountingProposer()
    first = _optimizer(golden_rows, stub_answerer, proposer).optimize(
        baseline_bundle, splits, out_dir=tmp_path
    )
    second = _optimizer(golden_rows, stub_answerer, proposer).optimize(
        baseline_bundle, splits, out_dir=tmp_path
    )
    assert first.resumed_layers == []
    assert second.resumed_layers == ["layer1", "layer2"]
    assert proposer.calls == 1, "Layer 2 must not rerun on resume"
    assert second.winner.fingerprint() == first.winner.fingerprint()


def test_state_from_other_settings_is_ignored(
    tmp_path, golden_rows, splits, baseline_bundle, stub_answerer
):
    proposer = CountingProposer()
    _optimizer(golden_rows, stub_answerer, proposer).optimize(
        baseline_bundle, splits, out_dir=tmp_path
    )
    other = baseline_bundle.with_instruction(
        "A different starting prompt.", "baseline2"
    )
    result = _optimizer(golden_rows, stub_answerer, proposer).optimize(
        other, splits, out_dir=tmp_path
    )
    assert result.resumed_layers == []
    assert proposer.calls == 2


def test_without_proposer_layer2_is_skipped(
    golden_rows, splits, baseline_bundle, stub_answerer
):
    result = _optimizer(golden_rows, stub_answerer, None).optimize(
        baseline_bundle, splits
    )
    assert result.layer2_candidates == []
    assert result.out_dir is None


def test_import_error_without_dspy(monkeypatch):
    from ragas.optimizers.prompt_optimization import signature

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dspy" or name.startswith("dspy."):
            raise ImportError("No module named 'dspy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"ragas\[dspy\]"):
        signature.import_dspy()


def test_each_round_uses_fresh_seeds(golden_rows, stub_answerer):
    """Round N must not redraw round 1's demo sets or reuse its MIPROv2 seed."""
    from ragas.optimizers.prompt_optimization.config import (
        FewShotSearchConfig,
        OptimizationGoal,
    )
    from ragas.optimizers.prompt_optimization.few_shot_search import (
        candidate_demo_sets,
    )
    from ragas.optimizers.prompt_optimization.metrics import Scorer
    from ragas.optimizers.prompt_optimization.optimizer import (
        ROUND_SEED_STRIDE,
        RAGPromptOptimizer,
    )

    goal = OptimizationGoal()
    opt = RAGPromptOptimizer(
        stub_answerer,
        Scorer(goal.metrics),
        goal,
        golden_rows,
        few_shot_config=FewShotSearchConfig(num_candidate_sets=4, seed=7),
    )
    r1, r2 = opt.round_few_shot_config(1), opt.round_few_shot_config(2)
    assert r1.seed == 7 and r2.seed == 7 + ROUND_SEED_STRIDE
    train = [r for r in golden_rows if r.split == "train"]
    labels1 = {s for s, _ in candidate_demo_sets(train, r1) if s >= 0}
    labels2 = {s for s, _ in candidate_demo_sets(train, r2) if s >= 0}
    assert labels1 and labels2 and not labels1 & labels2, "labels name the real seed"

    calls = []

    class RecordingProposer:
        def propose(self, bundle, train_rows, eval_rows, metric, seed_offset=0):
            calls.append(seed_offset)
            return []

    opt.proposer = RecordingProposer()
    splits = {
        s: [r for r in golden_rows if r.split == s] for s in ("train", "eval", "test")
    }
    from ragas.optimizers.prompt_optimization.bundle import PromptBundle

    start = PromptBundle("Answer from the context.")
    opt.optimize(start, splits, round_no=1)
    opt.optimize(start, splits, round_no=3)
    assert calls == [0, 2]
