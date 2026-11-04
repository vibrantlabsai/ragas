"""R6: Layer 2 runs real MIPROv2 (simulated models) with the Layer 1 demos frozen."""

import pytest

pytest.importorskip("dspy")
pytest.importorskip("optuna")

from ragas.optimizers.prompt_optimization.budget import BudgetTracker  # noqa: E402
from ragas.optimizers.prompt_optimization.bundle import FewShot  # noqa: E402
from ragas.optimizers.prompt_optimization.config import (  # noqa: E402
    InstructionSearchConfig,
    ModelConfig,
    OptimizationGoal,
)
from ragas.optimizers.prompt_optimization.instruction_search import (  # noqa: E402
    MIPROv2InstructionProposer,
)
from ragas.optimizers.prompt_optimization.metrics import Scorer, build_dspy_metric  # noqa: E402
from ragas.optimizers.prompt_optimization.runtime import build_simulated_runtime  # noqa: E402


def test_mipro_keeps_demos_and_original_instruction(
    golden_rows, splits, baseline_bundle
):
    runtime = build_simulated_runtime(golden_rows, ModelConfig(), BudgetTracker())
    demos = [FewShot.from_row(splits["train"][0]), FewShot.from_row(splits["train"][3])]
    layer1_best = baseline_bundle.with_few_shots(demos, source="layer1:seed=0")
    proposer = MIPROv2InstructionProposer(
        runtime.student_lm,
        runtime.proposer_lm,
        InstructionSearchConfig(num_candidates=3, num_trials=3, minibatch_size=10),
        num_threads=2,
    )
    metric = build_dspy_metric(
        OptimizationGoal("token_f1"),
        Scorer(["token_f1"]),
        {r.id: r for r in golden_rows},
    )
    bundles = proposer.propose(layer1_best, splits["train"], splits["eval"], metric)

    assert bundles[0].source == "layer2:original"
    assert bundles[0].instruction == baseline_bundle.instruction
    assert len(bundles) >= 2, "MIPROv2 should propose new instructions"
    assert len({b.instruction for b in bundles}) == len(bundles), (
        "instructions are de-duplicated"
    )
    for b in bundles:
        assert b.few_shots == demos, "Layer 1 demos must stay frozen"
    assert proposer.last_run["demos_frozen_in_mipro"] is True
    assert runtime.proposer_engine.calls > 0, (
        "the proposer model wrote the instructions"
    )


def test_needs_two_training_rows(golden_rows, splits, baseline_bundle):
    runtime = build_simulated_runtime(golden_rows, ModelConfig(), BudgetTracker())
    proposer = MIPROv2InstructionProposer(runtime.student_lm, runtime.proposer_lm)
    with pytest.raises(ValueError, match="2 training rows"):
        proposer.propose(
            baseline_bundle, splits["train"][:1], splits["eval"], lambda *a, **k: 0.0
        )
