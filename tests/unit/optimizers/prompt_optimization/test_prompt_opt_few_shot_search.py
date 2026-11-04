"""R5: Layer 1 golden random search keeps the instruction and uses golden rows only."""

from ragas.optimizers.prompt_optimization.config import (
    FewShotSearchConfig,
    OptimizationGoal,
)
from ragas.optimizers.prompt_optimization.few_shot_search import (
    candidate_demo_sets,
    search_few_shots,
)
from ragas.optimizers.prompt_optimization.metrics import Scorer


def test_seed_scheme_matches_bootstrap_random_search(splits):
    train = splits["train"]
    sets = candidate_demo_sets(
        train, FewShotSearchConfig(num_candidate_sets=8, max_demos=3)
    )
    seeds = [s for s, _ in sets]
    assert seeds[:2] == [-3, -2]
    assert sets[0][1] == []  # zero-shot
    assert [r.id for r in sets[1][1]] == [r.id for r in train[:3]]  # first k
    for seed, rows in sets[2:]:
        assert 1 <= len(rows) <= 3
        assert seed >= 0
    keys = [tuple(r.id for r in rows) for _, rows in sets]
    assert len(keys) == len(set(keys)), "duplicate demo sets must be removed"


def test_seed_scheme_is_reproducible(splits):
    cfg = FewShotSearchConfig(num_candidate_sets=6, max_demos=2, seed=3)
    a = candidate_demo_sets(splits["train"], cfg)
    b = candidate_demo_sets(splits["train"], cfg)
    assert [[r.id for r in rows] for _, rows in a] == [
        [r.id for r in rows] for _, rows in b
    ]


def test_search_freezes_instruction_and_uses_verbatim_train_rows(
    splits, baseline_bundle, stub_answerer
):
    result = search_few_shots(
        baseline_bundle,
        splits["train"],
        splits["eval"],
        stub_answerer,
        Scorer(["token_f1", "token_recall"]),
        OptimizationGoal("token_f1"),
        FewShotSearchConfig(num_candidate_sets=6, max_demos=2),
        num_threads=1,
    )
    train_by_id = {r.id: r for r in splits["train"]}
    other_ids = {r.id for name in ("eval", "test", "holdout") for r in splits[name]}
    for cand in result.candidates:
        assert cand.bundle.instruction == baseline_bundle.instruction
        for shot in cand.bundle.few_shots:
            assert shot.row_id not in other_ids, "demo leaked from a non-train split"
            row = train_by_id[shot.row_id]
            assert (shot.question, shot.contexts, shot.answer) == (
                row.question,
                row.contexts,
                row.reference,
            )
    # Demos make the stub concise, so the best set beats zero-shot.
    zero = next(c for c in result.candidates if c.seed == -3)
    assert (
        result.best.report.aggregates["token_f1"] > zero.report.aggregates["token_f1"]
    )
    assert result.best.bundle.few_shots
    assert result.baseline_eval is not None


def test_tie_break_prefers_fewer_demos(splits, baseline_bundle):
    # Every candidate scores the same: the zero-demo one must win.
    def flat(bundle, row):
        return row.reference

    result = search_few_shots(
        baseline_bundle,
        splits["train"],
        splits["eval"],
        flat,
        Scorer(["token_f1"]),
        OptimizationGoal("token_f1"),
        FewShotSearchConfig(num_candidate_sets=4, max_demos=2),
        num_threads=1,
    )
    assert result.best.bundle.few_shots == []


def test_floor_failure_falls_back_to_start(splits, baseline_bundle, stub_answerer):
    # Impossible floor: nothing passes, the starting prompt is kept.
    goal = OptimizationGoal("token_f1", floors={"token_recall": 2.0})
    result = search_few_shots(
        baseline_bundle,
        splits["train"],
        splits["eval"],
        stub_answerer,
        Scorer(["token_f1", "token_recall"]),
        goal,
        FewShotSearchConfig(num_candidate_sets=3, max_demos=2),
        num_threads=1,
    )
    assert result.no_candidate_passed
    assert result.best.bundle.few_shots == baseline_bundle.few_shots


def test_current_demos_are_a_candidate(splits, baseline_bundle, stub_answerer):
    from ragas.optimizers.prompt_optimization.bundle import FewShot

    start = baseline_bundle.with_few_shots(
        [FewShot.from_row(splits["train"][-1])], source="current"
    )
    result = search_few_shots(
        start,
        splits["train"],
        splits["eval"],
        stub_answerer,
        Scorer(["token_f1"]),
        OptimizationGoal("token_f1"),
        FewShotSearchConfig(num_candidate_sets=0, max_demos=2),
        baseline_aggregates={"token_f1": 0.0},
        num_threads=2,
    )
    assert "current" in [c.seed for c in result.candidates]
