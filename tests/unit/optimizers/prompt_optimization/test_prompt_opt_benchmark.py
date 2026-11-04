"""R8: the test benchmark ranks candidates, applies floors, and breaks ties."""

from ragas.optimizers.prompt_optimization.benchmark import benchmark_candidates
from ragas.optimizers.prompt_optimization.bundle import PromptBundle
from ragas.optimizers.prompt_optimization.config import OptimizationGoal
from ragas.optimizers.prompt_optimization.metrics import Scorer

SCORER = Scorer(["token_f1", "token_recall", "exact_match"])


def test_ranks_by_target_and_picks_best(splits, baseline_bundle, stub_answerer):
    concise = baseline_bundle.with_instruction("Be concise.", "concise")
    refuse = baseline_bundle.with_instruction(
        "Be concise. Say I don't know if missing.", "refuse"
    )
    result = benchmark_candidates(
        baseline_bundle,
        [concise, refuse],
        splits["test"],
        stub_answerer,
        SCORER,
        OptimizationGoal("token_f1"),
        num_threads=1,
    )
    assert result.winner.bundle.source == "refuse"
    assert [e.rank for e in result.leaderboard] == [1, 2, 3]
    assert result.leaderboard[-1].bundle.source == "baseline"
    assert result.selection_biased is True
    assert result.to_dict("token_f1")["winner"]["source"] == "refuse"


def test_floor_failures_rank_below_passing(splits, baseline_bundle):
    def answer(bundle, row):
        if bundle.source == "short":  # highest F1 here? no: drops recall
            return row.reference.split()[0]
        if bundle.source == "verbose":
            return (
                f"{row.reference} plus a lot of extra words that hurt precision badly"
            )
        return f"{row.reference} and more"

    short = baseline_bundle.with_instruction("short", "short")
    verbose = baseline_bundle.with_instruction("verbose", "verbose")
    goal = OptimizationGoal("token_f1", floors={"token_recall": "baseline"})
    result = benchmark_candidates(
        baseline_bundle,
        [short, verbose],
        splits["test"],
        answer,
        SCORER,
        goal,
        num_threads=1,
    )
    by_source = {e.bundle.source: e for e in result.leaderboard}
    assert not by_source["short"].floor_check.passed
    assert "token_recall" in by_source["short"].floor_check.failed_metrics
    assert result.winner.floor_check.passed
    assert result.leaderboard[-1].bundle.source == "short"


def test_no_passing_candidate_returns_baseline(splits, baseline_bundle, stub_answerer):
    goal = OptimizationGoal("token_f1", floors={"exact_match": 2.0})
    result = benchmark_candidates(
        baseline_bundle,
        [baseline_bundle.with_instruction("Be concise.", "c")],
        splits["test"],
        stub_answerer,
        SCORER,
        goal,
        num_threads=1,
    )
    assert result.no_candidate_passed
    assert result.winner.bundle is baseline_bundle


def test_tie_breaks_on_shorter_prompt_and_dedupes(splits, baseline_bundle):
    def same(bundle, row):
        return row.reference

    long_one = PromptBundle("x" * 500, source="long")
    short_one = PromptBundle("x", source="short")
    result = benchmark_candidates(
        long_one,
        [short_one, PromptBundle("x", source="dupe")],
        splits["test"],
        same,
        SCORER,
        OptimizationGoal("token_f1"),
        num_threads=1,
    )
    assert result.winner.bundle.source == "short"
    assert len(result.leaderboard) == 2, "identical prompt must be benchmarked once"
