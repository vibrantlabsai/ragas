"""Benchmark every candidate prompt on the test split and pick the winner."""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from ragas.optimizers.prompt_optimization.bundle import GoldenRow, PromptBundle
from ragas.optimizers.prompt_optimization.config import OptimizationGoal
from ragas.optimizers.prompt_optimization.metrics import (
    FloorCheck,
    Scorer,
    check_floors,
)
from ragas.optimizers.prompt_optimization.rag_evaluation import (
    AnswerFn,
    EvaluationReport,
    evaluate_bundle,
)


@dataclass
class LeaderboardEntry:
    rank: int
    bundle: PromptBundle
    report: EvaluationReport
    floor_check: FloorCheck

    def summary(self, target_metric: str) -> t.Dict[str, t.Any]:
        return {
            "rank": self.rank,
            "source": self.bundle.source,
            "fingerprint": self.bundle.fingerprint(),
            "num_demos": len(self.bundle.few_shots),
            "prompt_chars": self.bundle.prompt_length,
            "target": self.report.aggregates[target_metric],
            "aggregates": self.report.aggregates,
            "floors_passed": self.floor_check.passed,
            "failed_floors": self.floor_check.failed_metrics,
        }


@dataclass
class BenchmarkResult:
    winner: LeaderboardEntry
    leaderboard: t.List[LeaderboardEntry] = field(default_factory=list)
    baseline: t.Optional[LeaderboardEntry] = None
    resolved_floors: t.Dict[str, float] = field(default_factory=dict)
    no_candidate_passed: bool = False
    #: The winner was picked on test, so its test score is an optimistic estimate.
    selection_biased: bool = True

    def to_dict(self, target_metric: str) -> t.Dict[str, t.Any]:
        return {
            "winner": self.winner.summary(target_metric),
            "baseline": self.baseline.summary(target_metric) if self.baseline else None,
            "resolved_floors": self.resolved_floors,
            "no_candidate_passed": self.no_candidate_passed,
            "selection_biased": self.selection_biased,
            "leaderboard": [e.summary(target_metric) for e in self.leaderboard],
        }


def benchmark_candidates(
    baseline: PromptBundle,
    candidates: t.Sequence[PromptBundle],
    test_rows: t.Sequence[GoldenRow],
    answer_fn: AnswerFn,
    scorer: Scorer,
    goal: OptimizationGoal,
    baseline_aggregates: t.Optional[t.Mapping[str, float]] = None,
    num_threads: int = 8,
) -> BenchmarkResult:
    """Score the baseline and every candidate on test; rank and pick the winner.

    Ranking: candidates passing every floor first, then the highest target,
    then the shortest prompt. ``"baseline"`` floors resolve against the
    baseline's test scores unless ``baseline_aggregates`` is given. If no
    candidate passes, the baseline is returned so nothing regresses.
    """
    baseline_report = evaluate_bundle(
        baseline, test_rows, answer_fn, scorer, "test", num_threads
    )
    floors = goal.resolve_floors(baseline_aggregates or baseline_report.aggregates)

    seen = {baseline.fingerprint()}
    scored = [(baseline, baseline_report)]
    for bundle in candidates:
        fp = bundle.fingerprint()
        if fp in seen:
            continue
        seen.add(fp)
        scored.append(
            (
                bundle,
                evaluate_bundle(
                    bundle, test_rows, answer_fn, scorer, "test", num_threads
                ),
            )
        )

    target = goal.target_metric

    def key(item: t.Tuple[PromptBundle, EvaluationReport, FloorCheck]) -> t.Tuple:
        bundle, report, check = item
        return (not check.passed, -report.aggregates[target], bundle.prompt_length)

    ranked = sorted(
        ((b, r, check_floors(r.aggregates, floors)) for b, r in scored), key=key
    )
    leaderboard = [
        LeaderboardEntry(rank=i, bundle=b, report=r, floor_check=c)
        for i, (b, r, c) in enumerate(ranked, start=1)
    ]
    baseline_entry = next(e for e in leaderboard if e.bundle is baseline)
    passing = [e for e in leaderboard if e.floor_check.passed]
    winner = passing[0] if passing else baseline_entry

    return BenchmarkResult(
        winner=winner,
        leaderboard=leaderboard,
        baseline=baseline_entry,
        resolved_floors=floors,
        no_candidate_passed=not passing,
    )
