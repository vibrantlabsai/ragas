"""Layer 1: golden random search over few-shot demo sets, instruction frozen."""

from __future__ import annotations

import random
import typing as t
from dataclasses import dataclass, field

from ragas.optimizers.prompt_optimization.bundle import FewShot, GoldenRow, PromptBundle
from ragas.optimizers.prompt_optimization.config import (
    FewShotSearchConfig,
    OptimizationGoal,
)
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


def candidate_demo_sets(
    train_rows: t.Sequence[GoldenRow], config: FewShotSearchConfig
) -> t.List[t.Tuple[int, t.List[GoldenRow]]]:
    """Demo sets in BootstrapRS's seed order, using golden rows only.

    - seed -3: no demos (zero-shot)
    - seed -2: the first ``max_demos`` training rows, in file order
    - seeds 0..N-1: training rows shuffled with ``Random(seed)``, then the
      first ``Random(seed).randint(1, max_demos)`` of them

    Duplicate sets (same rows in the same order) are kept only once.
    """
    k = min(config.max_demos, len(train_rows))
    sets: t.List[t.Tuple[int, t.List[GoldenRow]]] = [(-3, [])]
    if k > 0:
        sets.append((-2, list(train_rows[:k])))
        for seed in range(config.num_candidate_sets):
            effective = config.seed + seed
            shuffled = list(train_rows)
            random.Random(effective).shuffle(shuffled)
            size = random.Random(effective).randint(1, k)
            sets.append((effective, shuffled[:size]))

    unique: t.List[t.Tuple[int, t.List[GoldenRow]]] = []
    seen: t.Set[t.Tuple[str, ...]] = set()
    for seed, rows in sets:
        key = tuple(r.id for r in rows)
        if key not in seen:
            seen.add(key)
            unique.append((seed, rows))
    return unique


@dataclass
class FewShotCandidate:
    seed: t.Union[int, str]
    bundle: PromptBundle
    report: EvaluationReport
    floor_check: FloorCheck

    def summary(self, target_metric: str) -> t.Dict[str, t.Any]:
        return {
            "seed": self.seed,
            "num_demos": len(self.bundle.few_shots),
            "demo_ids": [s.row_id for s in self.bundle.few_shots],
            "target": self.report.aggregates[target_metric],
            "aggregates": self.report.aggregates,
            "floors_passed": self.floor_check.passed,
            "failed_floors": self.floor_check.failed_metrics,
        }


@dataclass
class FewShotSearchResult:
    best: FewShotCandidate
    candidates: t.List[FewShotCandidate] = field(default_factory=list)
    baseline_eval: t.Optional[EvaluationReport] = None
    resolved_floors: t.Dict[str, float] = field(default_factory=dict)
    no_candidate_passed: bool = False


def _rank_key(c: FewShotCandidate, target: str) -> t.Tuple[float, int]:
    # Highest target first; fewer demos break ties (shorter, cheaper prompt).
    return (-c.report.aggregates[target], len(c.bundle.few_shots))


def search_few_shots(
    base_bundle: PromptBundle,
    train_rows: t.Sequence[GoldenRow],
    eval_rows: t.Sequence[GoldenRow],
    answer_fn: AnswerFn,
    scorer: Scorer,
    goal: OptimizationGoal,
    config: t.Optional[FewShotSearchConfig] = None,
    baseline_aggregates: t.Optional[t.Mapping[str, float]] = None,
    num_threads: int = 8,
) -> FewShotSearchResult:
    """Pick the demo set that maximizes the target on eval while passing floors.

    The instruction of ``base_bundle`` is never changed. If ``base_bundle``
    already has demos, that exact set is also a candidate, so the search can
    always keep what it started with.

    Args:
        baseline_aggregates: eval-split aggregates of the original prompt, used
            to resolve ``"baseline"`` floors. Defaults to evaluating
            ``base_bundle`` itself.
    """
    config = config or FewShotSearchConfig()
    baseline_eval = None
    if baseline_aggregates is None:
        baseline_eval = evaluate_bundle(
            base_bundle, eval_rows, answer_fn, scorer, "eval", num_threads
        )
        baseline_aggregates = baseline_eval.aggregates
    floors = goal.resolve_floors(baseline_aggregates)

    demo_sets: t.List[t.Tuple[t.Union[int, str], t.List[FewShot]]] = [
        (seed, [FewShot.from_row(r) for r in rows])
        for seed, rows in candidate_demo_sets(train_rows, config)
    ]
    if base_bundle.few_shots:
        current_ids = [s.row_id for s in base_bundle.few_shots]
        if all([s.row_id for s in shots] != current_ids for _, shots in demo_sets):
            demo_sets.append(("current", list(base_bundle.few_shots)))

    candidates = []
    for seed, shots in demo_sets:
        bundle = base_bundle.with_few_shots(shots, source=f"layer1:seed={seed}")
        report = evaluate_bundle(
            bundle, eval_rows, answer_fn, scorer, "eval", num_threads
        )
        candidates.append(
            FewShotCandidate(
                seed, bundle, report, check_floors(report.aggregates, floors)
            )
        )

    target = goal.target_metric
    ranked = sorted(candidates, key=lambda c: _rank_key(c, target))
    passing = [c for c in ranked if c.floor_check.passed]
    if passing:
        best, none_passed = passing[0], False
    else:
        # Nothing satisfies the floors: keep the starting demos unchanged.
        unchanged = [
            c for c in candidates if c.bundle.few_shots == base_bundle.few_shots
        ]
        best, none_passed = (unchanged[0] if unchanged else ranked[0]), True

    return FewShotSearchResult(
        best=best,
        candidates=ranked,
        baseline_eval=baseline_eval,
        resolved_floors=floors,
        no_candidate_passed=none_passed,
    )
