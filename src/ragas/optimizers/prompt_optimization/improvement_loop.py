"""Improvement loop: does the optimized prompt really beat the current one?

Each round optimizes the current prompt (using train/eval/test only), then runs
the current and the optimized prompt through the same RAG evaluation on the
holdout split, which the optimizer never sees. A round is accepted only when
the paired per-row gain on the target metric is significantly positive and
every floor still holds on holdout.
"""

from __future__ import annotations

import json
import random
import statistics
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from ragas.optimizers.prompt_optimization.budget import BudgetExceeded, BudgetTracker
from ragas.optimizers.prompt_optimization.bundle import GoldenRow, PromptBundle
from ragas.optimizers.prompt_optimization.config import LoopConfig, OptimizationGoal
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

IMPROVED = "improved"
NO_CHANGE = "no significant change"
REGRESSED = "regressed"
FLOOR_VIOLATION = "rejected: floor violated"
UNCHANGED_PROMPT = "optimizer kept the current prompt"


def paired_bootstrap_ci(
    deltas: t.Sequence[float],
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> t.Tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean paired difference."""
    if not deltas:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(
        statistics.fmean(deltas[rng.randrange(n)] for _ in range(n))
        for _ in range(n_bootstrap)
    )
    alpha = (1.0 - confidence) / 2.0
    low = means[int(alpha * (n_bootstrap - 1))]
    high = means[int((1.0 - alpha) * (n_bootstrap - 1))]
    return (round(low, 6), round(high, 6))


@dataclass
class RoundResult:
    round: int
    verdict: str
    accepted: bool
    before: EvaluationReport
    after: t.Optional[EvaluationReport]
    candidate: t.Optional[PromptBundle]
    mean_delta: float = 0.0
    ci: t.Tuple[float, float] = (0.0, 0.0)
    floor_check: t.Optional[FloorCheck] = None
    spend_usd: float = 0.0
    details: t.Dict[str, t.Any] = field(default_factory=dict)

    def to_dict(self, target_metric: str) -> t.Dict[str, t.Any]:
        return {
            "round": self.round,
            "verdict": self.verdict,
            "accepted": self.accepted,
            "before": self.before.aggregates,
            "after": self.after.aggregates if self.after else None,
            "target_before": self.before.aggregates.get(target_metric),
            "target_after": self.after.aggregates.get(target_metric)
            if self.after
            else None,
            "mean_delta": self.mean_delta,
            "ci": list(self.ci),
            "floors": self.floor_check.details if self.floor_check else [],
            "candidate_source": self.candidate.source if self.candidate else None,
            "candidate_fingerprint": self.candidate.fingerprint()
            if self.candidate
            else None,
            "spend_usd": round(self.spend_usd, 6),
            **self.details,
        }


@dataclass
class LoopReport:
    goal: OptimizationGoal
    baseline: EvaluationReport
    final_bundle: PromptBundle
    final: EvaluationReport
    rounds: t.List[RoundResult] = field(default_factory=list)
    stop_reason: str = ""
    cumulative_delta: float = 0.0
    cumulative_ci: t.Tuple[float, float] = (0.0, 0.0)
    spend: t.Dict[str, t.Any] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        return any(r.accepted for r in self.rounds)

    def to_dict(self) -> t.Dict[str, t.Any]:
        target = self.goal.target_metric
        return {
            "goal": self.goal.to_dict(),
            "stop_reason": self.stop_reason,
            "improved": self.improved,
            "baseline_holdout": self.baseline.aggregates,
            "final_holdout": self.final.aggregates,
            "cumulative_delta": self.cumulative_delta,
            "cumulative_ci": list(self.cumulative_ci),
            "final_source": self.final_bundle.source,
            "final_fingerprint": self.final_bundle.fingerprint(),
            "rounds": [r.to_dict(target) for r in self.rounds],
            "spend": self.spend,
        }

    def to_markdown(self) -> str:
        target = self.goal.target_metric
        lines = [
            "# Prompt optimization: improvement loop report",
            "",
            f"- Target metric: `{target}` (goal {self.goal.target_value})",
            f"- Floors: {self.goal.floors or 'none set'}",
            f"- Stop reason: {self.stop_reason}",
            f"- Improved: **{'yes' if self.improved else 'no'}**",
            f"- Holdout {target}: {self.baseline.aggregates.get(target, 0):.4f} -> "
            f"{self.final.aggregates.get(target, 0):.4f} "
            f"(gain {self.cumulative_delta:+.4f}, 95% CI "
            f"[{self.cumulative_ci[0]:+.4f}, {self.cumulative_ci[1]:+.4f}])",
            f"- Spend: ${self.spend.get('total_usd', 0):.4f} of ${self.spend.get('max_usd', 0):.2f}",
            "",
            "| Round | Verdict | Before | After | Mean gain | CI | Floors |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in self.rounds:
            after = f"{r.after.aggregates[target]:.4f}" if r.after else "-"
            floors = (
                "pass"
                if (r.floor_check and r.floor_check.passed)
                else (", ".join(r.floor_check.failed_metrics) if r.floor_check else "-")
            )
            lines.append(
                f"| {r.round} | {r.verdict} | {r.before.aggregates[target]:.4f} | {after} | "
                f"{r.mean_delta:+.4f} | [{r.ci[0]:+.4f}, {r.ci[1]:+.4f}] | {floors} |"
            )
        return "\n".join(lines) + "\n"

    def save(self, out_dir: t.Union[str, Path]) -> t.Dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {"json": out / "loop_report.json", "markdown": out / "loop_report.md"}
        paths["json"].write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        paths["markdown"].write_text(self.to_markdown(), encoding="utf-8")
        return paths


#: Optimizes a prompt for one round and returns the candidate to test.
OptimizeFn = t.Callable[[PromptBundle, int], PromptBundle]


def _paired_deltas(
    before: EvaluationReport, after: EvaluationReport, metric: str
) -> t.List[float]:
    b = before.scores_by_row(metric)
    a = after.scores_by_row(metric)
    return [a[row_id] - b[row_id] for row_id in b]


def run_improvement_loop(
    baseline: PromptBundle,
    optimize_fn: OptimizeFn,
    holdout_rows: t.Sequence[GoldenRow],
    answer_fn: AnswerFn,
    scorer: Scorer,
    goal: OptimizationGoal,
    config: t.Optional[LoopConfig] = None,
    budget: t.Optional[BudgetTracker] = None,
    num_threads: int = 8,
    on_round: t.Optional[t.Callable[[RoundResult], None]] = None,
) -> LoopReport:
    """Repeat optimize-then-verify rounds until the gain stops.

    ``"baseline"`` floors always refer to the *original* prompt's holdout
    scores, so accepted rounds can never drift below the starting point.
    """
    config = config or LoopConfig()
    target = goal.target_metric

    if budget is not None:
        budget.stage = "loop:holdout"
    baseline_report = evaluate_bundle(
        baseline, holdout_rows, answer_fn, scorer, "holdout", num_threads
    )
    floors = goal.resolve_floors(baseline_report.aggregates)

    current, current_report = baseline, baseline_report
    rounds: t.List[RoundResult] = []
    stop_reason = f"reached max_rounds ({config.max_rounds})"

    if current_report.aggregates[target] >= goal.target_value:
        stop_reason = "target already reached by the baseline"
    else:
        for round_no in range(1, config.max_rounds + 1):
            spend_start = budget.total_usd if budget else 0.0
            try:
                candidate = optimize_fn(current, round_no)
                if budget is not None:
                    budget.stage = "loop:holdout"
                if candidate.fingerprint() == current.fingerprint():
                    result = RoundResult(
                        round_no,
                        UNCHANGED_PROMPT,
                        False,
                        current_report,
                        None,
                        candidate,
                    )
                else:
                    after = evaluate_bundle(
                        candidate,
                        holdout_rows,
                        answer_fn,
                        scorer,
                        "holdout",
                        num_threads,
                    )
                    deltas = _paired_deltas(current_report, after, target)
                    mean_delta = round(statistics.fmean(deltas), 6) if deltas else 0.0
                    ci = paired_bootstrap_ci(
                        deltas,
                        config.confidence,
                        config.n_bootstrap,
                        config.seed + round_no,
                    )
                    floor_check = check_floors(after.aggregates, floors)
                    if not floor_check.passed:
                        verdict, accepted = FLOOR_VIOLATION, False
                    elif ci[0] > 0 and mean_delta >= config.min_gain:
                        verdict, accepted = IMPROVED, True
                    elif ci[1] < 0:
                        verdict, accepted = REGRESSED, False
                    else:
                        verdict, accepted = NO_CHANGE, False
                    result = RoundResult(
                        round_no,
                        verdict,
                        accepted,
                        current_report,
                        after,
                        candidate,
                        mean_delta=mean_delta,
                        ci=ci,
                        floor_check=floor_check,
                    )
            except BudgetExceeded as e:
                stop_reason = f"budget exhausted: {e}"
                break
            result.spend_usd = (budget.total_usd if budget else 0.0) - spend_start
            rounds.append(result)
            if on_round is not None:
                on_round(result)

            if not result.accepted:
                stop_reason = f"round {round_no}: {result.verdict}"
                break
            assert result.after is not None and result.candidate is not None
            current, current_report = result.candidate, result.after
            if current_report.aggregates[target] >= goal.target_value:
                stop_reason = f"target {goal.target_value} reached in round {round_no}"
                break

    deltas = _paired_deltas(baseline_report, current_report, target)
    cumulative = round(statistics.fmean(deltas), 6) if deltas else 0.0
    return LoopReport(
        goal=goal,
        baseline=baseline_report,
        final_bundle=current,
        final=current_report,
        rounds=rounds,
        stop_reason=stop_reason,
        cumulative_delta=cumulative,
        cumulative_ci=paired_bootstrap_ci(
            deltas, config.confidence, config.n_bootstrap, config.seed
        ),
        spend=budget.summary() if budget else {},
    )
