"""End-to-end flow: ragas eval (before) -> optimize -> ragas eval (after) -> compare."""

from __future__ import annotations

import json
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from ragas.optimizers.prompt_optimization.budget import BudgetExceeded, BudgetTracker
from ragas.optimizers.prompt_optimization.bundle import PromptBundle
from ragas.optimizers.prompt_optimization.config import (
    BudgetConfig,
    FewShotSearchConfig,
    InstructionSearchConfig,
    LoopConfig,
    ModelConfig,
    OptimizationGoal,
)
from ragas.optimizers.prompt_optimization.dataset import load_golden_rows, split_rows
from ragas.optimizers.prompt_optimization.export import export_bundle
from ragas.optimizers.prompt_optimization.improvement_loop import (
    LoopReport,
    paired_bootstrap_ci,
    run_improvement_loop,
)
from ragas.optimizers.prompt_optimization.instruction_search import (
    MIPROv2InstructionProposer,
)
from ragas.optimizers.prompt_optimization.metrics import Scorer
from ragas.optimizers.prompt_optimization.optimizer import RAGPromptOptimizer
from ragas.optimizers.prompt_optimization.rag_evaluation import EvaluationReport
from ragas.optimizers.prompt_optimization.ragas_eval import run_ragas_eval
from ragas.optimizers.prompt_optimization.runtime import (
    Runtime,
    build_live_runtime,
    build_simulated_runtime,
)

#: Metrics always reported in the before/after comparison.
REPORT_METRICS = (
    "token_f1",
    "token_precision",
    "token_recall",
    "token_f0_5",
    "exact_match",
)


@dataclass
class Comparison:
    """Before vs after on the holdout split, measured with the ragas eval."""

    split: str
    target_metric: str
    before: EvaluationReport
    after: EvaluationReport
    target_ci: t.Tuple[float, float]
    rows: t.List[t.Dict[str, t.Any]] = field(default_factory=list)

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            "split": self.split,
            "target_metric": self.target_metric,
            "before_source": self.before.source,
            "after_source": self.after.source,
            "target_ci_95": list(self.target_ci),
            "metrics": self.rows,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Before vs after (ragas eval, `{self.split}` split, {len(self.before.rows)} rows)",
            "",
            f"Before: `{self.before.source}` ({self.before.fingerprint})  ",
            f"After: `{self.after.source}` ({self.after.fingerprint})",
            "",
            "| Metric | Before | After | Change |",
            "|---|---|---|---|",
        ]
        for r in self.rows:
            mark = " (target)" if r["metric"] == self.target_metric else ""
            lines.append(
                f"| {r['metric']}{mark} | {r['before']:.4f} | {r['after']:.4f} | {r['delta']:+.4f} |"
            )
        lines += [
            "",
            f"Paired 95% bootstrap CI of the `{self.target_metric}` change: "
            f"[{self.target_ci[0]:+.4f}, {self.target_ci[1]:+.4f}]",
        ]
        return "\n".join(lines) + "\n"


def compare(
    before: EvaluationReport,
    after: EvaluationReport,
    target_metric: str,
    loop_config: LoopConfig,
) -> Comparison:
    rows = []
    for metric in before.aggregates:
        b, a = before.aggregates[metric], after.aggregates[metric]
        rows.append(
            {"metric": metric, "before": b, "after": a, "delta": round(a - b, 6)}
        )
    b_rows = before.scores_by_row(target_metric)
    a_rows = after.scores_by_row(target_metric)
    deltas = [a_rows[k] - b_rows[k] for k in b_rows]
    ci = paired_bootstrap_ci(
        deltas, loop_config.confidence, loop_config.n_bootstrap, loop_config.seed
    )
    return Comparison(before.split, target_metric, before, after, ci, rows)


@dataclass
class PipelineResult:
    before: EvaluationReport
    after: EvaluationReport
    comparison: Comparison
    loop: t.Optional[LoopReport]
    final_bundle: PromptBundle
    spend: t.Dict[str, t.Any]
    out_dir: Path
    stop_reason: str = ""


def run_pipeline(
    dataset: t.Union[str, Path],
    system_prompt: str,
    goal: OptimizationGoal,
    out_dir: t.Union[str, Path],
    models: t.Optional[ModelConfig] = None,
    budget_config: t.Optional[BudgetConfig] = None,
    loop_config: t.Optional[LoopConfig] = None,
    few_shot_config: t.Optional[FewShotSearchConfig] = None,
    instruction_config: t.Optional[InstructionSearchConfig] = None,
    mode: str = "live",
    simulated_behavior: str = "sensitive",
    num_threads: int = 8,
    runtime: t.Optional[Runtime] = None,
    skip_layer2: bool = False,
) -> PipelineResult:
    """Run the whole tool.

    1. Ragas eval of the baseline prompt on holdout ("before").
    2. Improvement loop: optimize (Layer 1 -> Layer 2 -> test benchmark) and
       accept a round only if holdout improves significantly within floors.
    3. Ragas eval of the final prompt on holdout ("after").
    4. Before/after comparison, final prompt export and spend summary.

    ``mode="simulated"`` swaps the Anthropic API for a deterministic simulation
    (no network, no cost) so the full flow can be checked offline.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = models or ModelConfig()
    loop_config = loop_config or LoopConfig()
    rows = load_golden_rows(dataset)
    splits = split_rows(rows)
    budget = (
        runtime.budget if runtime else BudgetTracker(budget_config or BudgetConfig())
    )

    if runtime is None:
        if mode == "live":
            runtime = build_live_runtime(models, budget)
        elif mode == "simulated":
            runtime = build_simulated_runtime(rows, models, budget, simulated_behavior)
        else:
            raise ValueError("mode must be 'live' or 'simulated'")

    metric_names = list(dict.fromkeys([*REPORT_METRICS, *goal.metrics]))
    scorer = Scorer(metric_names)
    baseline = PromptBundle(instruction=system_prompt.strip(), source="baseline")

    # 1. Before ----------------------------------------------------------
    budget.stage = "ragas_eval:before"
    before, before_csv = run_ragas_eval(
        baseline,
        splits["holdout"],
        runtime.answer_fn,
        scorer,
        out / "ragas_eval",
        "before",
        "holdout",
    )

    # 2. Optimize with the improvement loop ---------------------------------
    proposer = None
    if not skip_layer2:
        proposer = MIPROv2InstructionProposer(
            runtime.student_lm,
            runtime.proposer_lm,
            instruction_config,
            num_threads=num_threads,
        )
    optimizer = RAGPromptOptimizer(
        runtime.answer_fn,
        scorer,
        goal,
        rows,
        proposer=proposer,
        few_shot_config=few_shot_config,
        budget=budget,
        num_threads=num_threads,
        export_model=models.student,
    )

    def optimize_fn(current: PromptBundle, round_no: int) -> PromptBundle:
        return optimizer.optimize(
            current, splits, reference=baseline, round_no=round_no, out_dir=out
        ).winner

    loop_report: t.Optional[LoopReport] = None
    stop_reason = ""
    try:
        loop_report = run_improvement_loop(
            baseline,
            optimize_fn,
            splits["holdout"],
            runtime.answer_fn,
            scorer,
            goal,
            loop_config,
            budget=budget,
            num_threads=num_threads,
        )
        loop_report.save(out)
        final_bundle = loop_report.final_bundle
        stop_reason = loop_report.stop_reason
    except BudgetExceeded as e:
        final_bundle, stop_reason = (
            baseline,
            f"budget exhausted before the loop finished: {e}",
        )

    # 3. After -----------------------------------------------------------
    budget.stage = "ragas_eval:after"
    try:
        after, after_csv = run_ragas_eval(
            final_bundle,
            splits["holdout"],
            runtime.answer_fn,
            scorer,
            out / "ragas_eval",
            "after",
            "holdout",
        )
    except BudgetExceeded:
        after, after_csv = before, before_csv

    # 4. Compare, export, save ---------------------------------------------
    comparison = compare(before, after, goal.target_metric, loop_config)
    (out / "comparison.md").write_text(comparison.to_markdown(), encoding="utf-8")
    (out / "comparison.json").write_text(
        json.dumps(comparison.to_dict(), indent=2), encoding="utf-8"
    )
    export_bundle(
        final_bundle,
        out / "final",
        model=models.student,
        scores={"holdout_before": before.aggregates, "holdout_after": after.aggregates},
    )
    spend = budget.summary()
    (out / "spend.json").write_text(json.dumps(spend, indent=2), encoding="utf-8")
    (out / "run_info.json").write_text(
        json.dumps(
            {
                "mode": runtime.mode,
                "models": {
                    "student": models.student,
                    "proposer": models.proposer,
                    "effort": models.effort,
                },
                "goal": goal.to_dict(),
                "stop_reason": stop_reason,
                "before_csv": str(before_csv),
                "after_csv": str(after_csv),
                "student_calls": runtime.student_engine.calls,
                "proposer_calls": runtime.proposer_engine.calls,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return PipelineResult(
        before=before,
        after=after,
        comparison=comparison,
        loop=loop_report,
        final_bundle=final_bundle,
        spend=spend,
        out_dir=out,
        stop_reason=stop_reason,
    )
