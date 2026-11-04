"""Ragas eval of one prompt on one split (standalone, like rag_eval).

Use it to score any prompt, including an exported ``final_prompt.json``:

    ragas-prompt-optimization-eval --prompt prompts/baseline_system_prompt.md
    ragas-prompt-optimization-eval --prompt runs/<ts>/final/final_prompt.json --split test
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ragas-prompt-optimization-eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dataset = next(
        (
            c
            for c in (
                HERE / "datasets" / "hr_policy_qa_golden.jsonl",
                HERE / "evals" / "datasets" / "hr_policy_qa_golden.jsonl",
            )
            if c.exists()
        ),
        HERE / "datasets" / "hr_policy_qa_golden.jsonl",
    )
    p.add_argument("--dataset", type=Path, default=dataset)
    p.add_argument(
        "--prompt",
        type=Path,
        default=HERE / "prompts" / "baseline_system_prompt.md",
        help="a .md/.txt system prompt, or an exported final_prompt.json/.md",
    )
    p.add_argument(
        "--split", default="holdout", choices=["train", "eval", "test", "holdout"]
    )
    p.add_argument("--student", default="claude-haiku-5-5")
    p.add_argument("--effort", default="low")
    p.add_argument("--max-usd", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true", help="simulated model, no network")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

    from ragas.optimizers.prompt_optimization import (
        BudgetConfig,
        BudgetTracker,
        ModelConfig,
        PromptBundle,
        load_golden_rows,
        load_prompt_bundle,
        split_rows,
    )
    from ragas.optimizers.prompt_optimization.metrics import Scorer
    from ragas.optimizers.prompt_optimization.pipeline import REPORT_METRICS
    from ragas.optimizers.prompt_optimization.ragas_eval import run_ragas_eval
    from ragas.optimizers.prompt_optimization.runtime import (
        build_live_runtime,
        build_simulated_runtime,
    )

    rows = load_golden_rows(args.dataset)
    split = split_rows(rows, required=[args.split])[args.split]
    text = args.prompt.read_text(encoding="utf-8")
    if "RAGAS:INSTRUCTION:START" in text or args.prompt.suffix == ".json":
        bundle = load_prompt_bundle(args.prompt)
    else:
        bundle = PromptBundle(instruction=text.strip(), source=args.prompt.stem)

    budget = BudgetTracker(BudgetConfig(max_usd=args.max_usd))
    models = ModelConfig(student=args.student, effort=args.effort)
    runtime = (
        build_simulated_runtime(rows, models, budget)
        if args.dry_run
        else build_live_runtime(models, budget)
    )
    out_dir = args.out_dir or HERE / "runs" / f"eval-{datetime.now():%Y%m%d-%H%M%S}"
    report, csv_path = run_ragas_eval(
        bundle,
        split,
        runtime.answer_fn,
        Scorer(list(REPORT_METRICS)),
        out_dir,
        "eval",
        args.split,
    )
    print(f"Ragas eval of `{bundle.source}` on {args.split} ({len(split)} rows):")
    for metric, value in report.aggregates.items():
        print(f"  {metric:<16} {value:.4f}")
    print(f"Rows: {csv_path}")
    print(f"Spend: ${budget.total_usd:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
