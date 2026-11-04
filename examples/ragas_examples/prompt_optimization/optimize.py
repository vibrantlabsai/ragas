"""prompt_optimization: improve a RAG app's prompt and prove it with ragas evals.

Flow:
    1. Ragas eval of your current (baseline) prompt on the holdout split.
    2. Optimize: Layer 1 few-shot search (golden demos, instruction frozen),
       Layer 2 MIPROv2 instruction search (demos frozen), benchmark on test.
       Repeat while the holdout gain is significant and floors hold.
    3. Ragas eval of the optimized prompt on the same holdout split.
    4. Before/after comparison, the final prompt with cache-breakpoint
       markers, and the spend.

Examples:
    ragas-prompt-optimization --dry-run          # offline simulation, $0
    ragas-prompt-optimization --smoke            # 2 live calls, about $0.01
    ragas-prompt-optimization --max-usd 5        # full live run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import typing as t
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_NAME = "hr_policy_qa_golden.jsonl"


def default_dataset(base: Path = HERE) -> Path:
    """The shipped dataset: ``datasets/`` in the repo, ``evals/datasets/`` after quickstart."""
    for candidate in (
        base / "datasets" / DATASET_NAME,
        base / "evals" / "datasets" / DATASET_NAME,
    ):
        if candidate.exists():
            return candidate
    return base / "datasets" / DATASET_NAME


DEFAULT_DATASET = default_dataset()
DEFAULT_PROMPT = HERE / "prompts" / "baseline_system_prompt.md"
DEFAULT_GOAL = HERE / "config" / "optimization_goal.json"


ENV_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def load_env_file(*search_dirs: Path) -> t.Optional[Path]:
    """Load Anthropic credentials from the first ``.env`` found, without overriding.

    Only ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` are read, and a value
    already set in the environment always wins. Values are never printed.
    """
    dirs = search_dirs or (Path.cwd(), HERE)
    for directory in dirs:
        env_path = directory / ".env"
        if not env_path.is_file():
            continue
        for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            key, sep, value = line.partition("=")
            key = key.strip()
            if not sep or key not in ENV_KEYS or os.environ.get(key):
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value:
                os.environ[key] = value
        return env_path
    return None


def _silence_progress_bars() -> None:
    """Hide DSPy's per-candidate progress bars (MIPROv2 forces them on).

    DSPy passes ``disable=`` explicitly, so ``TQDM_DISABLE`` has no effect;
    this CLI-only patch forces ``disable=True`` for this process.
    """
    try:
        import tqdm
    except ImportError:
        return
    original = tqdm.tqdm.__init__
    if getattr(original, "_ragas_quiet", False):
        return

    def quiet_init(self: t.Any, *args: t.Any, **kwargs: t.Any) -> None:
        kwargs["disable"] = True
        original(self, *args, **kwargs)

    quiet_init._ragas_quiet = True  # type: ignore[attr-defined]
    tqdm.tqdm.__init__ = quiet_init  # type: ignore[method-assign]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ragas-prompt-optimization",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="golden JSONL/JSON with user_input, retrieved_contexts, reference, split",
    )
    p.add_argument(
        "--system-prompt",
        type=Path,
        default=DEFAULT_PROMPT,
        help="file with the current system prompt",
    )
    p.add_argument(
        "--goal",
        type=Path,
        default=DEFAULT_GOAL,
        help="JSON with target_metric, target_value, floors, floor_tolerance",
    )
    p.add_argument(
        "--student", default="claude-haiku-5-5", help="model your RAG app runs"
    )
    p.add_argument(
        "--proposer", default="claude-sonnet-5-5", help="model that writes instructions"
    )
    p.add_argument(
        "--effort", default="low", help="Claude effort level for both models"
    )
    p.add_argument("--rounds", type=int, default=3, help="maximum improvement rounds")
    p.add_argument(
        "--min-gain",
        type=float,
        default=0.02,
        help="minimum mean holdout gain to accept a round",
    )
    p.add_argument("--demo-sets", type=int, default=16, help="Layer 1 random demo sets")
    p.add_argument(
        "--max-demos", type=int, default=4, help="Layer 1 maximum demos per set"
    )
    p.add_argument(
        "--instructions", type=int, default=8, help="Layer 2 instruction candidates"
    )
    p.add_argument("--trials", type=int, default=20, help="Layer 2 MIPROv2 trials")
    p.add_argument(
        "--minibatch-size", type=int, default=10, help="Layer 2 minibatch size on eval"
    )
    p.add_argument("--threads", type=int, default=8, help="parallel model calls")
    p.add_argument("--max-usd", type=float, default=5.0, help="hard spending cap")
    p.add_argument(
        "--out-dir", type=Path, default=None, help="default: runs/<timestamp>"
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="simulate the models (no network, no cost) to check the whole flow",
    )
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="make one student and one proposer call to check credentials and caching",
    )
    p.add_argument(
        "--simulated-behavior",
        default="sensitive",
        choices=["sensitive", "insensitive", "recall_dropping"],
        help="dry-run model behaviour (insensitive/recall_dropping are controls)",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="show DSPy progress logs"
    )
    return p


def run_smoke(args: argparse.Namespace, out_dir: Path) -> int:
    """Two identical student calls (checks caching) and one proposer call."""
    from ragas.optimizers.prompt_optimization import (
        BudgetConfig,
        BudgetTracker,
        ModelConfig,
        PromptBundle,
        load_golden_rows,
    )
    from ragas.optimizers.prompt_optimization.bundle import FewShot
    from ragas.optimizers.prompt_optimization.runtime import build_live_runtime

    rows = load_golden_rows(args.dataset)
    train = [r for r in rows if r.split == "train"]
    budget = BudgetTracker(BudgetConfig(max_usd=min(args.max_usd, 0.25)))
    models = ModelConfig(
        student=args.student, proposer=args.proposer, effort=args.effort
    )
    runtime = build_live_runtime(models, budget, cache=False)
    # Long enough instruction + demos to pass the 512-token caching minimum.
    bundle = PromptBundle(
        instruction=args.system_prompt.read_text(encoding="utf-8").strip(),
        few_shots=[FewShot.from_row(r) for r in train[:8]],
        source="smoke",
    )
    question = next(r for r in rows if r.split == "holdout")
    results = []
    for attempt in (1, 2):
        answer = runtime.answer_fn(bundle, question)
        results.append({"attempt": attempt, "answer": answer})
    proposer_reply = runtime.proposer_lm("Reply with the single word: ready")
    usage = budget.summary()
    report = {
        "student_model": args.student,
        "proposer_model": args.proposer,
        "question": question.question,
        "reference": question.reference,
        "student_answers": results,
        "proposer_reply": proposer_reply,
        "cache_read_tokens_total": usage["tokens"].get("cache_read_input_tokens", 0),
        "spend": usage,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke_usage.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))
    ok = report["cache_read_tokens_total"] > 0
    print(
        f"\nPrompt cache {'HIT' if ok else 'MISS'} on the repeated call. Spend: ${usage['total_usd']:.4f}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    if not args.verbose:
        _silence_progress_bars()
        for name in ("dspy", "optuna", "httpx", "anthropic"):
            logging.getLogger(name).setLevel(logging.ERROR)
        try:  # DSPy installs its own handler; switch it off unless --verbose
            import dspy

            dspy.disable_logging()
        except ImportError:
            pass
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

    out_dir = args.out_dir or HERE / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")

    if not args.dry_run:
        load_env_file()
    if not (args.dry_run or args.smoke) and not (
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    ):
        print(
            "Note: ANTHROPIC_API_KEY is not set; relying on an `ant auth login` profile if present.",
            file=sys.stderr,
        )

    if args.smoke:
        return run_smoke(args, out_dir)

    from ragas.optimizers.prompt_optimization import (
        BudgetConfig,
        FewShotSearchConfig,
        InstructionSearchConfig,
        LoopConfig,
        ModelConfig,
        OptimizationGoal,
        run_pipeline,
    )

    result = run_pipeline(
        dataset=args.dataset,
        system_prompt=args.system_prompt.read_text(encoding="utf-8"),
        goal=OptimizationGoal.from_json(args.goal),
        out_dir=out_dir,
        models=ModelConfig(
            student=args.student, proposer=args.proposer, effort=args.effort
        ),
        budget_config=BudgetConfig(max_usd=args.max_usd),
        loop_config=LoopConfig(max_rounds=args.rounds, min_gain=args.min_gain),
        few_shot_config=FewShotSearchConfig(
            num_candidate_sets=args.demo_sets, max_demos=args.max_demos
        ),
        instruction_config=InstructionSearchConfig(
            num_candidates=args.instructions,
            num_trials=args.trials,
            minibatch_size=args.minibatch_size,
        ),
        mode="simulated" if args.dry_run else "live",
        simulated_behavior=args.simulated_behavior,
        num_threads=args.threads,
    )

    print()
    print(result.comparison.to_markdown())
    if result.loop:
        print(result.loop.to_markdown())
    print(f"Stop reason: {result.stop_reason}")
    print(f"Final prompt: {out_dir / 'final' / 'final_prompt.md'}")
    if args.dry_run:
        print(
            f"Simulated spend (dry run, nothing billed): ${result.spend['total_usd']:.4f} "
            f"for {result.spend['calls']} simulated calls at live prices"
        )
    else:
        print(
            f"Spend: ${result.spend['total_usd']:.4f} of ${result.spend['max_usd']:.2f} "
            f"({result.spend['calls']} calls)"
        )
    print(f"All outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
