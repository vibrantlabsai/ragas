"""Shared test data and stubs for the prompt_optimization tests (no network)."""

from __future__ import annotations

import typing as t
from pathlib import Path

from ragas.optimizers.prompt_optimization.bundle import GoldenRow, PromptBundle

try:
    import dspy  # noqa: F401

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

REFUSAL = "I don't know based on the provided documents."
REPO_ROOT = Path(__file__).resolve().parents[4]
TEMPLATE_DIR = REPO_ROOT / "examples" / "ragas_examples" / "prompt_optimization"
GOLDEN_DATASET = TEMPLATE_DIR / "datasets" / "hr_policy_qa_golden.jsonl"
BASELINE_PROMPT = TEMPLATE_DIR / "prompts" / "baseline_system_prompt.md"
GOAL_FILE = TEMPLATE_DIR / "config" / "optimization_goal.json"

_FACTS = [
    (
        "How many sick days do employees get?",
        "8 sick days per year",
        "Sick leave is tracked separately from PTO: employees receive 8 sick days per year.",
    ),
    (
        "How often are laptops replaced?",
        "Every 3 years",
        "Company laptops are replaced every 3 years.",
    ),
    (
        "Who approves fully remote roles?",
        "A VP",
        "Fully remote roles must be approved by a VP.",
    ),
    (
        "What is the referral bonus?",
        "$2,000",
        "Employees receive a $2,000 referral bonus.",
    ),
    (
        "How long is probation?",
        "90 days",
        "New hires complete a 90-day probation period.",
    ),
    (
        "When are salaries paid?",
        "On the last working day of each month",
        "Salaries are paid on the last working day of each month.",
    ),
    (
        "How many public holidays are there?",
        "11 per year",
        "Orbitline observes 11 paid public holidays per year.",
    ),
    (
        "What is the learning budget?",
        "$1,200 per year",
        "Each employee has an annual learning budget of $1,200.",
    ),
    (
        "How long must passwords be?",
        "At least 14 characters",
        "Passwords must be at least 14 characters long.",
    ),
    ("How much is the replacement badge fee?", "$25", "A replacement badge costs $25."),
    (
        "How much notice should employees give?",
        "4 weeks",
        "Employees who resign are asked to give 4 weeks of notice.",
    ),
    (
        "How many volunteer days are paid?",
        "2 per year",
        "Employees get 2 paid volunteer days per year.",
    ),
]
_UNANSWERABLE = [
    (
        "Does Orbitline reimburse gym memberships?",
        "The wellness program offers monthly webinars.",
    ),
    ("When do stock options vest?", "401(k) matching contributions vest immediately."),
    ("Is there a company car program?", "On-call roles get a mobile phone allowance."),
    (
        "How much is the childcare subsidy?",
        "Primary caregivers receive 16 weeks of parental leave.",
    ),
]
DISTRACTOR = "The office is open from 7am to 8pm on weekdays."


def make_rows() -> t.List[GoldenRow]:
    """16 rows: 4 per split, one unanswerable in each split."""
    rows: t.List[GoldenRow] = []
    splits = ["train", "eval", "test", "holdout"]
    for i, (q, a, ctx) in enumerate(_FACTS):
        rows.append(
            GoldenRow(f"f{i:02d}", q, (ctx, DISTRACTOR), a, splits[i % 4], True)
        )
    for i, (q, ctx) in enumerate(_UNANSWERABLE):
        rows.append(
            GoldenRow(f"u{i:02d}", q, (ctx, DISTRACTOR), REFUSAL, splits[i % 4], False)
        )
    return rows


class StubAnswerer:
    """Prompt-sensitive answer function that needs no DSPy and no model.

    - "concise" in the instruction, or any demo  -> exact reference
    - otherwise                                  -> verbose answer containing the reference
    - unanswerable rows refuse only if the instruction says "don't know" or a
      demo shows a refusal; otherwise they hallucinate.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, bundle: PromptBundle, row: GoldenRow) -> str:
        self.calls += 1
        text = bundle.instruction.lower()
        concise = "concise" in text or bool(bundle.few_shots)
        refuses = "don't know" in text or any(
            REFUSAL in s.answer for s in bundle.few_shots
        )
        if not row.answerable:
            return REFUSAL if refuses else "Yes, that benefit is available to everyone."
        if concise:
            return row.reference
        return (
            f"Great question! {row.reference}. Happy to help with anything else today."
        )
