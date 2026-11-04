"""Build the student/proposer models, live (Anthropic) or simulated (dry run)."""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

from ragas.optimizers.prompt_optimization.anthropic_engine import (
    RagasAnthropicEngine,
    make_dspy_lm,
)
from ragas.optimizers.prompt_optimization.budget import BudgetTracker
from ragas.optimizers.prompt_optimization.bundle import GoldenRow
from ragas.optimizers.prompt_optimization.config import ModelConfig
from ragas.optimizers.prompt_optimization.signature import DSPyAnswerer


@dataclass
class Runtime:
    """Everything the optimizer needs to call models."""

    mode: str
    student_engine: RagasAnthropicEngine
    proposer_engine: RagasAnthropicEngine
    student_lm: t.Any
    proposer_lm: t.Any
    answer_fn: DSPyAnswerer
    budget: BudgetTracker


def build_live_runtime(
    models: ModelConfig,
    budget: BudgetTracker,
    client: t.Any = None,
    cache: bool = True,
) -> Runtime:
    """Claude via the ragas wrappers: ``llm_factory(provider="anthropic")``.

    Reads credentials the usual Anthropic way (``ANTHROPIC_API_KEY`` or an
    ``ant auth login`` profile) unless ``client`` is given.
    """
    import anthropic

    from ragas.llms import llm_factory

    client = client or anthropic.Anthropic()
    student_llm = llm_factory(
        models.student,
        provider="anthropic",
        client=client,
        max_tokens=models.max_tokens,
    )
    proposer_llm = llm_factory(
        models.proposer,
        provider="anthropic",
        client=client,
        max_tokens=models.max_tokens,
    )
    student_engine = RagasAnthropicEngine(
        student_llm, budget=budget, effort=models.effort, max_tokens=models.max_tokens
    )
    proposer_engine = RagasAnthropicEngine(
        proposer_llm, budget=budget, effort=models.effort, max_tokens=models.max_tokens
    )
    student_lm = make_dspy_lm(student_engine, cache=cache)
    proposer_lm = make_dspy_lm(proposer_engine, cache=cache)
    return Runtime(
        mode="live",
        student_engine=student_engine,
        proposer_engine=proposer_engine,
        student_lm=student_lm,
        proposer_lm=proposer_lm,
        answer_fn=DSPyAnswerer(student_lm),
        budget=budget,
    )


def build_simulated_runtime(
    rows: t.Sequence[GoldenRow],
    models: ModelConfig,
    budget: BudgetTracker,
    behavior: str = "sensitive",
    instructions: t.Optional[t.Sequence[str]] = None,
) -> Runtime:
    """No network, no cost: the real engine talks to a simulated Anthropic client."""
    from ragas.optimizers.prompt_optimization.simulated import (
        DEFAULT_INSTRUCTION_CANDIDATES,
        SimulatedAnthropicClient,
        SimulatedProposer,
        SimulatedStudent,
    )

    student_engine = RagasAnthropicEngine(
        client=SimulatedAnthropicClient(SimulatedStudent(rows, behavior)),
        model=models.student,
        budget=budget,
        effort=models.effort,
        max_tokens=models.max_tokens,
    )
    proposer_engine = RagasAnthropicEngine(
        client=SimulatedAnthropicClient(
            SimulatedProposer(instructions or DEFAULT_INSTRUCTION_CANDIDATES)
        ),
        model=models.proposer,
        budget=budget,
        effort=models.effort,
        max_tokens=models.max_tokens,
    )
    # No DSPy disk cache: simulated answers must never leak into live runs.
    student_lm = make_dspy_lm(student_engine, cache=False, num_retries=0)
    proposer_lm = make_dspy_lm(proposer_engine, cache=False, num_retries=0)
    return Runtime(
        mode=f"simulated:{behavior}",
        student_engine=student_engine,
        proposer_engine=proposer_engine,
        student_lm=student_lm,
        proposer_lm=proposer_lm,
        answer_fn=DSPyAnswerer(student_lm),
        budget=budget,
    )
