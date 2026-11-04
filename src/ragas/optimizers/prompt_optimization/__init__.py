"""Optimize a RAG app's own prompt with DSPy, verified with ragas evaluations.

Flow (see :func:`run_pipeline`):

1. Ragas eval of the baseline prompt on the holdout split ("before").
2. Improvement loop. Each round:
   - Layer 1: golden random search for few-shot demos, instruction frozen.
   - Layer 2: MIPROv2 instruction search, demos frozen.
   - Benchmark of every candidate on the test split, with the user's floors.
   - Accept the round only if holdout improves significantly within floors.
3. Ragas eval of the final prompt on holdout ("after"), and a comparison.

This package is separate from the judge-prompt optimizers in
``ragas.optimizers`` (``DSPyOptimizer``, ``GeneticOptimizer``), which tune
ragas's own metric prompts.
"""

from ragas.optimizers.prompt_optimization.budget import BudgetExceeded, BudgetTracker
from ragas.optimizers.prompt_optimization.bundle import FewShot, GoldenRow, PromptBundle
from ragas.optimizers.prompt_optimization.config import (
    BudgetConfig,
    FewShotSearchConfig,
    InstructionSearchConfig,
    LoopConfig,
    ModelConfig,
    OptimizationGoal,
)
from ragas.optimizers.prompt_optimization.dataset import load_golden_rows, split_rows
from ragas.optimizers.prompt_optimization.export import (
    export_bundle,
    load_prompt_bundle,
)
from ragas.optimizers.prompt_optimization.improvement_loop import run_improvement_loop
from ragas.optimizers.prompt_optimization.optimizer import RAGPromptOptimizer
from ragas.optimizers.prompt_optimization.pipeline import run_pipeline

__all__ = [
    "BudgetConfig",
    "BudgetExceeded",
    "BudgetTracker",
    "FewShot",
    "FewShotSearchConfig",
    "GoldenRow",
    "InstructionSearchConfig",
    "LoopConfig",
    "ModelConfig",
    "OptimizationGoal",
    "PromptBundle",
    "RAGPromptOptimizer",
    "export_bundle",
    "load_golden_rows",
    "load_prompt_bundle",
    "run_improvement_loop",
    "run_pipeline",
    "split_rows",
]
