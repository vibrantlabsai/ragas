from ragas.optimizers.base import Optimizer
from ragas.optimizers.genetic import GeneticOptimizer

try:
    from ragas.optimizers.dspy_optimizer import DSPyOptimizer

    __all__ = [
        "Optimizer",
        "GeneticOptimizer",
        "DSPyOptimizer",
    ]
except ImportError:
    __all__ = [
        "Optimizer",
        "GeneticOptimizer",
    ]
