"""Paired, coverage-aware comparison of saved evaluation observations."""

from .collection import ComparisonAdapter, plan_for_dataset
from .config import (
    ComparisonLimits,
    ComparisonPlan,
    ExpectedCase,
    MetricSpec,
    SliceSpec,
)
from .engine import compare
from .evaluation import (
    EvaluationKey,
    compare_evaluations,
    evaluation_rows,
    plan_for_evaluation_dataset,
)
from .export import write_observations
from .reports import ComparisonReport

__all__ = [
    "ComparisonLimits",
    "ComparisonPlan",
    "ExpectedCase",
    "MetricSpec",
    "SliceSpec",
    "ComparisonReport",
    "compare",
    "ComparisonAdapter",
    "plan_for_dataset",
    "EvaluationKey",
    "compare_evaluations",
    "evaluation_rows",
    "plan_for_evaluation_dataset",
    "write_observations",
]
