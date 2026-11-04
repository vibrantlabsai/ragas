"""Configuration objects for RAG prompt optimization."""

from __future__ import annotations

import json
import typing as t
from dataclasses import asdict, dataclass, field
from pathlib import Path

FloorValue = t.Union[float, t.Literal["baseline"]]

#: Metrics the optimizer can compute without an LLM judge.
NON_LLM_METRICS = (
    "token_precision",
    "token_recall",
    "token_f1",
    "token_f0_5",
    "exact_match",
)


@dataclass
class OptimizationGoal:
    """What to improve and what must not regress.

    Attributes:
        target_metric: The metric the optimizer maximizes, e.g. ``"token_f1"``.
        target_value: Stop early once this value is reached (0-1 scale).
        floors: Minimum values for metrics the user cares about. Only metrics
            listed here are constrained. A value is either an absolute number
            or ``"baseline"``, meaning "no worse than the starting prompt".
        floor_tolerance: Slack subtracted from ``"baseline"`` floors, so tiny
            noise does not reject a candidate.
    """

    target_metric: str = "token_f1"
    target_value: float = 1.0
    floors: t.Dict[str, FloorValue] = field(default_factory=dict)
    floor_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.target_value <= 1.0:
            raise ValueError("target_value must be between 0 and 1")
        for metric, floor in self.floors.items():
            if floor != "baseline" and not isinstance(floor, (int, float)):
                raise ValueError(
                    f"floor for {metric!r} must be a number or 'baseline'; got {floor!r}"
                )

    @property
    def metrics(self) -> t.List[str]:
        """Every metric that must be computed: the target plus floored metrics."""
        names = [self.target_metric]
        names.extend(m for m in self.floors if m != self.target_metric)
        return names

    def resolve_floors(
        self, baseline: t.Optional[t.Mapping[str, float]] = None
    ) -> t.Dict[str, float]:
        """Turn ``"baseline"`` floors into numbers using baseline aggregates."""
        resolved: t.Dict[str, float] = {}
        for metric, floor in self.floors.items():
            if floor == "baseline":
                if baseline is None or metric not in baseline:
                    raise ValueError(
                        f"floor for {metric!r} is 'baseline' but no baseline score "
                        "was provided"
                    )
                resolved[metric] = round(baseline[metric] - self.floor_tolerance, 6)
            else:
                resolved[metric] = float(floor)
        return resolved

    @classmethod
    def from_json(cls, path: t.Union[str, Path]) -> "OptimizationGoal":
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(
            target_metric=data.get("target_metric", "token_f1"),
            target_value=float(data.get("target_value", 1.0)),
            floors=data.get("floors", {}),
            floor_tolerance=float(data.get("floor_tolerance", 0.0)),
        )

    def to_dict(self) -> t.Dict[str, t.Any]:
        return asdict(self)


@dataclass
class FewShotSearchConfig:
    """Layer 1: golden random search over few-shot demo sets.

    Mirrors the seed scheme of DSPy's BootstrapFewShotWithRandomSearch:
    seed -3 is zero-shot, seed -2 is the first ``max_demos`` training rows,
    and seeds ``0..num_candidate_sets-1`` shuffle the training rows and take a
    random size between 1 and ``max_demos``.
    """

    num_candidate_sets: int = 16
    max_demos: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_demos < 1:
            raise ValueError("max_demos must be at least 1")
        if self.num_candidate_sets < 0:
            raise ValueError("num_candidate_sets must be non-negative")


@dataclass
class InstructionSearchConfig:
    """Layer 2: MIPROv2 instruction-only search with the Layer 1 demos fixed."""

    num_candidates: int = 8
    num_trials: int = 20
    minibatch_size: int = 10
    seed: int = 9


@dataclass
class ModelPrice:
    """Price per million tokens, in US dollars."""

    input: float
    output: float
    cache_read: t.Optional[float] = None
    cache_write: t.Optional[float] = None

    def cache_read_price(self) -> float:
        return self.cache_read if self.cache_read is not None else self.input * 0.1

    def cache_write_price(self) -> float:
        return self.cache_write if self.cache_write is not None else self.input * 1.25


#: Anthropic first-party prices (per million tokens) for the default models.
DEFAULT_PRICES: t.Dict[str, ModelPrice] = {
    "claude-haiku-5-5": ModelPrice(input=0.10, output=0.50),
    "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00, cache_read=0.20),
}


@dataclass
class BudgetConfig:
    """Hard spending cap for one optimization run."""

    max_usd: float = 5.0
    prices: t.Dict[str, ModelPrice] = field(
        default_factory=lambda: dict(DEFAULT_PRICES)
    )


@dataclass
class ModelConfig:
    """Which Claude models play which role.

    Attributes:
        student: The model the RAG app runs (answers questions).
        proposer: The model MIPROv2 uses to write instruction candidates.
        effort: Effort level sent to the student and proposer.
        max_tokens: Output cap per call.
    """

    student: str = "claude-haiku-5-5"
    proposer: str = "claude-sonnet-5-5"
    effort: t.Optional[str] = "low"
    max_tokens: int = 2048


@dataclass
class LoopConfig:
    """Improvement loop: does the optimized prompt really beat the current one?

    A round is accepted only if the bootstrap confidence interval of the
    paired per-row gain on the holdout split is entirely above zero, the mean
    gain is at least ``min_gain``, and every floor holds on holdout.
    """

    max_rounds: int = 3
    min_gain: float = 0.02
    confidence: float = 0.95
    n_bootstrap: int = 2000
    seed: int = 0
