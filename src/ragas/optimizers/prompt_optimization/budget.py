"""Spend tracking with a hard cap."""

from __future__ import annotations

import threading
import typing as t
from collections import defaultdict
from dataclasses import dataclass

from ragas.optimizers.prompt_optimization.config import BudgetConfig, ModelPrice


class BudgetExceeded(RuntimeError):
    """Raised when an optimization run reaches its spending cap."""


@dataclass
class Usage:
    """Token usage of one model call, as reported by the provider."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class BudgetTracker:
    """Thread-safe running total of model spend, with a per-stage breakdown.

    ``record`` adds a call's cost. When the total reaches ``max_usd`` the
    tracker raises :class:`BudgetExceeded` on the next ``check``, so in-flight
    work can finish but no new calls start.
    """

    def __init__(self, config: t.Optional[BudgetConfig] = None):
        self.config = config or BudgetConfig()
        self._lock = threading.Lock()
        self._total = 0.0
        self._calls = 0
        self._by_stage: t.Dict[str, float] = defaultdict(float)
        self._by_model: t.Dict[str, float] = defaultdict(float)
        self._tokens: t.Dict[str, int] = defaultdict(int)
        self.stage = "setup"

    def price_for(self, model: str) -> ModelPrice:
        if model not in self.config.prices:
            raise KeyError(
                f"No price configured for model {model!r}. Add it to BudgetConfig.prices."
            )
        return self.config.prices[model]

    def cost_of(self, model: str, usage: Usage) -> float:
        price = self.price_for(model)
        per_token = 1e-6
        return per_token * (
            usage.input_tokens * price.input
            + usage.output_tokens * price.output
            + usage.cache_read_input_tokens * price.cache_read_price()
            + usage.cache_creation_input_tokens * price.cache_write_price()
        )

    def check(self) -> None:
        """Raise if the cap is already reached. Call before starting a model call."""
        with self._lock:
            if self._total >= self.config.max_usd:
                raise BudgetExceeded(
                    f"Budget of ${self.config.max_usd:.2f} reached "
                    f"(spent ${self._total:.4f} over {self._calls} calls)."
                )

    def record(self, model: str, usage: Usage) -> float:
        cost = self.cost_of(model, usage)
        with self._lock:
            self._total += cost
            self._calls += 1
            self._by_stage[self.stage] += cost
            self._by_model[model] += cost
            self._tokens["input_tokens"] += usage.input_tokens
            self._tokens["output_tokens"] += usage.output_tokens
            self._tokens["cache_read_input_tokens"] += usage.cache_read_input_tokens
            self._tokens["cache_creation_input_tokens"] += (
                usage.cache_creation_input_tokens
            )
        return cost

    @property
    def total_usd(self) -> float:
        with self._lock:
            return self._total

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.config.max_usd - self.total_usd)

    def summary(self) -> t.Dict[str, t.Any]:
        with self._lock:
            return {
                "max_usd": self.config.max_usd,
                "total_usd": round(self._total, 6),
                "calls": self._calls,
                "by_stage": {k: round(v, 6) for k, v in self._by_stage.items()},
                "by_model": {k: round(v, 6) for k, v in self._by_model.items()},
                "tokens": dict(self._tokens),
            }
