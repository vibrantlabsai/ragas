"""Scoring with ragas metrics, floor checks, and the bridge to DSPy metrics."""

from __future__ import annotations

import asyncio
import statistics
import threading
import typing as t
from dataclasses import dataclass, field

from ragas.optimizers.prompt_optimization.bundle import GoldenRow
from ragas.optimizers.prompt_optimization.config import OptimizationGoal

_thread_state = threading.local()


def run_without_loop(coro: t.Coroutine[t.Any, t.Any, t.Any]) -> t.Any:
    """Run a coroutine that never awaits I/O, without creating an event loop.

    Pure metrics (token overlap, exact match) are ``async def`` but never
    suspend. Driving them directly avoids creating an event loop per call:
    on Windows every new loop opens a localhost socket pair, and thousands of
    them across threads can deadlock in ``socket.accept``.
    """
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    coro.close()
    raise RuntimeError(
        "coroutine suspended; it needs an event loop (use run_in_thread_loop)"
    )


def run_in_thread_loop(coro: t.Coroutine[t.Any, t.Any, t.Any]) -> t.Any:
    """Run a coroutine on one persistent event loop per thread (for LLM metrics)."""
    loop = getattr(_thread_state, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _thread_state.loop = loop
    return loop.run_until_complete(coro)


class RowMetric(t.Protocol):
    """Scores one answer for one golden row on a 0-1 scale."""

    name: str

    def __call__(self, row: GoldenRow, answer: str) -> float: ...


@dataclass
class RagasReferenceMetric:
    """Adapter for a ragas ``collections`` metric that takes reference/response."""

    name: str
    metric: t.Any

    def __call__(self, row: GoldenRow, answer: str) -> float:
        coro = self.metric.ascore(reference=row.reference, response=answer)
        return float(run_without_loop(coro).value)


@dataclass
class RagasFaithfulnessMetric:
    """Adapter for ragas ``collections.Faithfulness`` (needs an LLM judge)."""

    metric: t.Any
    name: str = "faithfulness"

    def __call__(self, row: GoldenRow, answer: str) -> float:
        value = run_in_thread_loop(
            self.metric.ascore(
                user_input=row.question,
                response=answer,
                retrieved_contexts=list(row.contexts),
            )
        ).value
        # Faithfulness is undefined (NaN) when the answer has no statements,
        # e.g. a refusal. Count that as fully faithful: it asserts nothing.
        return 1.0 if value != value else float(value)


def default_metric_registry() -> t.Dict[str, RowMetric]:
    """Non-LLM ragas metrics, keyed by the names used in goals and reports."""
    from ragas.metrics.collections import ExactMatch, TokenOverlap

    return {
        "token_precision": RagasReferenceMetric(
            "token_precision", TokenOverlap(mode="precision")
        ),
        "token_recall": RagasReferenceMetric(
            "token_recall", TokenOverlap(mode="recall")
        ),
        "token_f1": RagasReferenceMetric("token_f1", TokenOverlap(mode="f1")),
        "token_f0_5": RagasReferenceMetric(
            "token_f0_5", TokenOverlap(mode="fbeta", beta=0.5)
        ),
        "exact_match": RagasReferenceMetric("exact_match", ExactMatch(normalize=True)),
    }


class Scorer:
    """Scores answers with a fixed set of named metrics."""

    def __init__(
        self,
        metric_names: t.Sequence[str],
        registry: t.Optional[t.Mapping[str, RowMetric]] = None,
    ):
        registry = dict(registry) if registry is not None else default_metric_registry()
        unknown = [name for name in metric_names if name not in registry]
        if unknown:
            raise KeyError(
                f"Unknown metric(s) {unknown}. Available: {sorted(registry)}. "
                "LLM-judged metrics such as faithfulness must be added to the registry."
            )
        # Keep order, drop duplicates.
        self.metric_names = list(dict.fromkeys(metric_names))
        self._metrics = {name: registry[name] for name in self.metric_names}

    def score(self, row: GoldenRow, answer: str) -> t.Dict[str, float]:
        return {name: self._metrics[name](row, answer) for name in self.metric_names}


def aggregate(per_row: t.Sequence[t.Mapping[str, float]]) -> t.Dict[str, float]:
    """Mean of every metric across rows."""
    if not per_row:
        return {}
    names = per_row[0].keys()
    return {
        name: round(statistics.fmean(r[name] for r in per_row), 6) for name in names
    }


@dataclass
class FloorCheck:
    passed: bool
    details: t.List[t.Dict[str, t.Any]] = field(default_factory=list)

    @property
    def failed_metrics(self) -> t.List[str]:
        return [d["metric"] for d in self.details if not d["passed"]]


def check_floors(
    aggregates: t.Mapping[str, float], floors: t.Mapping[str, float]
) -> FloorCheck:
    """Check aggregates against resolved (numeric) floors. Only floored metrics count."""
    details = []
    for metric, floor in floors.items():
        if metric not in aggregates:
            raise KeyError(f"floor set for {metric!r} but it was not scored")
        value = aggregates[metric]
        details.append(
            {
                "metric": metric,
                "value": value,
                "floor": floor,
                "passed": value >= floor - 1e-9,
            }
        )
    return FloorCheck(passed=all(d["passed"] for d in details), details=details)


def build_dspy_metric(
    goal: OptimizationGoal, scorer: Scorer, rows_by_id: t.Mapping[str, GoldenRow]
) -> t.Callable[..., float]:
    """A DSPy metric returning the target metric on a 0-1 scale.

    DSPy calls ``metric(gold, pred, trace=None, ...)``. ``gold`` is a
    ``dspy.Example`` carrying ``row_id``. Floors are aggregate constraints, so
    they are checked after search on whole splits, not per example here.
    """

    def metric(
        gold: t.Any,
        pred: t.Any,
        trace: t.Any = None,
        pred_name: t.Any = None,
        pred_trace: t.Any = None,
    ) -> float:
        row = rows_by_id[gold.row_id]
        answer = getattr(pred, "answer", None)
        if not isinstance(answer, str):
            return 0.0
        return scorer.score(row, answer)[goal.target_metric]

    return metric
