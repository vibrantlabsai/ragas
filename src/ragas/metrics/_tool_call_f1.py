from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from ragas.dataset_schema import MultiTurnSample
from ragas.messages import AIMessage
from ragas.metrics.base import MetricType, MultiTurnMetric

if t.TYPE_CHECKING:
    from langchain_core.callbacks.base import Callbacks


def _make_hashable(obj: t.Any) -> t.Any:
    """Recursively convert an object to a hashable representation."""
    if isinstance(obj, dict):
        return frozenset((k, _make_hashable(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return tuple(_make_hashable(item) for item in obj)
    elif isinstance(obj, set):
        return frozenset(_make_hashable(item) for item in obj)
    return obj


@dataclass
class ToolCallF1(MultiTurnMetric):
    name: str = "tool_call_f1"
    batch_size: int = 1
    is_multi_turn: bool = True
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.MULTI_TURN: {
                "reference_tool_calls",
                "user_input",
            }
        }
    )

    def init(self, run_config):
        pass

    async def _multi_turn_ascore(
        self, sample: MultiTurnSample, callbacks: t.Optional[Callbacks] = None
    ) -> float:
        expected: set[tuple[str, frozenset]] = set()
        if sample.reference_tool_calls:
            for call in sample.reference_tool_calls:
                expected.add((call.name, _make_hashable(call.args)))

        actual: set[tuple[str, frozenset]] = set()
        for msg in sample.user_input:
            if isinstance(msg, AIMessage) and msg.tool_calls is not None:
                for call in msg.tool_calls:
                    actual.add((call.name, _make_hashable(call.args)))

        tp = len(actual & expected)
        fp = len(actual - expected)
        fn = len(expected - actual)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return round(f1, 4)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._multi_turn_ascore(MultiTurnSample(**row), callbacks)
