from __future__ import annotations

import typing as t

import pytest
from langchain_core.callbacks import (
    AsyncCallbackManager,
    BaseCallbackHandler,
    CallbackManagerForChainGroup,
)

from ragas.dataset_schema import SingleTurnSample
from ragas.integrations.langchain import EvaluatorChain
from ragas.metrics.base import MetricType, SingleTurnMetric


class CallbackRecordingMetric(SingleTurnMetric):
    name = "recording_metric"
    _required_columns = {MetricType.SINGLE_TURN: {"user_input", "response"}}

    def __init__(self):
        self.received_callbacks: t.Optional[object] = None

    def init(self, _run_config):
        pass

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks) -> float:
        self.received_callbacks = callbacks
        return 0.5


class ChainStartRecorder(BaseCallbackHandler):
    def __init__(self):
        self.starts: list[dict[str, t.Optional[str]]] = []

    def on_chain_start(
        self,
        serialized: t.Optional[dict[str, t.Any]],
        _inputs: t.Any,
        *,
        run_id,
        parent_run_id=None,
        **_kwargs: t.Any,
    ) -> t.Any:
        self.starts.append(
            {
                "name": serialized.get("name") if serialized else None,
                "run_id": str(run_id),
                "parent_run_id": (
                    str(parent_run_id) if parent_run_id is not None else None
                ),
            }
        )


@pytest.mark.asyncio
async def test_evaluator_chain_acall_records_metric_run_under_async_parent():
    handler = ChainStartRecorder()
    root_callbacks = AsyncCallbackManager(
        handlers=[handler],
        inheritable_handlers=[handler],
    )
    run_manager = await root_callbacks.on_chain_start(
        {"name": "outer_chain"},
        {"user_input": "Question?", "response": "Answer."},
    )

    metric = CallbackRecordingMetric()
    evaluator = EvaluatorChain(metric=metric)
    result = await evaluator._acall(
        SingleTurnSample(user_input="Question?", response="Answer."),
        run_manager=run_manager,
    )

    assert result == {metric.name: 0.5}
    metric_run = next(
        (start for start in handler.starts if start["name"] == metric.name),
        None,
    )
    assert metric_run is not None, (
        f"Expected a recorded chain start for {metric.name!r}, "
        f"got {[start['name'] for start in handler.starts]!r}"
    )
    assert metric_run["parent_run_id"] == str(run_manager.run_id)


@pytest.mark.asyncio
async def test_evaluator_chain_acall_uses_group_callbacks_without_run_manager():
    metric = CallbackRecordingMetric()
    evaluator = EvaluatorChain(metric=metric)

    result = await evaluator._acall(
        SingleTurnSample(user_input="Question?", response="Answer."),
    )

    assert result == {metric.name: 0.5}
    assert isinstance(metric.received_callbacks, CallbackManagerForChainGroup)
