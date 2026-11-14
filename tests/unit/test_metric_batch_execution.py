import asyncio
import typing as t

import pytest

from ragas.async_utils import BatchExecutionError
from ragas.metrics.base import SimpleBaseMetric
from ragas.metrics.result import MetricResult


class AsyncBatchMetric(SimpleBaseMetric):
    def __init__(
        self,
        fail_on: t.Optional[t.Set[int]] = None,
        delays: t.Optional[t.Dict[int, float]] = None,
    ):
        super().__init__(name="async_batch_metric")
        self.fail_on = fail_on or set()
        self.delays = delays or {}
        self.active = 0
        self.max_seen = 0

    def score(self, **kwargs) -> MetricResult:
        return MetricResult(value=kwargs["value"])

    async def ascore(self, **kwargs) -> MetricResult:
        value = kwargs["value"]
        self.active += 1
        self.max_seen = max(self.max_seen, self.active)
        try:
            await asyncio.sleep(self.delays.get(value, 0.0))
            if value in self.fail_on:
                raise ValueError(f"failed value {value}")
            return MetricResult(value=value)
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_abatch_score_default_raises_on_failure():
    metric = AsyncBatchMetric(fail_on={2})

    with pytest.raises(BatchExecutionError) as exc_info:
        await metric.abatch_score(
            [
                {"value": 1},
                {"value": 2},
                {"value": 3},
            ]
        )

    assert len(exc_info.value.errors) == 1
    assert exc_info.value.errors[0].index == 1
    assert exc_info.value.errors[0].error_type == "ValueError"


@pytest.mark.asyncio
async def test_abatch_score_raise_includes_completed_failures():
    metric = AsyncBatchMetric(fail_on={2, 3})

    with pytest.raises(BatchExecutionError) as exc_info:
        await metric.abatch_score(
            [
                {"value": 1},
                {"value": 2},
                {"value": 3},
            ]
        )

    assert sorted(error.index for error in exc_info.value.errors) == [1, 2]


@pytest.mark.asyncio
async def test_abatch_score_returns_results_in_input_order():
    metric = AsyncBatchMetric(delays={1: 0.03, 2: 0.01, 3: 0.02})

    results = await metric.abatch_score(
        [
            {"value": 1},
            {"value": 2},
            {"value": 3},
        ]
    )

    assert [result.value for result in results] == [1, 2, 3]


@pytest.mark.asyncio
async def test_abatch_score_max_concurrency():
    metric = AsyncBatchMetric(delays={1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01})

    results = await metric.abatch_score(
        [
            {"value": 1},
            {"value": 2},
            {"value": 3},
            {"value": 4},
        ],
        max_concurrency=2,
    )

    assert [result.value for result in results] == [1, 2, 3, 4]
    assert metric.max_seen <= 2


@pytest.mark.asyncio
async def test_abatch_score_record_failure_policy():
    metric = AsyncBatchMetric(fail_on={2})

    results = await metric.abatch_score(
        [
            {"value": 1},
            {"value": 2},
            {"value": 3},
        ],
        failure_policy="record",
    )

    assert [result.value for result in results] == [1, None, 3]
    assert results[1].reason == "ValueError: failed value 2"


@pytest.mark.asyncio
async def test_abatch_score_skip_failure_policy():
    metric = AsyncBatchMetric(fail_on={2})

    results = await metric.abatch_score(
        [
            {"value": 1},
            {"value": 2},
            {"value": 3},
        ],
        failure_policy="skip",
    )

    assert [result.value for result in results] == [1, 3]


@pytest.mark.asyncio
async def test_abatch_score_invalid_max_concurrency():
    metric = AsyncBatchMetric()

    with pytest.raises(ValueError, match="max_concurrency"):
        await metric.abatch_score([{"value": 1}], max_concurrency=0)


@pytest.mark.asyncio
async def test_abatch_score_invalid_failure_policy():
    metric = AsyncBatchMetric()

    with pytest.raises(ValueError, match="failure_policy"):
        await metric.abatch_score([{"value": 1}], failure_policy="ignore")
