import pytest
from unittest.mock import AsyncMock, Mock

from ragas.dataset_schema import SingleTurnSample
from ragas.integrations.langchain import EvaluatorChain
from ragas.metrics.base import SingleTurnMetric


class DummyMetric(SingleTurnMetric):
    name = "dummy"

    def init(self, run_config):
        pass

    async def _single_turn_ascore(self, sample, callbacks=None):
        self.received_callbacks = callbacks
        return 1.0


@pytest.mark.asyncio
async def test_acall_passes_callback_manager_child():
    metric = DummyMetric()

    # Mock the public API that EvaluatorChain calls
    metric.single_turn_ascore = AsyncMock(return_value=1.0)

    chain = EvaluatorChain(metric=metric)

    child_callbacks = Mock()

    run_manager = Mock()
    run_manager.get_child.return_value = child_callbacks

    sample = SingleTurnSample(
        user_input="hello",
        response="world",
    )

    await chain._acall(
        sample,
        run_manager=run_manager,
    )

    metric.single_turn_ascore.assert_awaited_once()

    _, kwargs = metric.single_turn_ascore.await_args

    assert kwargs["callbacks"] is child_callbacks