import math
import typing as t

import pytest
from langchain_core.outputs import Generation, LLMResult

from ragas.dataset_schema import SingleTurnSample
from ragas.llms.base import BaseRagasLLM
from ragas.metrics._nv_metrics import (
    AnswerAccuracy,
    ContextRelevance,
    ResponseGroundedness,
)


class ScriptedLLM(BaseRagasLLM):
    """Returns the given replies in order, one per call."""

    def __init__(self, replies: t.List[str]):
        super().__init__()
        self.replies = list(replies)

    def _next(self) -> LLMResult:
        return LLMResult(generations=[[Generation(text=self.replies.pop(0))]])

    def generate_text(self, prompt, *args, **kwargs) -> LLMResult:  # type: ignore
        return self._next()

    async def agenerate_text(self, prompt, *args, **kwargs) -> LLMResult:  # type: ignore
        return self._next()

    def is_finished(self, response: LLMResult) -> bool:
        return True


SAMPLE = SingleTurnSample(
    user_input="What is the capital of France?",
    response="Paris is the capital of France.",
    reference="Paris",
    retrieved_contexts=["France's capital city is Paris, located on the Seine."],
)

NV_METRICS = [
    (AnswerAccuracy, "4"),
    (ContextRelevance, "2"),
    (ResponseGroundedness, "2"),
]


@pytest.mark.parametrize("metric_cls", [cls for cls, _ in NV_METRICS])
def test_average_scores_falls_back_to_whichever_judge_succeeded(metric_cls):
    metric = metric_cls()

    assert metric.average_scores(0.5, 1.0) == 0.75
    assert metric.average_scores(1.0, float("nan")) == 1.0
    assert metric.average_scores(float("nan"), 1.0) == 1.0
    assert math.isnan(metric.average_scores(float("nan"), float("nan")))


@pytest.mark.parametrize("metric_cls, valid_reply", NV_METRICS)
@pytest.mark.parametrize("failing_judge", [0, 1])
@pytest.mark.asyncio
async def test_single_failing_judge_does_not_nan_the_score(
    metric_cls, valid_reply, failing_judge
):
    metric = metric_cls()
    failed = ["no rating"] * metric.retry
    replies = failed + [valid_reply] if failing_judge == 0 else [valid_reply] + failed
    metric.llm = ScriptedLLM(replies)

    score = await metric._single_turn_ascore(SAMPLE, callbacks=None)

    assert score == 1.0
