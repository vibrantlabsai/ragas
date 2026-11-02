from __future__ import annotations

import typing as t

import pytest
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import StringPromptValue as PromptValue

from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.llms.base import BaseRagasLLM
from ragas.messages import HumanMessage
from ragas.metrics import AspectCritic, SimpleCriteriaScore


class ScriptedLLM(BaseRagasLLM):
    """Offline LLM that records how many completions were asked for."""

    def __init__(self, outputs: t.List[int]):
        super().__init__()
        self.outputs = outputs
        self.requested_n: t.Optional[int] = None

    def _result(self, n: int) -> LLMResult:
        self.requested_n = n
        gens = [
            Generation(
                text='{"reason": "scripted", "verdict": %d, "score": %d}' % (v, v)
            )
            for v in self.outputs[:n]
        ]
        return LLMResult(generations=[gens])

    def generate_text(self, prompt: PromptValue, n: int = 1) -> LLMResult:  # type: ignore
        return self._result(n)

    async def agenerate_text(
        self, prompt: PromptValue, n: int = 1
    ) -> LLMResult:  # type: ignore
        return self._result(n)

    async def generate(  # type: ignore
        self, prompt, n: int = 1, temperature: float = 0.01, stop=None, callbacks=None
    ) -> LLMResult:
        return self._result(n)

    def is_finished(self, response: LLMResult) -> bool:
        return True


def _single_turn_sample() -> SingleTurnSample:
    return SingleTurnSample(user_input="hi", response="hello")


def _multi_turn_sample() -> MultiTurnSample:
    return MultiTurnSample(user_input=[HumanMessage(content="hi")])


@pytest.mark.asyncio
async def test_aspect_critic_strictness_requests_majority_vote():
    llm = ScriptedLLM([1, 0, 0])
    metric = AspectCritic(
        name="scripted", definition="Is the response fine?", llm=llm, strictness=3
    )

    score = await metric.single_turn_ascore(_single_turn_sample())

    assert llm.requested_n == 3, "strictness must be forwarded as n to the LLM"
    assert score == 0, "majority vote over [1, 0, 0] must be 0"


@pytest.mark.asyncio
async def test_aspect_critic_strictness_multi_turn():
    llm = ScriptedLLM([1, 1, 0])
    metric = AspectCritic(
        name="scripted", definition="Is the interaction fine?", llm=llm, strictness=3
    )

    score = await metric.multi_turn_ascore(_multi_turn_sample())

    assert llm.requested_n == 3, "strictness must be forwarded as n to the LLM"
    assert score == 1, "majority vote over [1, 1, 0] must be 1"


@pytest.mark.asyncio
async def test_simple_criteria_score_strictness_requests_majority_vote():
    llm = ScriptedLLM([0, 1, 1])
    metric = SimpleCriteriaScore(
        name="scripted", definition="Score 0 or 1", llm=llm, strictness=3
    )

    score = await metric.single_turn_ascore(_single_turn_sample())

    assert llm.requested_n == 3, "strictness must be forwarded as n to the LLM"
    assert score == 1, "majority vote over [0, 1, 1] must be 1"
