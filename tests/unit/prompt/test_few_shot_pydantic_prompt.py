import math
import typing as t

import pytest
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from pydantic import BaseModel

from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM
from ragas.prompt.few_shot_pydantic_prompt import FewShotPydanticPrompt
from ragas.prompt.pydantic_prompt import PydanticPrompt


class FakeInput(BaseModel):
    question: str


class FakeOutput(BaseModel):
    answer: str


class BaseFakePrompt(PydanticPrompt[FakeInput, FakeOutput]):
    instruction = "answer the question"
    input_model = FakeInput
    output_model = FakeOutput
    examples = [
        (FakeInput(question="alpha"), FakeOutput(answer="A")),
        (FakeInput(question="beta"), FakeOutput(answer="B")),
    ]


class DeterministicEmbeddings(BaseRagasEmbeddings):
    """Embeddings with cos(query, "alpha") = 0.5 and cos(query, "beta") = sqrt(3)/2."""

    def _embed(self, text: str) -> t.List[float]:
        if "alpha" in text:
            return [1.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0]
        return [0.5, math.sqrt(3) / 2]

    def embed_query(self, text: str) -> t.List[float]:
        return self._embed(text)

    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        return [self._embed(text) for text in texts]

    async def aembed_query(self, text: str) -> t.List[float]:
        return self._embed(text)

    async def aembed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        return [self._embed(text) for text in texts]


class ValidJsonLLM(BaseRagasLLM):
    def generate_text(  # type: ignore
        self, prompt: PromptValue, *args, **kwargs
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text='{"answer": "ok"}')]])

    async def agenerate_text(  # type: ignore
        self, prompt: PromptValue, *args, **kwargs
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text='{"answer": "ok"}')]])

    def is_finished(self, response: LLMResult) -> bool:
        return True


def _selected_examples(prompt: FewShotPydanticPrompt) -> t.List[t.Tuple[str, str]]:
    return [(i.question, o.answer) for i, o in prompt.examples]


@pytest.mark.asyncio
async def test_generate_multiple_uses_configured_threshold():
    prompt = FewShotPydanticPrompt.from_pydantic_prompt(
        BaseFakePrompt(), DeterministicEmbeddings()
    )
    prompt.threshold_for_examples = 0.4

    outputs = await prompt.generate_multiple(
        llm=ValidJsonLLM(), data=FakeInput(question="query")
    )

    assert outputs == [FakeOutput(answer="ok")]
    # similarity of "alpha" (0.5) and "beta" (sqrt(3)/2) are both >= 0.4
    assert _selected_examples(prompt) == [("alpha", "A"), ("beta", "B")]


@pytest.mark.asyncio
async def test_generate_multiple_default_threshold_filters_low_similarity():
    prompt = FewShotPydanticPrompt.from_pydantic_prompt(
        BaseFakePrompt(), DeterministicEmbeddings()
    )

    await prompt.generate_multiple(llm=ValidJsonLLM(), data=FakeInput(question="query"))

    # only "beta" (sqrt(3)/2) is above the default threshold of 0.7
    assert _selected_examples(prompt) == [("beta", "B")]
