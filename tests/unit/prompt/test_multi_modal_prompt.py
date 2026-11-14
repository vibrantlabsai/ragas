"""Tests for ImageTextPrompt.generate_multiple with LangChain and Ragas LLMs."""

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.outputs import Generation, LLMResult
from pydantic import BaseModel

from ragas.llms.base import BaseRagasLLM
from ragas.prompt.multi_modal_prompt import ImageTextPrompt


class ImageTextInput(BaseModel):
    user_input: str

    def to_string_list(self):
        return [self.user_input]


class ImageTextOutput(BaseModel):
    answer: str


class FakeImageTextPrompt(ImageTextPrompt[ImageTextInput, ImageTextOutput]):
    instruction = "Answer the question."
    input_model = ImageTextInput
    output_model = ImageTextOutput
    examples = []


class FixedCompletionsRagasLLM(BaseRagasLLM):
    """Ragas LLM that always returns its fixed completions, ignoring n."""

    def __init__(self, completions):
        super().__init__()
        self.completions = completions

    def _result(self) -> LLMResult:
        return LLMResult(
            generations=[[Generation(text=text) for text in self.completions]]
        )

    def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        return self._result()

    async def agenerate_text(
        self, prompt, n=1, temperature=0.01, stop=None, callbacks=None
    ):
        return self._result()

    def is_finished(self, response: LLMResult) -> bool:
        return True


class TestImageTextPromptGenerateMultiple:
    @pytest.mark.asyncio
    async def test_generate_multiple_langchain_llm_n_greater_than_one(self):
        """generate_multiple(n>1) with a LangChain LLM must return n outputs, not raise IndexError."""
        prompt = FakeImageTextPrompt()
        llm = FakeListLLM(responses=['{"answer": "42"}', '{"answer": "43"}'])

        outputs = await prompt.generate_multiple(
            llm, ImageTextInput(user_input="What is the answer?"), n=2
        )

        assert len(outputs) == 2
        assert [output.answer for output in outputs] == ["42", "43"]

    @pytest.mark.asyncio
    async def test_generate_multiple_langchain_llm_n_one(self):
        """generate_multiple(n=1) with a LangChain LLM returns a single output."""
        prompt = FakeImageTextPrompt()
        llm = FakeListLLM(responses=['{"answer": "42"}'])

        outputs = await prompt.generate_multiple(
            llm, ImageTextInput(user_input="What is the answer?"), n=1
        )

        assert len(outputs) == 1
        assert outputs[0].answer == "42"

    @pytest.mark.asyncio
    async def test_generate_multiple_ragas_llm_n_greater_than_one(self):
        """generate_multiple(n>1) with a Ragas LLM returns n outputs."""
        prompt = FakeImageTextPrompt()
        llm = FixedCompletionsRagasLLM(['{"answer": "42"}', '{"answer": "43"}'])

        outputs = await prompt.generate_multiple(
            llm, ImageTextInput(user_input="What is the answer?"), n=2
        )

        assert len(outputs) == 2
        assert [output.answer for output in outputs] == ["42", "43"]

    @pytest.mark.asyncio
    async def test_generate_multiple_ragas_llm_fewer_generations_than_requested(
        self, caplog
    ):
        """A Ragas LLM returning fewer generations than requested degrades gracefully."""
        prompt = FakeImageTextPrompt()
        llm = FixedCompletionsRagasLLM(['{"answer": "42"}'])

        outputs = await prompt.generate_multiple(
            llm, ImageTextInput(user_input="What is the answer?"), n=2
        )

        assert len(outputs) == 1
        assert outputs[0].answer == "42"
        assert any(
            "instead of requested 2" in record.message for record in caplog.records
        )
