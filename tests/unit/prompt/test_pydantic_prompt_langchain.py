import pytest
from langchain_core.language_models.fake import FakeListLLM
from pydantic import BaseModel

from ragas.prompt.pydantic_prompt import PydanticPrompt


class DummyInput(BaseModel):
    question: str


class DummyOutput(BaseModel):
    answer: str


class DummyPrompt(PydanticPrompt[DummyInput, DummyOutput]):
    instruction = "Answer the question"
    input_model = DummyInput
    output_model = DummyOutput


@pytest.mark.asyncio
async def test_langchain_llm_parses_markdown_fenced_json():
    """Regression: direct LangChain LLMs should accept JSON wrapped in markdown fences."""
    fenced = '```json\n{"answer": "yes"}\n```'
    llm = FakeListLLM(responses=[fenced])
    prompt = DummyPrompt()
    data = DummyInput(question="Is the sky blue?")
    result = await prompt.generate(llm=llm, data=data, retries_left=0)
    assert result.answer == "yes"


@pytest.mark.asyncio
async def test_langchain_llm_parses_plain_json():
    plain = '{"answer": "yes"}'
    llm = FakeListLLM(responses=[plain])
    prompt = DummyPrompt()
    data = DummyInput(question="Is the sky blue?")
    result = await prompt.generate(llm=llm, data=data, retries_left=0)
    assert result.answer == "yes"
