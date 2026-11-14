"""
Regression tests for #2575: InstructorBaseRagasLLM does not fire callbacks.

When a caller registers a LangChain callback (e.g. a cost-tracking handler)
and uses InstructorBaseRagasLLM as the LLM backend, the callback's
on_llm_start / on_llm_end methods must be invoked so that token-usage and
other observability data are captured correctly.
"""
from __future__ import annotations

import typing as t
import uuid
from unittest.mock import MagicMock

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.prompt.pydantic_prompt import PydanticPrompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleOutput(BaseModel):
    answer: str


class SimpleInput(BaseModel):
    question: str


class _RecordingCallback(BaseCallbackHandler):
    """Captures every on_llm_start / on_llm_end invocation."""

    def __init__(self):
        super().__init__()
        self.llm_start_calls: list[dict] = []
        self.llm_end_calls: list[LLMResult] = []

    def on_llm_start(
        self,
        serialized: dict,
        prompts: list[str],
        *,
        run_id: uuid.UUID,
        **kwargs,
    ) -> None:
        self.llm_start_calls.append({"serialized": serialized, "prompts": prompts})

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        **kwargs,
    ) -> None:
        self.llm_end_calls.append(response)


class _FakeInstructorLLM(InstructorBaseRagasLLM):
    """Minimal synchronous InstructorBaseRagasLLM that returns a fixed answer."""

    is_async: bool = False

    def generate(
        self, prompt: str, response_model: t.Type
    ):
        instance = response_model(answer="42")
        # Attach a fake _raw_response with token usage so the extraction path
        # is exercised.
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        usage.total_tokens = 15
        raw = MagicMock()
        raw.usage = usage
        instance._raw_response = raw
        return instance

    async def agenerate(
        self, prompt: str, response_model: t.Type
    ):  # pragma: no cover
        raise NotImplementedError


class _SimplePrompt(PydanticPrompt[SimpleInput, SimpleOutput]):
    instruction = "Answer the question."
    input_model = SimpleInput
    output_model = SimpleOutput


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_instructor_llm_fires_on_llm_start_and_end():
    """on_llm_start and on_llm_end must both be called when using InstructorLLM."""
    callback = _RecordingCallback()
    llm = _FakeInstructorLLM()
    prompt = _SimplePrompt()

    await prompt.generate(
        data=SimpleInput(question="What is 6 * 7?"),
        llm=llm,
        callbacks=[callback],
    )

    assert len(callback.llm_start_calls) == 1, (
        "on_llm_start was not fired for InstructorBaseRagasLLM"
    )
    assert len(callback.llm_end_calls) == 1, (
        "on_llm_end was not fired for InstructorBaseRagasLLM"
    )


@pytest.mark.asyncio
async def test_instructor_llm_passes_token_usage_to_callback():
    """Token usage extracted from _raw_response must appear in the LLMResult
    delivered to on_llm_end so cost-tracking callbacks receive the data."""
    callback = _RecordingCallback()
    llm = _FakeInstructorLLM()
    prompt = _SimplePrompt()

    await prompt.generate(
        data=SimpleInput(question="What is 6 * 7?"),
        llm=llm,
        callbacks=[callback],
    )

    result = callback.llm_end_calls[0]
    assert result.llm_output is not None, (
        "LLMResult.llm_output must not be None when _raw_response.usage is present"
    )
    usage = result.llm_output["token_usage"]
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_instructor_llm_no_raw_response_does_not_crash():
    """When _raw_response is absent (older Instructor versions), the prompt
    must still succeed and callbacks must still fire — just without usage data."""

    class _NoRawResponseLLM(InstructorBaseRagasLLM):
        is_async: bool = False

        def generate(self, prompt: str, response_model: t.Type):
            # No _raw_response attribute set
            return response_model(answer="ok")

        async def agenerate(self, prompt: str, response_model: t.Type):  # pragma: no cover
            raise NotImplementedError

    callback = _RecordingCallback()
    llm = _NoRawResponseLLM()
    prompt = _SimplePrompt()

    await prompt.generate(
        data=SimpleInput(question="Hello?"),
        llm=llm,
        callbacks=[callback],
    )

    assert len(callback.llm_start_calls) == 1
    assert len(callback.llm_end_calls) == 1
    # Without a raw response, llm_output should be absent / None
    assert callback.llm_end_calls[0].llm_output is None
