"""is_finished() against the metadata real providers actually send.

The Anthropic cases carry both `finish_reason` and `stop_reason`, because
langchain-anthropic sets both to the same value. A fixture with only one of
them passes whether or not the two keys are read together, so it cannot tell a
working implementation from a broken one.
"""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from ragas.llms import LangchainLLMWrapper


class _FakeLLM:
    """Stands in for a BaseLanguageModel. is_finished never calls it."""


def _wrapper() -> LangchainLLMWrapper:
    return LangchainLLMWrapper(langchain_llm=_FakeLLM())  # type: ignore[arg-type]


def _from_metadata(**metadata) -> LLMResult:
    message = AIMessage(content="ok", response_metadata=metadata)
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def _from_generation_info(**info) -> LLMResult:
    return LLMResult(generations=[[Generation(text="ok", generation_info=info)]])


@pytest.mark.parametrize(
    "metadata",
    [
        # What langchain-anthropic returns. Both keys, same value.
        {"finish_reason": "end_turn", "stop_reason": "end_turn"},
        # Newer langchain-core adds a normalised key alongside them.
        {
            "finish_reason": "end_turn",
            "stop_reason": "end_turn",
            "normalized_finish_reason": "stop",
        },
        # stop_reason alone, for providers that send only that.
        {"stop_reason": "end_turn"},
    ],
    ids=["both_keys", "both_keys_with_normalized", "stop_reason_only"],
)
def test_anthropic_completion_is_finished(metadata):
    assert _wrapper().is_finished(_from_metadata(**metadata)) is True


@pytest.mark.parametrize(
    "finish_reason", ["stop", "STOP", "MAX_TOKENS", "eos_token"], ids=lambda r: r
)
def test_existing_providers_still_finish(finish_reason):
    assert _wrapper().is_finished(_from_metadata(finish_reason=finish_reason)) is True
    assert (
        _wrapper().is_finished(_from_generation_info(finish_reason=finish_reason))
        is True
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"finish_reason": "length"},
        # Anthropic truncating sets both keys too, and must stay unfinished:
        # catching truncation is what this function is for.
        {"finish_reason": "max_tokens", "stop_reason": "max_tokens"},
        {"finish_reason": "end_turn", "normalized_finish_reason": "length"},
    ],
    ids=["openai_length", "anthropic_truncated", "normalized_overrides"],
)
def test_truncation_is_still_not_finished(metadata):
    assert _wrapper().is_finished(_from_metadata(**metadata)) is False


def test_metadata_without_any_reason_defaults_to_finished():
    assert _wrapper().is_finished(_from_metadata(model="claude-sonnet-4-6")) is True
