"""
Tests for LangchainLLMWrapper temperature handling.

Covers issue #2067: LangchainLLMWrapper overwrites user-set temperature on the
underlying LLM object, causing BadRequestError with models like o3 that only
accept temperature=1.0.
"""
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langchain_core.messages import AIMessage

@pytest.fixture(autouse=True)
def mock_track():
    with patch("ragas.llms.base.track"):
        yield


def make_mock_llm(temperature: float = 1.0, supports_n: bool = False) -> MagicMock:
    """Build a minimal mock LangChain LLM."""
    mock_llm = MagicMock()
    mock_llm.temperature = temperature
    mock_llm.model_name = "mock-model"
    mock_llm.model = None

    # Only add .n if the model supports it (controls which branch is taken)
    if supports_n:
        mock_llm.n = 1

    # Fake a ChatGeneration response (what ChatOpenAI returns)
    ai_msg = AIMessage(content='{"reason": "looks good", "verdict": 1}')
    chat_gen = ChatGeneration(
        message=ai_msg,
        generation_info={"finish_reason": "stop"},
    )
    llm_result = LLMResult(generations=[[chat_gen]])

    mock_llm.generate_prompt = MagicMock(return_value=llm_result)
    mock_llm.agenerate_prompt = AsyncMock(return_value=llm_result)
    return mock_llm


def make_wrapper(mock_llm, bypass_temperature: bool = False):
    """Construct a LangchainLLMWrapper suppressing the deprecation warning."""
    from ragas.llms.base import LangchainLLMWrapper

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LangchainLLMWrapper(
            mock_llm, bypass_temperature=bypass_temperature
        )


# ---------------------------------------------------------------------------
# generate_text (sync) tests
# ---------------------------------------------------------------------------

class TestGenerateTextTemperature:
    """generate_text() must respect the temperature already on the LLM object."""

    def test_user_temperature_preserved_when_none_passed(self):
        """When no temperature arg is given, use the LLM's own temperature."""
        mock_llm = make_mock_llm(temperature=1.0)
        wrapper = make_wrapper(mock_llm)

        from langchain_core.prompt_values import StringPromptValue
        prompt = StringPromptValue(text="hello")

        wrapper.generate_text(prompt, n=1, temperature=None)

        # The temperature set on the mock LLM during the call must be 1.0, not 0.01
        assert mock_llm.temperature == 1.0, (
            f"Expected temperature=1.0 (user's value) but got {mock_llm.temperature}"
        )

    def test_explicit_temperature_overrides_llm_temperature(self):
        """An explicit temperature arg must override the LLM's temperature."""
        mock_llm = make_mock_llm(temperature=1.0)
        wrapper = make_wrapper(mock_llm)

        from langchain_core.prompt_values import StringPromptValue
        prompt = StringPromptValue(text="hello")

        wrapper.generate_text(prompt, n=1, temperature=0.5)

        assert mock_llm.temperature == 1.0  # restored to original after call

    def test_ragas_default_used_when_llm_has_no_temperature(self):
        """If the LLM has no temperature attribute, fall back to ragas default."""
        mock_llm = make_mock_llm(temperature=1.0)
        del mock_llm.temperature  # simulate LLM without temperature attr

        wrapper = make_wrapper(mock_llm)

        from langchain_core.prompt_values import StringPromptValue
        prompt = StringPromptValue(text="hello")

        # Should not raise even without a temperature attribute
        wrapper.generate_text(prompt, n=1, temperature=None)


# ---------------------------------------------------------------------------
# agenerate_text (async) tests
# ---------------------------------------------------------------------------

class TestAgenerateTextTemperature:
    """agenerate_text() must respect the temperature already on the LLM object."""

    @pytest.mark.asyncio
    async def test_user_temperature_preserved_when_none_passed(self):
        """When no temperature arg is given, use the LLM's own temperature."""
        mock_llm = make_mock_llm(temperature=1.0)
        wrapper = make_wrapper(mock_llm)

        from langchain_core.prompt_values import StringPromptValue
        prompt = StringPromptValue(text="hello")

        await wrapper.agenerate_text(prompt, n=1, temperature=None)

        assert mock_llm.temperature == 1.0, (
            f"Expected temperature=1.0 (user's value) but got {mock_llm.temperature}"
        )

    @pytest.mark.asyncio
    async def test_explicit_temperature_overrides_llm_temperature(self):
        """An explicit temperature arg must override the LLM's temperature."""
        mock_llm = make_mock_llm(temperature=1.0)
        wrapper = make_wrapper(mock_llm)

        from langchain_core.prompt_values import StringPromptValue
        prompt = StringPromptValue(text="hello")

        await wrapper.agenerate_text(prompt, n=1, temperature=0.2)

        # After the call, temperature is restored to the original value
        assert mock_llm.temperature == 1.0

    @pytest.mark.asyncio
    async def test_bypass_temperature_flag_still_works(self):
        """bypass_temperature=True should skip temperature manipulation entirely."""
        mock_llm = make_mock_llm(temperature=1.0)
        wrapper = make_wrapper(mock_llm, bypass_temperature=True)

        from langchain_core.prompt_values import StringPromptValue
        prompt = StringPromptValue(text="hello")

        await wrapper.agenerate_text(prompt, n=1, temperature=None)

        # bypass_temperature means we never touched the LLM's temperature
        assert mock_llm.temperature == 1.0

    @pytest.mark.asyncio
    async def test_o3_scenario_does_not_send_wrong_temperature(self):
        """
        Regression test for issue #2067.

        User creates ChatOpenAI(model="o3", temperature=1.0).
        RAGAS must not overwrite that with 0.01 before calling agenerate_prompt.
        """
        mock_llm = make_mock_llm(temperature=1.0)
        wrapper = make_wrapper(mock_llm)

        from langchain_core.prompt_values import StringPromptValue
        prompt = StringPromptValue(text="hello")

        temperatures_seen = []

        original_agenerate = mock_llm.agenerate_prompt.side_effect

        async def capture_temperature(*args, **kwargs):
            # Record what temperature the LLM had at call time
            temperatures_seen.append(mock_llm.temperature)
            return mock_llm.agenerate_prompt.return_value

        mock_llm.agenerate_prompt.side_effect = capture_temperature

        await wrapper.agenerate_text(prompt, n=1, temperature=None)

        assert len(temperatures_seen) == 1
        assert temperatures_seen[0] == 1.0, (
            f"o3 was called with temperature={temperatures_seen[0]}, "
            f"expected 1.0. This would cause a BadRequestError with o3."
        )