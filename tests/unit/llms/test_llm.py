from __future__ import annotations

import typing as t
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue

from ragas.llms.base import BaseRagasLLM, LangchainLLMWrapper


class FakeTestLLM(BaseRagasLLM):
    def llm(self):
        return self

    def generate_text(
        self,
        prompt: PromptValue,
        n=1,
        temperature: float = 0.01,
        stop=None,
        callbacks=[],
    ):
        generations = [[Generation(text=prompt.to_string())] * n]
        return LLMResult(generations=generations)

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n=1,
        temperature: t.Optional[float] = 0.01,
        stop=None,
        callbacks=[],
    ):
        temp_val = temperature if temperature is not None else 0.01
        return self.generate_text(prompt, n, temp_val, stop, callbacks)

    def is_finished(self, response: LLMResult) -> bool:
        return True


class MockLangchainLLM:
    """Mock Langchain LLM for testing bypass_n functionality."""

    def __init__(self):
        self.n = None  # This makes hasattr(self.langchain_llm, "n") return True
        self.temperature = None
        self.model_name = "mock-model"

    def generate_prompt(self, prompts, n=None, stop=None, callbacks=None):
        # Track if n was passed to the method
        self._n_passed = n
        # Simulate the behavior where if n is passed, we return n generations per prompt
        # If n is not passed, we return one generation per prompt
        num_prompts = len(prompts)
        if n is not None:
            # If n is specified, return n generations for each prompt
            generations = [
                [Generation(text="test response")] * n for _ in range(num_prompts)
            ]
        else:
            # If n is not specified, return one generation per prompt
            generations = [
                [Generation(text="test response")] for _ in range(num_prompts)
            ]
        return LLMResult(generations=generations)

    async def agenerate_prompt(self, prompts, n=None, stop=None, callbacks=None):
        # Track if n was passed to the method
        self._n_passed = n
        # If n is not passed as parameter but self.n is set, use self.n
        if n is None and hasattr(self, "n") and self.n is not None:
            n = self.n
        # Simulate the behavior where if n is passed, we return n generations per prompt
        # If n is not passed, we return one generation per prompt
        num_prompts = len(prompts)
        if n is not None:
            # If n is specified, return n generations for each prompt
            generations = [
                [Generation(text="test response")] * n for _ in range(num_prompts)
            ]
        else:
            # If n is not specified, return one generation per prompt
            generations = [
                [Generation(text="test response")] for _ in range(num_prompts)
            ]
        return LLMResult(generations=generations)


def create_mock_prompt():
    """Create a mock prompt for testing."""
    prompt = MagicMock(spec=PromptValue)
    prompt.to_string.return_value = "test prompt"
    return prompt


class TestLangchainLLMWrapperBypassN:
    """Test bypass_n functionality in LangchainLLMWrapper."""

    def test_bypass_n_true_sync_does_not_pass_n(self):
        """Test that when bypass_n=True, n is not passed to underlying LLM in sync method."""
        mock_llm = MockLangchainLLM()
        # Mock is_multiple_completion_supported to return True for this test
        with patch(
            "ragas.llms.base.is_multiple_completion_supported", return_value=True
        ):
            wrapper = LangchainLLMWrapper(langchain_llm=mock_llm, bypass_n=True)
            prompt = create_mock_prompt()

            # Call generate_text with n=3
            result = wrapper.generate_text(prompt, n=3)

            # Verify that n was not passed to the underlying LLM
            assert mock_llm._n_passed is None
            # When bypass_n=True, the wrapper should duplicate prompts instead of passing n
            # The result should still have 3 generations (created by duplicating prompts)
            assert len(result.generations[0]) == 3

    def test_bypass_n_false_sync_passes_n(self):
        """Test that when bypass_n=False (default), n is passed to underlying LLM in sync method."""
        mock_llm = MockLangchainLLM()
        # Mock is_multiple_completion_supported to return True for this test
        with patch(
            "ragas.llms.base.is_multiple_completion_supported", return_value=True
        ):
            wrapper = LangchainLLMWrapper(langchain_llm=mock_llm, bypass_n=False)
            prompt = create_mock_prompt()

            # Call generate_text with n=3
            result = wrapper.generate_text(prompt, n=3)

            # Verify that n was passed to the underlying LLM
            assert mock_llm._n_passed == 3
            # Result should have 3 generations
            assert len(result.generations[0]) == 3

    @pytest.mark.asyncio
    async def test_bypass_n_true_async_does_not_pass_n(self):
        """Test that when bypass_n=True, n is not passed to underlying LLM in async method."""
        mock_llm = MockLangchainLLM()
        wrapper = LangchainLLMWrapper(langchain_llm=mock_llm, bypass_n=True)
        prompt = create_mock_prompt()

        # Call agenerate_text with n=3
        result = await wrapper.agenerate_text(prompt, n=3)

        # Verify that n was not passed to the underlying LLM
        assert mock_llm._n_passed is None
        # When bypass_n=True, the wrapper should duplicate prompts instead of passing n
        # The result should still have 3 generations (created by duplicating prompts)
        assert len(result.generations[0]) == 3

    @pytest.mark.asyncio
    async def test_bypass_n_false_async_passes_n(self):
        """Test that when bypass_n=False (default), n is passed to underlying LLM in async method."""
        mock_llm = MockLangchainLLM()
        wrapper = LangchainLLMWrapper(langchain_llm=mock_llm, bypass_n=False)
        prompt = create_mock_prompt()

        # Call agenerate_text with n=3
        result = await wrapper.agenerate_text(prompt, n=3)

        # Verify that n was passed to the underlying LLM (via n attribute)
        assert mock_llm.n == 3
        # Result should have 3 generations
        assert len(result.generations[0]) == 3

    def test_default_bypass_n_behavior(self):
        """Test that default behavior (bypass_n=False) remains unchanged."""
        mock_llm = MockLangchainLLM()
        # Mock is_multiple_completion_supported to return True for this test
        with patch(
            "ragas.llms.base.is_multiple_completion_supported", return_value=True
        ):
            # Create wrapper without explicitly setting bypass_n (should default to False)
            wrapper = LangchainLLMWrapper(langchain_llm=mock_llm)
            prompt = create_mock_prompt()

            # Call generate_text with n=2
            result = wrapper.generate_text(prompt, n=2)

            # Verify that n was passed to the underlying LLM (default behavior)
            assert mock_llm._n_passed == 2
            assert len(result.generations[0]) == 2

    @pytest.mark.asyncio
    async def test_default_bypass_n_behavior_async(self):
        """Test that default behavior (bypass_n=False) remains unchanged in async method."""
        mock_llm = MockLangchainLLM()
        # Create wrapper without explicitly setting bypass_n (should default to False)
        wrapper = LangchainLLMWrapper(langchain_llm=mock_llm)
        prompt = create_mock_prompt()

        # Call agenerate_text with n=2
        result = await wrapper.agenerate_text(prompt, n=2)

        # Verify that n was passed to the underlying LLM (default behavior)
        assert mock_llm.n == 2
        assert len(result.generations[0]) == 2

    def test_bypass_n_true_with_multiple_completion_supported(self):
        """Test bypass_n=True with LLM that supports multiple completions."""
        # Create a mock LLM that would normally support multiple completions
        mock_llm = MockLangchainLLM()
        # Mock the is_multiple_completion_supported to return True for this test
        with patch(
            "ragas.llms.base.is_multiple_completion_supported", return_value=True
        ):
            wrapper = LangchainLLMWrapper(langchain_llm=mock_llm, bypass_n=True)
            prompt = create_mock_prompt()

            # Call generate_text with n=3
            result = wrapper.generate_text(prompt, n=3)

            # Verify that n was not passed to the underlying LLM due to bypass_n=True
            assert mock_llm._n_passed is None
            # Result should still have 3 generations (created by duplicating prompts)
            assert len(result.generations[0]) == 3

    @pytest.mark.asyncio
    async def test_bypass_n_true_with_multiple_completion_supported_async(self):
        """Test bypass_n=True with LLM that supports multiple completions in async method."""
        mock_llm = MockLangchainLLM()
        with patch(
            "ragas.llms.base.is_multiple_completion_supported", return_value=True
        ):
            wrapper = LangchainLLMWrapper(langchain_llm=mock_llm, bypass_n=True)
            prompt = create_mock_prompt()

            # Call agenerate_text with n=3
            result = await wrapper.agenerate_text(prompt, n=3)

            # Verify that n was not passed to the underlying LLM due to bypass_n=True
            assert mock_llm._n_passed is None
            # Result should still have 3 generations
            assert len(result.generations[0]) == 3


class _LLMWithTemperature:
    """Minimal LangChain-like LLM that records the temperature in effect"""

    """when each generation call runs."""

    def __init__(self, temperature):
        self.temperature = temperature
        self.observed = []

    def generate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        self.observed.append(self.temperature)
        return LLMResult(generations=[[Generation(text="r")] for _ in prompts])

    async def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        self.observed.append(self.temperature)
        return LLMResult(generations=[[Generation(text="r")] for _ in prompts])


class TestLangchainLLMWrapperRespectsLLMTemperature:
    """Regression tests for #1976: a temperature configured on the wrapped
    LangChain LLM must be honoured when the caller does not specify one."""

    @pytest.mark.asyncio
    async def test_async_uses_underlying_temperature_when_caller_passes_none(self):
        llm = _LLMWithTemperature(temperature=0.7)
        with patch(
            "ragas.llms.base.is_multiple_completion_supported", return_value=True
        ):
            wrapper = LangchainLLMWrapper(langchain_llm=llm)
            await wrapper.agenerate_text(create_mock_prompt(), n=1, temperature=None)
            await wrapper.agenerate_text(create_mock_prompt(), n=3, temperature=None)
        assert llm.observed == [0.7, 0.7]

    @pytest.mark.asyncio
    async def test_async_caller_temperature_still_wins(self):
        llm = _LLMWithTemperature(temperature=0.7)
        with patch(
            "ragas.llms.base.is_multiple_completion_supported", return_value=True
        ):
            wrapper = LangchainLLMWrapper(langchain_llm=llm)
            await wrapper.agenerate_text(create_mock_prompt(), n=1, temperature=0.4)
        assert llm.observed == [0.4]

    def test_sync_uses_underlying_temperature_when_caller_passes_none(self):
        llm = _LLMWithTemperature(temperature=0.7)
        with patch(
            "ragas.llms.base.is_multiple_completion_supported", return_value=True
        ):
            wrapper = LangchainLLMWrapper(langchain_llm=llm)
            wrapper.generate_text(create_mock_prompt(), n=1, temperature=None)
            wrapper.generate_text(create_mock_prompt(), n=3, temperature=None)
        assert llm.observed == [0.7, 0.7]
