"""Tests for Groq integration in evaluation path."""

from unittest.mock import Mock

import pytest

from ragas.evaluation import _wrap_if_groq
from ragas.llms import GroqLLMWrapper
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig


class TestGroqEvaluationIntegration:
    """Test Groq automatic wrapping in evaluation."""

    @pytest.fixture
    def run_config(self):
        """Create a run config for testing."""
        return RunConfig()

    def test_wrap_if_groq_leaves_base_ragas_llm_unchanged(self, run_config):
        """Test that BaseRagasLLM instances are not wrapped."""
        # Create a mock BaseRagasLLM
        mock_llm = Mock(spec=BaseRagasLLM)

        result = _wrap_if_groq(mock_llm, run_config)

        # Should return the same instance
        assert result is mock_llm

    def test_wrap_if_groq_wraps_groq_client(self, run_config):
        """Test that Groq clients are detected and wrapped."""
        # Create a mock Groq client with the right structure
        mock_groq = Mock()
        mock_groq.chat.completions.create = Mock()
        type(mock_groq).__name__ = "Groq"
        type(mock_groq).__module__ = "groq"

        result = _wrap_if_groq(mock_groq, run_config, model="llama3-70b-8192")

        # Should be wrapped in GroqLLMWrapper
        assert isinstance(result, GroqLLMWrapper)
        assert result.groq_client is mock_groq
        assert result.model == "llama3-70b-8192"

    def test_wrap_if_groq_detects_groq_by_type_name(self, run_config):
        """Test detection by type name."""
        # Create a mock with Groq in type name
        mock_client = Mock()
        mock_client.chat.completions.create = Mock()
        type(mock_client).__name__ = "AsyncGroq"
        type(mock_client).__module__ = "groq.resources"

        result = _wrap_if_groq(mock_client, run_config)

        assert isinstance(result, GroqLLMWrapper)

    def test_wrap_if_groq_leaves_openai_unchanged(self, run_config):
        """Test that OpenAI clients are not wrapped as Groq."""
        # Create a mock OpenAI client (similar structure but different type)
        mock_openai = Mock()
        mock_openai.chat.completions.create = Mock()
        type(mock_openai).__name__ = "OpenAI"
        type(mock_openai).__module__ = "openai"

        result = _wrap_if_groq(mock_openai, run_config)

        # Should NOT be wrapped (leave for other handling)
        assert result is mock_openai
        assert not isinstance(result, GroqLLMWrapper)

    def test_wrap_if_groq_leaves_ambiguous_unchanged(self, run_config):
        """Test that ambiguous clients are not wrapped."""
        # Create a mock with the right structure but unclear type
        mock_client = Mock()
        mock_client.chat.completions.create = Mock()
        type(mock_client).__name__ = "CustomClient"
        type(mock_client).__module__ = "custom.module"

        result = _wrap_if_groq(mock_client, run_config)

        # Should NOT be wrapped (ambiguous)
        assert result is mock_client
        assert not isinstance(result, GroqLLMWrapper)

    def test_wrap_if_groq_handles_none(self, run_config):
        """Test that None is handled gracefully."""
        # This shouldn't happen in practice, but test defensive coding
        mock_none = None

        # Should not raise an error
        result = _wrap_if_groq(mock_none, run_config)
        assert result is None

    def test_wrap_if_groq_handles_missing_attributes(self, run_config):
        """Test that clients without expected attributes are not wrapped."""
        # Create a mock without the expected structure
        mock_client = Mock()
        # No chat.completions.create

        result = _wrap_if_groq(mock_client, run_config)

        # Should return unchanged
        assert result is mock_client

    def test_wrap_if_groq_custom_model(self, run_config):
        """Test wrapping with custom model."""
        mock_groq = Mock()
        mock_groq.chat.completions.create = Mock()
        type(mock_groq).__name__ = "Groq"
        type(mock_groq).__module__ = "groq"

        result = _wrap_if_groq(mock_groq, run_config, model="mixtral-8x7b-32768")

        assert isinstance(result, GroqLLMWrapper)
        assert result.model == "mixtral-8x7b-32768"

    def test_wrap_if_groq_preserves_run_config(self, run_config):
        """Test that run_config is passed to the wrapper."""
        run_config.timeout = 60
        run_config.max_retries = 5

        mock_groq = Mock()
        mock_groq.chat.completions.create = Mock()
        type(mock_groq).__name__ = "Groq"
        type(mock_groq).__module__ = "groq"

        result = _wrap_if_groq(mock_groq, run_config)

        assert isinstance(result, GroqLLMWrapper)
        # The wrapper should have the run_config
        assert result.run_config.timeout == 60
        assert result.run_config.max_retries == 5


class TestGroqEvaluationImport:
    """Test that Groq imports are available in evaluation module."""

    def test_groq_wrapper_imported(self):
        """Test that GroqLLMWrapper can be imported from evaluation context."""
        from ragas.evaluation import GroqLLMWrapper as EvalGroqWrapper

        assert EvalGroqWrapper is GroqLLMWrapper

    def test_wrap_if_groq_function_exists(self):
        """Test that _wrap_if_groq helper is available."""
        from ragas.evaluation import _wrap_if_groq as wrap_func

        assert callable(wrap_func)
