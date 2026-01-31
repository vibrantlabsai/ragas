"""Tests for Groq LLM wrapper."""

from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import StringPromptValue

from ragas.llms.groq_wrapper import GroqLLMWrapper


class TestGroqLLMWrapper:
    """Test cases for Groq LLM wrapper."""

    @pytest.fixture
    def mock_groq_client(self):
        """Mock Groq client for testing."""
        mock_client = Mock()
        # Create a proper async mock for the chat.completions.create method
        mock_client.chat.completions.create = AsyncMock()
        return mock_client

    @pytest.fixture
    def groq_wrapper(self, mock_groq_client):
        """Create Groq wrapper instance for testing."""
        return GroqLLMWrapper(
            groq_client=mock_groq_client,
            model="llama3-70b-8192",
            rpm_limit=30,
        )

    def test_initialization(self, mock_groq_client):
        """Test Groq wrapper initialization."""
        wrapper = GroqLLMWrapper(
            groq_client=mock_groq_client,
            model="llama3-70b-8192",
            rpm_limit=60,
        )

        assert wrapper.model == "llama3-70b-8192"
        assert wrapper.groq_client == mock_groq_client
        assert wrapper.rpm_limit == 60
        assert wrapper._semaphore._value == 60

    def test_initialization_defaults(self, mock_groq_client):
        """Test Groq wrapper initialization with defaults."""
        wrapper = GroqLLMWrapper(groq_client=mock_groq_client)

        assert wrapper.model == "llama3-70b-8192"
        assert wrapper.rpm_limit == 30
        assert wrapper._semaphore._value == 30

    def test_extract_json_fenced(self, groq_wrapper):
        """Test JSON extraction from fenced code block."""
        raw = """Here is the JSON:
```json
{"key": "value"}
```
Hope that helps!"""
        result = groq_wrapper._extract_json(raw)
        assert result == '{"key": "value"}'

    def test_extract_json_naked_object(self, groq_wrapper):
        """Test JSON extraction from naked JSON object."""
        raw = """Some text before {"key": "value", "nested": {"a": 1}} and after"""
        result = groq_wrapper._extract_json(raw)
        assert '{"key": "value"' in result

    def test_extract_json_naked_array(self, groq_wrapper):
        """Test JSON extraction from naked JSON array."""
        raw = """Here is data: [{"id": 1}, {"id": 2}] done"""
        result = groq_wrapper._extract_json(raw)
        assert result == '[{"id": 1}, {"id": 2}]'

    def test_extract_json_no_json(self, groq_wrapper):
        """Test JSON extraction when no JSON is present."""
        raw = "Just plain text with no JSON"
        result = groq_wrapper._extract_json(raw)
        assert result == raw

    @pytest.mark.asyncio
    async def test_acall_once(self, groq_wrapper, mock_groq_client):
        """Test async single call to Groq API."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"result": "success"}'
        mock_groq_client.chat.completions.create.return_value = mock_response

        result = await groq_wrapper._acall_once("test prompt", 0.5)

        # Verify the call was made correctly
        mock_groq_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_groq_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "llama3-70b-8192"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "test prompt"

        # Verify result
        assert result == '{"result": "success"}'

    @pytest.mark.asyncio
    async def test_acall_once_with_json_extraction(
        self, groq_wrapper, mock_groq_client
    ):
        """Test async call with JSON extraction from fenced block."""
        # Mock response with fenced JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """Here is the response:
```json
{"extracted": true}
```"""
        mock_groq_client.chat.completions.create.return_value = mock_response

        result = await groq_wrapper._acall_once("test prompt", 0.5)
        assert result == '{"extracted": true}'

    @pytest.mark.asyncio
    async def test_agenerate_text(self, groq_wrapper, mock_groq_client):
        """Test asynchronous text generation."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated response"
        mock_groq_client.chat.completions.create.return_value = mock_response

        prompt = StringPromptValue(text="Test prompt")
        result = await groq_wrapper.agenerate_text(prompt, n=1, temperature=0.7)

        assert isinstance(result, LLMResult)
        assert len(result.generations) == 1
        assert len(result.generations[0]) == 1
        assert result.generations[0][0].text == "Generated response"

    @pytest.mark.asyncio
    async def test_agenerate_text_multiple_completions(
        self, groq_wrapper, mock_groq_client
    ):
        """Test asynchronous text generation with n > 1."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_groq_client.chat.completions.create.return_value = mock_response

        prompt = StringPromptValue(text="Test prompt")
        result = await groq_wrapper.agenerate_text(prompt, n=3, temperature=0.7)

        # Should call API 3 times
        assert mock_groq_client.chat.completions.create.call_count == 3

        # Should return 3 generations
        assert len(result.generations) == 1
        assert len(result.generations[0]) == 3

    def test_generate_text(self, groq_wrapper, mock_groq_client):
        """Test synchronous text generation."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated response"
        mock_groq_client.chat.completions.create.return_value = mock_response

        prompt = StringPromptValue(text="Test prompt")
        result = groq_wrapper.generate_text(prompt, n=1, temperature=0.7)

        assert isinstance(result, LLMResult)
        assert len(result.generations) == 1
        assert len(result.generations[0]) == 1
        assert result.generations[0][0].text == "Generated response"

    def test_is_finished(self, groq_wrapper):
        """Test is_finished method."""
        result = LLMResult(
            generations=[[Generation(text="test")]],
        )
        assert groq_wrapper.is_finished(result) is True

    def test_repr(self, groq_wrapper):
        """Test string representation."""
        repr_str = repr(groq_wrapper)
        assert "GroqLLMWrapper" in repr_str
        assert "llama3-70b-8192" in repr_str
        assert "rpm_limit=30" in repr_str

    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_groq_client):
        """Test that rate limiting semaphore is used."""
        wrapper = GroqLLMWrapper(
            groq_client=mock_groq_client,
            model="llama3-70b-8192",
            rpm_limit=2,  # Very low limit for testing
        )

        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_groq_client.chat.completions.create.return_value = mock_response

        # Semaphore should allow up to 2 concurrent calls
        assert wrapper._semaphore._value == 2

        # Make a call
        await wrapper._acall_once("test", 0.5)

        # After call completes, semaphore should be back to 2
        assert wrapper._semaphore._value == 2


class TestGroqLLMWrapperImport:
    """Test that GroqLLMWrapper can be imported from ragas.llms."""

    def test_import_from_llms(self):
        """Test importing GroqLLMWrapper from ragas.llms."""
        from ragas.llms import GroqLLMWrapper as ImportedWrapper

        assert ImportedWrapper is GroqLLMWrapper

    def test_in_all(self):
        """Test that GroqLLMWrapper is in __all__."""
        from ragas.llms import __all__

        assert "GroqLLMWrapper" in __all__
