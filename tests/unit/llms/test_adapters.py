from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from ragas.llms.adapters import auto_detect_adapter, get_adapter
from ragas.llms.adapters.instructor import InstructorAdapter
from ragas.llms.adapters.litellm import LiteLLMAdapter


class LLMResponseModel(BaseModel):
    response: str


class MockClient:
    """Mock client that simulates an LLM client."""

    def __init__(self, is_async=False):
        self.is_async = is_async
        self.chat = Mock()
        self.chat.completions = Mock()
        self.messages = Mock()
        self.messages.create = Mock()
        if is_async:

            async def async_create(*args, **kwargs):
                return LLMResponseModel(response="Mock response")

            self.chat.completions.create = async_create
            self.messages.create = async_create
        else:

            def sync_create(*args, **kwargs):
                return LLMResponseModel(response="Mock response")

            self.chat.completions.create = sync_create
            self.messages.create = sync_create


class MockInstructor:
    """Mock instructor client that wraps the base client."""

    def __init__(self, client):
        self.client = client
        self.chat = Mock()
        self.chat.completions = Mock()

        if client.is_async:

            async def async_create(*args, **kwargs):
                return LLMResponseModel(response="Instructor response")

            self.chat.completions.create = async_create
        else:

            def sync_create(*args, **kwargs):
                return LLMResponseModel(response="Instructor response")

            self.chat.completions.create = sync_create


class TestAdapterRegistry:
    """Test adapter retrieval and management."""

    def test_get_instructor_adapter(self):
        """Test getting instructor adapter."""
        adapter = get_adapter("instructor")
        assert isinstance(adapter, InstructorAdapter)

    def test_get_litellm_adapter(self):
        """Test getting litellm adapter."""
        adapter = get_adapter("litellm")
        assert isinstance(adapter, LiteLLMAdapter)

    def test_get_unknown_adapter_raises_error(self):
        """Test that requesting unknown adapter raises ValueError."""
        with pytest.raises(ValueError, match="Unknown adapter: unknown"):
            get_adapter("unknown")


class TestAutoDetectAdapter:
    """Test auto-detection logic for adapters."""

    def test_auto_detect_google_provider_uses_litellm(self):
        """Test that google provider auto-detects litellm."""
        client = MockClient()
        adapter_name = auto_detect_adapter(client, "google")
        assert adapter_name == "litellm"

    def test_auto_detect_gemini_provider_uses_litellm(self):
        """Test that gemini provider auto-detects litellm."""
        client = MockClient()
        adapter_name = auto_detect_adapter(client, "gemini")
        assert adapter_name == "litellm"

    def test_auto_detect_openai_uses_instructor(self):
        """Test that openai provider defaults to instructor."""
        client = MockClient()
        adapter_name = auto_detect_adapter(client, "openai")
        assert adapter_name == "instructor"

    def test_auto_detect_anthropic_uses_instructor(self):
        """Test that anthropic provider defaults to instructor."""
        client = MockClient()
        adapter_name = auto_detect_adapter(client, "anthropic")
        assert adapter_name == "instructor"

    def test_auto_detect_litellm_client_uses_litellm_adapter(self):
        """Test that litellm client type auto-detects litellm adapter."""
        # Create a mock client that appears to be from litellm module
        client = Mock()
        client.__class__.__module__ = "litellm.types"

        adapter_name = auto_detect_adapter(client, "openai")
        assert adapter_name == "litellm"

    def test_auto_detect_case_insensitive(self):
        """Test that auto-detect is case-insensitive."""
        client = MockClient()

        for provider in ["GOOGLE", "Gemini", "GEMINI", "Google"]:
            adapter_name = auto_detect_adapter(client, provider)
            assert adapter_name == "litellm"


class TestInstructorAdapter:
    """Test InstructorAdapter implementation."""

    def test_instructor_adapter_create_llm(self, monkeypatch):
        """Test creating LLM with InstructorAdapter."""

        def mock_from_openai(client):
            return MockInstructor(client)

        monkeypatch.setattr("instructor.from_openai", mock_from_openai)

        adapter = InstructorAdapter()
        client = MockClient()
        llm = adapter.create_llm(client, "gpt-4o", "openai")

        assert llm is not None
        assert llm.model == "gpt-4o"
        assert llm.provider == "openai"

    def test_instructor_adapter_with_kwargs(self, monkeypatch):
        """Test InstructorAdapter passes through kwargs."""

        def mock_from_openai(client):
            return MockInstructor(client)

        monkeypatch.setattr("instructor.from_openai", mock_from_openai)

        adapter = InstructorAdapter()
        client = MockClient()
        llm = adapter.create_llm(
            client, "gpt-4o", "openai", temperature=0.7, max_tokens=2000
        )

        assert llm.model_args.get("temperature") == 0.7
        assert llm.model_args.get("max_tokens") == 2000

    def test_instructor_adapter_error_handling(self, monkeypatch):
        """Test that InstructorAdapter handles errors properly."""

        def mock_from_openai_error(client):
            raise RuntimeError("Patching failed")

        monkeypatch.setattr("instructor.from_openai", mock_from_openai_error)

        adapter = InstructorAdapter()
        client = MockClient()

        with pytest.raises(ValueError, match="Failed to patch"):
            adapter.create_llm(client, "gpt-4o", "openai")


class TestLiteLLMAdapter:
    """Test LiteLLMAdapter implementation."""

    def test_litellm_adapter_create_llm(self):
        """Test creating LLM with LiteLLMAdapter."""
        adapter = LiteLLMAdapter()
        client = MockClient()
        llm = adapter.create_llm(client, "gemini-2.0-flash", "google")

        assert llm is not None
        assert llm.model == "gemini-2.0-flash"
        assert llm.provider == "google"

    def test_litellm_adapter_with_kwargs(self):
        """Test LiteLLMAdapter passes through kwargs."""
        adapter = LiteLLMAdapter()
        client = MockClient()
        llm = adapter.create_llm(
            client, "gemini-2.0-flash", "google", temperature=0.5, max_tokens=1500
        )

        assert llm.model_args.get("temperature") == 0.5
        assert llm.model_args.get("max_tokens") == 1500

    def test_litellm_adapter_returns_litellm_structured_llm(self):
        """Test that LiteLLMAdapter returns LiteLLMStructuredLLM."""
        from ragas.llms.litellm_llm import LiteLLMStructuredLLM

        adapter = LiteLLMAdapter()
        client = MockClient()
        llm = adapter.create_llm(client, "gemini-2.0-flash", "google")

        assert isinstance(llm, LiteLLMStructuredLLM)


class TestAdapterIntegration:
    """Test adapter integration with llm_factory."""

    def test_llm_factory_with_explicit_adapter(self, monkeypatch):
        """Test llm_factory with explicit adapter selection."""
        from ragas.llms.base import llm_factory

        def mock_from_openai(client):
            return MockInstructor(client)

        monkeypatch.setattr("instructor.from_openai", mock_from_openai)

        client = MockClient()
        llm = llm_factory("gpt-4o", client=client, adapter="instructor")

        assert llm.model == "gpt-4o"
        assert llm.provider == "openai"

    def test_llm_factory_auto_detects_google_provider(self, monkeypatch):
        """Test that llm_factory auto-detects litellm for google."""
        from ragas.llms.base import llm_factory

        client = MockClient()
        llm = llm_factory("gemini-2.0-flash", provider="google", client=client)

        assert llm.model == "gemini-2.0-flash"
        assert isinstance(llm, object)  # Should be LiteLLMStructuredLLM

    def test_llm_factory_invalid_adapter_raises_error(self):
        """Test that invalid adapter name raises ValueError."""
        from ragas.llms.base import llm_factory

        client = MockClient()

        with pytest.raises(ValueError, match="Unknown adapter"):
            llm_factory("gpt-4o", client=client, adapter="invalid_adapter")
