from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from ragas.llms.base import llm_factory


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
            # Async client - create a proper async function
            async def async_create(*args, **kwargs):
                return LLMResponseModel(response="Instructor response")

            self.chat.completions.create = async_create
        else:
            # Sync client - create a regular function
            def sync_create(*args, **kwargs):
                return LLMResponseModel(response="Instructor response")

            self.chat.completions.create = sync_create


@pytest.fixture
def mock_sync_client():
    """Create a mock synchronous client."""
    return MockClient(is_async=False)


@pytest.fixture
def mock_async_client():
    """Create a mock asynchronous client."""
    return MockClient(is_async=True)


def test_llm_factory_initialization(mock_sync_client, monkeypatch):
    """Test llm_factory initialization."""

    def mock_from_openai(client, mode=None):
        return MockInstructor(client)

    monkeypatch.setattr("instructor.from_openai", mock_from_openai)

    llm = llm_factory("gpt-4", provider="openai", client=mock_sync_client)

    assert llm.model == "gpt-4"  # type: ignore
    assert llm.client is not None  # type: ignore
    assert not llm.is_async  # type: ignore


def test_llm_factory_async_detection(mock_async_client, monkeypatch):
    """Test that llm_factory correctly detects async clients."""

    def mock_from_openai(client, mode=None):
        return MockInstructor(client)

    monkeypatch.setattr("instructor.from_openai", mock_from_openai)

    llm = llm_factory("gpt-4", provider="openai", client=mock_async_client)

    assert llm.is_async  # type: ignore


def test_llm_factory_with_model_args(mock_sync_client, monkeypatch):
    """Test llm_factory with model arguments."""

    def mock_from_openai(client, mode=None):
        return MockInstructor(client)

    monkeypatch.setattr("instructor.from_openai", mock_from_openai)

    llm = llm_factory(
        "gpt-4", provider="openai", client=mock_sync_client, temperature=0.7
    )

    assert llm.model == "gpt-4"  # type: ignore
    assert llm.model_args.get("temperature") == 0.7  # type: ignore


def test_unsupported_provider(monkeypatch):
    """Test that invalid clients are handled gracefully for unknown providers."""
    mock_client = Mock()
    mock_client.chat = None
    mock_client.messages = None

    with pytest.raises(ValueError, match="Failed to initialize"):
        llm_factory("test-model", provider="unsupported", client=mock_client)


def test_sync_llm_generate(mock_sync_client, monkeypatch):
    """Test sync LLM generation."""

    def mock_from_openai(client, mode=None):
        return MockInstructor(client)

    monkeypatch.setattr("instructor.from_openai", mock_from_openai)

    llm = llm_factory("gpt-4", provider="openai", client=mock_sync_client)

    result = llm.generate("Test prompt", LLMResponseModel)

    assert isinstance(result, LLMResponseModel)
    assert result.response == "Instructor response"


@pytest.mark.asyncio
async def test_async_llm_agenerate(mock_async_client, monkeypatch):
    """Test async LLM generation."""

    def mock_from_openai(client, mode=None):
        return MockInstructor(client)

    monkeypatch.setattr("instructor.from_openai", mock_from_openai)

    llm = llm_factory("gpt-4", provider="openai", client=mock_async_client)

    result = await llm.agenerate("Test prompt", LLMResponseModel)

    assert isinstance(result, LLMResponseModel)
    assert result.response == "Instructor response"


def test_sync_client_agenerate_error(mock_sync_client, monkeypatch):
    """Test that using agenerate with sync client raises TypeError."""

    def mock_from_openai(client, mode=None):
        return MockInstructor(client)

    monkeypatch.setattr("instructor.from_openai", mock_from_openai)

    llm = llm_factory("gpt-4", provider="openai", client=mock_sync_client)

    with pytest.raises(
        TypeError, match="Cannot use agenerate\\(\\) with a synchronous client"
    ):
        import asyncio

        asyncio.run(llm.agenerate("Test prompt", LLMResponseModel))


def test_provider_support(monkeypatch):
    """Test that major providers are supported."""
    import instructor

    # Mock all provider-specific methods
    def mock_from_openai(client, mode=None):
        return MockInstructor(client)

    def mock_from_anthropic(client):
        return MockInstructor(client)

    def mock_from_gemini(client):
        return MockInstructor(client)

    def mock_from_litellm(client, mode=None):
        return MockInstructor(client)

    # Use setattr with the module object directly to avoid attribute existence checks
    monkeypatch.setattr(instructor, "from_openai", mock_from_openai, raising=False)
    monkeypatch.setattr(
        instructor, "from_anthropic", mock_from_anthropic, raising=False
    )
    monkeypatch.setattr(instructor, "from_gemini", mock_from_gemini, raising=False)
    monkeypatch.setattr(instructor, "from_litellm", mock_from_litellm, raising=False)

    # Test all major providers
    for provider in ["openai", "anthropic", "google", "gemini", "litellm"]:
        mock_client = MockClient(is_async=False)
        llm = llm_factory("test-model", provider=provider, client=mock_client)
        assert llm.model == "test-model"  # type: ignore


def test_llm_model_args_storage(mock_sync_client, monkeypatch):
    """Test that model arguments are properly stored."""

    def mock_from_openai(client, mode=None):
        return MockInstructor(client)

    monkeypatch.setattr("instructor.from_openai", mock_from_openai)

    model_args = {"temperature": 0.7, "max_tokens": 1000, "top_p": 0.9}

    llm = llm_factory("gpt-4", provider="openai", client=mock_sync_client, **model_args)

    assert llm.model_args == model_args  # type: ignore


def test_llm_factory_missing_client():
    """Test that missing client raises ValueError."""
    with pytest.raises(ValueError, match="requires a client instance"):
        llm_factory("gpt-4", provider="openai")


def test_llm_factory_missing_model():
    """Test that missing model raises ValueError."""
    mock_client = Mock()

    with pytest.raises(ValueError, match="model parameter is required"):
        llm_factory("", provider="openai", client=mock_client)


def test_openai_compatible_providers_with_openai_client(monkeypatch):
    """
    Test that OpenAI-compatible providers (DeepSeek, Groq, Mistral, etc.)
    work correctly with OpenAI SDK clients.

    This tests the fix for issue #2560 where provider="deepseek" with
    AsyncOpenAI client was failing with "'AsyncOpenAI' object has no attribute 'messages'"
    """

    def mock_from_openai(client, mode=None):
        return MockInstructor(client)

    monkeypatch.setattr("instructor.from_openai", mock_from_openai)

    # Test OpenAI-compatible providers that use chat.completions.create
    openai_compatible_providers = ["deepseek", "groq", "mistral", "cohere", "xai"]

    for provider in openai_compatible_providers:
        # Create a mock client with OpenAI-style API (chat.completions.create)
        mock_client = MockClient(is_async=True)
        # Remove messages attribute to simulate OpenAI client
        delattr(mock_client, "messages")

        # This should work now - it detects chat.completions.create and uses from_openai
        llm = llm_factory("test-model", provider=provider, client=mock_client)

        assert llm.model == "test-model"
        assert llm.is_async


def test_llm_factory_with_custom_mode(mock_sync_client, monkeypatch):
    """Test that llm_factory accepts and uses custom instructor mode."""
    import instructor

    captured_mode = None

    def mock_from_openai(client, mode=None):
        nonlocal captured_mode
        captured_mode = mode
        return MockInstructor(client)

    monkeypatch.setattr("instructor.from_openai", mock_from_openai)

    llm = llm_factory(
        "gpt-4",
        provider="openai",
        client=mock_sync_client,
        mode=instructor.Mode.MD_JSON,
    )

    assert llm.model == "gpt-4"
    assert captured_mode == instructor.Mode.MD_JSON


def test_llm_factory_default_mode_is_json(mock_sync_client, monkeypatch):
    """Test that llm_factory defaults to Mode.JSON when no mode is specified."""
    import instructor

    captured_mode = None

    def mock_from_openai(client, mode=None):
        nonlocal captured_mode
        captured_mode = mode
        return MockInstructor(client)

    monkeypatch.setattr("instructor.from_openai", mock_from_openai)

    llm = llm_factory("gpt-4", provider="openai", client=mock_sync_client)

    assert llm.model == "gpt-4"
    assert captured_mode == instructor.Mode.JSON


def test_llm_factory_mode_with_generic_provider(monkeypatch):
    """Test that mode parameter works with generic providers via _patch_client_for_provider."""
    import instructor

    captured_mode = None

    def mock_from_openai(client, mode=None):
        nonlocal captured_mode
        captured_mode = mode
        return MockInstructor(client)

    monkeypatch.setattr("instructor.from_openai", mock_from_openai)

    mock_client = MockClient(is_async=False)
    delattr(mock_client, "messages")

    llm = llm_factory(
        "custom-model",
        provider="custom-provider",
        client=mock_client,
        mode=instructor.Mode.TOOLS,
    )

    assert llm.model == "custom-model"
    assert captured_mode == instructor.Mode.TOOLS
