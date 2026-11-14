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
        self.last_kwargs = None

        if client.is_async:
            # Async client - create a proper async function
            async def async_create(*args, **kwargs):
                self.last_kwargs = kwargs
                return LLMResponseModel(response="Instructor response")

            self.chat.completions.create = async_create
        else:
            # Sync client - create a regular function
            def sync_create(*args, **kwargs):
                self.last_kwargs = kwargs
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


@pytest.mark.parametrize(
    "model",
    [
        "openai.gpt-6-sol",
        "us.openai.gpt-6-luna",
        "global.openai.gpt-6-astra",
    ],
)
def test_openai_params_mapping_for_bedrock_model_ids(model):
    """Bedrock's provider prefixes should not hide reasoning model names."""
    from ragas.llms.base import InstructorLLM

    llm = InstructorLLM(client=Mock(), model=model, provider="openai")

    assert llm._map_openai_params() == {
        "temperature": 1.0,
        "max_completion_tokens": 1024,
    }


@pytest.mark.parametrize("model", ["openai.gpt-4.1", "openai.gpt-oss-120b-1:0"])
def test_openai_params_mapping_for_non_reasoning_bedrock_model_ids(model):
    """Bedrock prefixes should not classify legacy models as reasoning models."""
    from ragas.llms.base import InstructorLLM

    llm = InstructorLLM(client=Mock(), model=model, provider="openai")

    assert llm._map_openai_params() == {
        "temperature": 0.01,
        "top_p": 0.1,
        "max_tokens": 1024,
    }


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


@pytest.mark.asyncio
async def test_async_openai_gpt5_maps_max_tokens_to_max_completion_tokens(
    mock_async_client, monkeypatch
):
    """Test async OpenAI GPT-5 models use max_completion_tokens."""
    instructor_client = None

    def mock_from_openai(client, mode=None):
        nonlocal instructor_client
        instructor_client = MockInstructor(client)
        return instructor_client

    monkeypatch.setattr("instructor.from_openai", mock_from_openai)

    llm = llm_factory(
        "gpt-5.2",
        provider="openai",
        client=mock_async_client,
        max_tokens=2000,
    )

    result = await llm.agenerate("Test prompt", LLMResponseModel)

    assert result.response == "Instructor response"
    assert instructor_client is not None
    assert instructor_client.last_kwargs["max_completion_tokens"] == 2000
    assert "max_tokens" not in instructor_client.last_kwargs


def test_sync_openai_gpt5_maps_max_tokens_to_max_completion_tokens(
    mock_sync_client, monkeypatch
):
    """Test sync OpenAI GPT-5 models use max_completion_tokens."""
    instructor_client = None

    def mock_from_openai(client, mode=None):
        nonlocal instructor_client
        instructor_client = MockInstructor(client)
        return instructor_client

    monkeypatch.setattr("instructor.from_openai", mock_from_openai)

    llm = llm_factory(
        "gpt-5.2",
        provider="openai",
        client=mock_sync_client,
        max_tokens=2000,
    )

    result = llm.generate("Test prompt", LLMResponseModel)

    assert result.response == "Instructor response"
    assert instructor_client is not None
    assert instructor_client.last_kwargs["max_completion_tokens"] == 2000
    assert "max_tokens" not in instructor_client.last_kwargs


def test_sync_client_agenerate_via_to_thread(mock_sync_client, monkeypatch):
    """Test that agenerate with sync client works via asyncio.to_thread."""

    def mock_from_openai(client, mode=None):
        return MockInstructor(client)

    monkeypatch.setattr("instructor.from_openai", mock_from_openai)

    llm = llm_factory("gpt-4", provider="openai", client=mock_sync_client)

    import asyncio

    result = asyncio.run(llm.agenerate("Test prompt", LLMResponseModel))

    assert isinstance(result, LLMResponseModel)
    assert result.response == "Instructor response"


def test_provider_support(monkeypatch):
    """Test that major providers are supported."""
    import instructor

    # Mock all provider-specific methods
    def mock_from_openai(client, mode=None):
        return MockInstructor(client)

    def mock_from_anthropic(client, mode=None):
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


def _mock_anthropic(monkeypatch):
    monkeypatch.setattr(
        "instructor.from_anthropic",
        lambda client, **kwargs: MockInstructor(client),
        raising=False,
    )


def test_anthropic_omits_default_sampling_params(mock_sync_client, monkeypatch):
    """Current Claude models reject temperature/top_p, so the defaults are not sent."""
    _mock_anthropic(monkeypatch)

    llm = llm_factory("claude-opus-5-5", provider="anthropic", client=mock_sync_client)
    params = llm._map_provider_params()  # type: ignore

    assert "temperature" not in params
    assert "top_p" not in params
    assert params["max_tokens"] == 1024


def test_anthropic_forwards_explicit_sampling_params(mock_sync_client, monkeypatch):
    """A sampling parameter the caller passes is forwarded, even at a falsy value."""
    _mock_anthropic(monkeypatch)

    llm = llm_factory(
        "claude-sonnet-4-6",
        provider="anthropic",
        client=mock_sync_client,
        temperature=0.0,
    )
    params = llm._map_provider_params()  # type: ignore

    assert params["temperature"] == 0.0
    assert "top_p" not in params


def test_anthropic_forwards_sampling_params_set_on_model_args():
    """Values set on InstructorModelArgs count as explicit; its defaults do not."""
    from ragas.llms.base import InstructorLLM, InstructorModelArgs

    llm = InstructorLLM(
        client=MockInstructor(MockClient(is_async=False)),
        model="claude-sonnet-4-6",
        provider="anthropic",
        model_args=InstructorModelArgs(top_p=0.5),
    )
    params = llm._map_provider_params()

    assert params["top_p"] == 0.5
    assert "temperature" not in params


def test_other_providers_keep_default_sampling_params(mock_sync_client, monkeypatch):
    """The Anthropic handling does not change what other providers receive."""
    monkeypatch.setattr(
        "instructor.from_openai", lambda client, mode=None: MockInstructor(client)
    )

    llm = llm_factory("gpt-4o", provider="openai", client=mock_sync_client)
    params = llm._map_provider_params()  # type: ignore

    assert params["temperature"] == 0.01
    assert params["top_p"] == 0.1


@pytest.mark.parametrize(
    "provider, instructor_factory",
    [
        ("anthropic", "from_anthropic"),
        ("google", "from_gemini"),
        ("perplexity", "from_perplexity"),
    ],
)
def test_llm_factory_forwards_explicit_mode(
    provider, instructor_factory, mock_sync_client, monkeypatch
):
    """An explicit mode reaches providers that have no ragas default mode."""
    import instructor

    captured = {}

    def mock_factory(client, **kwargs):
        captured.update(kwargs)
        return MockInstructor(client)

    monkeypatch.setattr(f"instructor.{instructor_factory}", mock_factory, raising=False)

    llm_factory(
        "test-model",
        provider=provider,
        client=mock_sync_client,
        adapter="instructor",
        mode=instructor.Mode.MD_JSON,
    )

    assert captured == {"mode": instructor.Mode.MD_JSON}


def test_llm_factory_defaults_anthropic_to_json_mode(mock_sync_client, monkeypatch):
    """Without a mode, Anthropic gets Mode.JSON instead of instructor's forced tool call."""
    import instructor

    captured = {}

    def mock_from_anthropic(client, **kwargs):
        captured.update(kwargs)
        return MockInstructor(client)

    monkeypatch.setattr("instructor.from_anthropic", mock_from_anthropic, raising=False)

    llm_factory(
        "claude-test",
        provider="anthropic",
        client=mock_sync_client,
        adapter="instructor",
    )

    assert captured == {"mode": instructor.Mode.JSON}


@pytest.mark.parametrize(
    "provider, instructor_factory",
    [
        ("google", "from_gemini"),
        ("perplexity", "from_perplexity"),
    ],
)
def test_llm_factory_keeps_instructor_default_mode(
    provider, instructor_factory, mock_sync_client, monkeypatch
):
    """Without a mode, those providers get instructor's own default, not Mode.JSON."""
    captured = {"called": False}

    def mock_factory(client, **kwargs):
        captured["called"] = True
        captured.update(kwargs)
        return MockInstructor(client)

    monkeypatch.setattr(f"instructor.{instructor_factory}", mock_factory, raising=False)

    llm_factory(
        "test-model", provider=provider, client=mock_sync_client, adapter="instructor"
    )

    assert captured == {"called": True}
