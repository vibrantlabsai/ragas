from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.llms.litellm_llm import LiteLLMStructuredLLM


class ResponseModel(BaseModel):
    content: str


class MockInstructorClient:
    def __init__(self, is_async=False):
        self.is_async = is_async
        self.chat = Mock()
        self.chat.completions = Mock()
        self.last_messages = None

        if is_async:

            async def async_create(*args, **kwargs):
                self.last_messages = kwargs.get("messages")
                return ResponseModel(content="async response")

            self.chat.completions.create = async_create
        else:

            def sync_create(*args, **kwargs):
                self.last_messages = kwargs.get("messages")
                return ResponseModel(content="sync response")

            self.chat.completions.create = sync_create


class TestInstructorLLMSystemPrompt:
    def test_system_prompt_via_model_args(self):
        client = MockInstructorClient(is_async=False)
        model_args = InstructorModelArgs(system_prompt="You are a helpful assistant")
        llm = InstructorLLM(
            client=client, model="gpt-4o", provider="openai", model_args=model_args
        )

        result = llm.generate("What is AI?", ResponseModel)

        assert client.last_messages is not None
        assert len(client.last_messages) == 2
        assert client.last_messages[0]["role"] == "system"
        assert client.last_messages[0]["content"] == "You are a helpful assistant"
        assert client.last_messages[1]["role"] == "user"
        assert client.last_messages[1]["content"] == "What is AI?"
        assert result.content == "sync response"

    def test_system_prompt_via_kwargs(self):
        client = MockInstructorClient(is_async=False)
        llm = InstructorLLM(
            client=client,
            model="gpt-4o",
            provider="openai",
            system_prompt="You are an expert",
        )

        _ = llm.generate("Explain quantum physics", ResponseModel)

        assert client.last_messages is not None
        assert len(client.last_messages) == 2
        assert client.last_messages[0]["role"] == "system"
        assert client.last_messages[0]["content"] == "You are an expert"
        assert client.last_messages[1]["role"] == "user"

    def test_no_system_prompt(self):
        client = MockInstructorClient(is_async=False)
        llm = InstructorLLM(client=client, model="gpt-4o", provider="openai")

        _ = llm.generate("Hello", ResponseModel)

        assert client.last_messages is not None
        assert len(client.last_messages) == 1
        assert client.last_messages[0]["role"] == "user"
        assert client.last_messages[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_system_prompt_async(self):
        client = MockInstructorClient(is_async=True)
        model_args = InstructorModelArgs(system_prompt="You are a technical writer")
        llm = InstructorLLM(
            client=client, model="gpt-4o", provider="openai", model_args=model_args
        )

        result = await llm.agenerate("Write documentation", ResponseModel)

        assert client.last_messages is not None
        assert len(client.last_messages) == 2
        assert client.last_messages[0]["role"] == "system"
        assert client.last_messages[0]["content"] == "You are a technical writer"
        assert client.last_messages[1]["role"] == "user"
        assert result.content == "async response"

    @pytest.mark.asyncio
    async def test_no_system_prompt_async(self):
        client = MockInstructorClient(is_async=True)
        llm = InstructorLLM(client=client, model="gpt-4o", provider="openai")

        _ = await llm.agenerate("Test prompt", ResponseModel)

        assert client.last_messages is not None
        assert len(client.last_messages) == 1
        assert client.last_messages[0]["role"] == "user"

    def test_system_prompt_not_in_model_args_dict(self):
        client = MockInstructorClient(is_async=False)
        model_args = InstructorModelArgs(
            system_prompt="You are helpful", temperature=0.5
        )
        llm = InstructorLLM(
            client=client, model="gpt-4o", provider="openai", model_args=model_args
        )

        assert "system_prompt" not in llm.model_args
        assert llm.model_args.get("temperature") == 0.5
        assert llm.system_prompt == "You are helpful"


class TestLiteLLMStructuredLLMSystemPrompt:
    def test_system_prompt_parameter(self):
        client = MockInstructorClient(is_async=False)
        llm = LiteLLMStructuredLLM(
            client=client,
            model="gemini-2.0-flash",
            provider="google",
            system_prompt="You are a code reviewer",
        )

        _ = llm.generate("Review this code", ResponseModel)

        assert client.last_messages is not None
        assert len(client.last_messages) == 2
        assert client.last_messages[0]["role"] == "system"
        assert client.last_messages[0]["content"] == "You are a code reviewer"
        assert client.last_messages[1]["role"] == "user"
        assert client.last_messages[1]["content"] == "Review this code"

    def test_no_system_prompt(self):
        client = MockInstructorClient(is_async=False)
        llm = LiteLLMStructuredLLM(
            client=client, model="gemini-2.0-flash", provider="google"
        )

        _ = llm.generate("Test", ResponseModel)

        assert client.last_messages is not None
        assert len(client.last_messages) == 1
        assert client.last_messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_system_prompt_async(self):
        client = MockInstructorClient(is_async=True)
        llm = LiteLLMStructuredLLM(
            client=client,
            model="gemini-2.0-flash",
            provider="google",
            system_prompt="You are an analyst",
        )

        _ = await llm.agenerate("Analyze data", ResponseModel)

        assert client.last_messages is not None
        assert len(client.last_messages) == 2
        assert client.last_messages[0]["role"] == "system"
        assert client.last_messages[0]["content"] == "You are an analyst"
        assert client.last_messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_no_system_prompt_async(self):
        client = MockInstructorClient(is_async=True)
        llm = LiteLLMStructuredLLM(
            client=client, model="gemini-2.0-flash", provider="google"
        )

        _ = await llm.agenerate("Test", ResponseModel)

        assert client.last_messages is not None
        assert len(client.last_messages) == 1
        assert client.last_messages[0]["role"] == "user"

    def test_system_prompt_with_other_kwargs(self):
        client = MockInstructorClient(is_async=False)
        llm = LiteLLMStructuredLLM(
            client=client,
            model="gemini-2.0-flash",
            provider="google",
            system_prompt="You are helpful",
            temperature=0.7,
            max_tokens=2000,
        )

        assert llm.system_prompt == "You are helpful"
        assert llm.model_args.get("temperature") == 0.7
        assert llm.model_args.get("max_tokens") == 2000


class TestLLMFactorySystemPrompt:
    def test_llm_factory_with_system_prompt(self, monkeypatch):
        from ragas.llms.base import llm_factory

        def mock_from_openai(client, mode=None):
            mock_client = MockInstructorClient(is_async=False)
            mock_client.client = client
            return mock_client

        monkeypatch.setattr("instructor.from_openai", mock_from_openai)

        client = Mock()
        llm = llm_factory(
            "gpt-4o",
            client=client,
            provider="openai",
            system_prompt="You are a teacher",
        )

        assert llm.system_prompt == "You are a teacher"

    def test_llm_factory_litellm_with_system_prompt(self):
        from ragas.llms.base import llm_factory

        client = Mock()
        llm = llm_factory(
            "gemini-2.0-flash",
            client=client,
            provider="google",
            adapter="litellm",
            system_prompt="You are a scientist",
        )

        assert llm.system_prompt == "You are a scientist"
