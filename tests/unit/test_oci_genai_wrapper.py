"""Tests for OCI Gen AI wrapper."""

from unittest.mock import Mock, create_autospec, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue

pytest.importorskip("oci")

from oci import generative_ai_inference, util  # noqa: E402
from oci.response import Response  # noqa: E402

from ragas.llms.oci_genai_wrapper import (  # noqa: E402
    OCIGenAIWrapper,
    oci_genai_factory,
)

models = generative_ai_inference.models

COMPARTMENT_ID = "ocid1.compartment.oc1..example"
ENDPOINT_ID = "ocid1.endpoint.oc1..example"


def cohere_response(text: str) -> Response:
    """Build the response the SDK returns for Cohere chat models."""
    chat_response = models.CohereChatResponse(text=text, finish_reason="COMPLETE")
    return Response(200, {}, models.ChatResult(chat_response=chat_response), None)


def generic_response(*texts: str) -> Response:
    """Build the response the SDK returns for generic chat models."""
    message = models.AssistantMessage(
        content=[models.TextContent(text=text) for text in texts]
    )
    chat_response = models.GenericChatResponse(
        choices=[models.ChatChoice(index=0, message=message, finish_reason="stop")]
    )
    return Response(200, {}, models.ChatResult(chat_response=chat_response), None)


def sent_details(client) -> models.ChatDetails:
    """Return the ChatDetails passed to the most recent chat call."""
    return client.chat.call_args.args[0]


class TestOCIGenAIWrapper:
    """Test cases for OCI Gen AI wrapper."""

    @pytest.fixture
    def mock_oci_client(self):
        """Mock OCI client that only allows calls the real client supports."""
        yield create_autospec(
            generative_ai_inference.GenerativeAiInferenceClient, instance=True
        )

    @pytest.fixture
    def oci_wrapper(self, mock_oci_client):
        """Create OCI wrapper instance for a Cohere model."""
        return OCIGenAIWrapper(
            model_id="cohere.command-a-03-2025",
            compartment_id=COMPARTMENT_ID,
            client=mock_oci_client,
        )

    @pytest.fixture
    def generic_wrapper(self, mock_oci_client):
        """Create OCI wrapper instance for a generic-format model."""
        return OCIGenAIWrapper(
            model_id="meta.llama-3.3-70b-instruct",
            compartment_id=COMPARTMENT_ID,
            client=mock_oci_client,
        )

    def test_initialization(self, mock_oci_client):
        """Test OCI wrapper initialization."""
        wrapper = OCIGenAIWrapper(
            model_id="cohere.command-a-03-2025",
            compartment_id=COMPARTMENT_ID,
            client=mock_oci_client,
        )

        assert wrapper.model_id == "cohere.command-a-03-2025"
        assert wrapper.compartment_id == COMPARTMENT_ID
        assert wrapper.client == mock_oci_client

    def test_initialization_with_endpoint(self, mock_oci_client):
        """Test OCI wrapper initialization with endpoint."""
        wrapper = OCIGenAIWrapper(
            model_id="cohere.command-a-03-2025",
            compartment_id=COMPARTMENT_ID,
            endpoint_id=ENDPOINT_ID,
            client=mock_oci_client,
        )

        assert wrapper.endpoint_id == ENDPOINT_ID

    def test_default_client_uses_inference_api(self):
        """The default client must be the inference client, not control plane."""
        wrapper = OCIGenAIWrapper(
            model_id="cohere.command-a-03-2025",
            compartment_id=COMPARTMENT_ID,
            config={"region": "us-chicago-1"},
        )
        path = "ragas.llms.oci_genai_wrapper.GenerativeAiInferenceClient"

        with patch(path) as client_class:
            assert wrapper._get_client() is client_class.return_value

        client_class.assert_called_once_with({"region": "us-chicago-1"})

    def test_convert_prompt_to_messages(self, oci_wrapper):
        """Test prompt conversion to role-aware messages."""
        prompt = StringPromptValue(text="Hello, world!")
        result = oci_wrapper._convert_prompt_to_messages(prompt)
        assert isinstance(result, list)
        # Last message should be the user message with content
        assert result[-1]["role"] == "user"
        assert result[-1]["content"] == "Hello, world!"

    def test_cohere_request(self, oci_wrapper, mock_oci_client):
        """Cohere models use the message, preamble and chat history fields."""
        mock_oci_client.chat.return_value = cohere_response("Answer")
        prompt = ChatPromptValue(
            messages=[
                SystemMessage(content="Be brief"),
                HumanMessage(content="Hi"),
                AIMessage(content="Hello"),
                HumanMessage(content="Test prompt"),
            ]
        )

        oci_wrapper.generate_text(prompt, temperature=0.5, stop=["END"])

        details = sent_details(mock_oci_client)
        request = details.chat_request
        assert details.compartment_id == COMPARTMENT_ID
        assert isinstance(request, models.CohereChatRequest)
        assert request.message == "Test prompt"
        assert request.preamble_override == "Be brief"
        assert [type(m) for m in request.chat_history] == [
            models.CohereUserMessage,
            models.CohereChatBotMessage,
        ]
        assert [m.message for m in request.chat_history] == ["Hi", "Hello"]
        assert request.temperature == 0.5
        assert request.max_tokens == 1000
        assert request.stop_sequences == ["END"]

    def test_generic_request(self, generic_wrapper, mock_oci_client):
        """Non-Cohere models use typed messages with text content."""
        mock_oci_client.chat.return_value = generic_response("Answer")
        prompt = ChatPromptValue(
            messages=[
                SystemMessage(content="Be brief"),
                HumanMessage(content="Hi"),
                AIMessage(content="Hello"),
                HumanMessage(content="Test prompt"),
            ]
        )

        generic_wrapper.generate_text(prompt, temperature=0.5, stop=["END"])

        request = sent_details(mock_oci_client).chat_request
        assert isinstance(request, models.GenericChatRequest)
        assert [type(m) for m in request.messages] == [
            models.SystemMessage,
            models.UserMessage,
            models.AssistantMessage,
            models.UserMessage,
        ]
        assert request.messages[-1].content[0].text == "Test prompt"
        assert isinstance(request.messages[-1].content[0], models.TextContent)
        assert request.temperature == 0.5
        assert request.max_tokens == 1000
        assert request.stop == ["END"]

    def test_on_demand_serving_mode(self, oci_wrapper, mock_oci_client):
        """Without an endpoint, the model is served on demand."""
        mock_oci_client.chat.return_value = cohere_response("Answer")

        oci_wrapper.generate_text(StringPromptValue(text="Test prompt"))

        serving_mode = sent_details(mock_oci_client).serving_mode
        assert isinstance(serving_mode, models.OnDemandServingMode)
        assert serving_mode.model_id == "cohere.command-a-03-2025"

    def test_dedicated_serving_mode(self, mock_oci_client):
        """An endpoint selects dedicated serving and keeps the model's format."""
        mock_oci_client.chat.return_value = cohere_response("Answer")
        wrapper = OCIGenAIWrapper(
            model_id="cohere.command-a-03-2025",
            compartment_id=COMPARTMENT_ID,
            endpoint_id=ENDPOINT_ID,
            client=mock_oci_client,
        )

        wrapper.generate_text(StringPromptValue(text="Test prompt"))

        details = sent_details(mock_oci_client)
        assert isinstance(details.serving_mode, models.DedicatedServingMode)
        assert details.serving_mode.endpoint_id == ENDPOINT_ID
        assert isinstance(details.chat_request, models.CohereChatRequest)

    def test_chat_details_serialize(self, generic_wrapper, mock_oci_client):
        """Requests must be SDK models that serialize to the wire format."""
        mock_oci_client.chat.return_value = generic_response("Answer")

        generic_wrapper.generate_text(StringPromptValue(text="Test prompt"))

        body = util.to_dict(sent_details(mock_oci_client))
        assert body["serving_mode"]["serving_type"] == "ON_DEMAND"
        assert body["chat_request"]["api_format"] == "GENERIC"
        assert body["chat_request"]["messages"][-1]["role"] == "USER"
        assert body["chat_request"]["messages"][-1]["content"] == [
            {"type": "TEXT", "text": "Test prompt"}
        ]

    def test_generate_text_cohere_response(self, oci_wrapper, mock_oci_client):
        """Test synchronous text generation with a Cohere response."""
        mock_oci_client.chat.return_value = cohere_response("Generated text")

        prompt = StringPromptValue(text="Test prompt")
        result = oci_wrapper.generate_text(prompt, n=1, temperature=0.5)

        assert isinstance(result, LLMResult)
        assert len(result.generations) == 1
        assert len(result.generations[0]) == 1
        assert result.generations[0][0].text == "Generated text"
        mock_oci_client.chat.assert_called_once()

    def test_generate_text_generic_response(self, generic_wrapper, mock_oci_client):
        """Generic responses join the text parts of the first choice."""
        mock_oci_client.chat.return_value = generic_response("Generated ", "text")

        result = generic_wrapper.generate_text(StringPromptValue(text="Test prompt"))

        assert result.generations[0][0].text == "Generated text"

    def test_generate_text_multiple_completions(self, oci_wrapper, mock_oci_client):
        """Test multiple completions generation."""
        mock_oci_client.chat.return_value = cohere_response("Generated text")

        prompt = StringPromptValue(text="Test prompt")
        result = oci_wrapper.generate_text(prompt, n=3, temperature=0.5)

        assert isinstance(result, LLMResult)
        assert len(result.generations) == 3
        assert mock_oci_client.chat.call_count == 3

    def test_empty_choices_raises(self, generic_wrapper, mock_oci_client):
        """A response without text must fail rather than score a repr."""
        chat_response = models.GenericChatResponse(choices=[])
        mock_oci_client.chat.return_value = Response(
            200, {}, models.ChatResult(chat_response=chat_response), None
        )

        with pytest.raises(ValueError, match="no choices"):
            generic_wrapper.generate_text(StringPromptValue(text="Test prompt"))

    @pytest.mark.asyncio
    async def test_agenerate_text(self, oci_wrapper, mock_oci_client):
        """Test asynchronous text generation."""
        mock_oci_client.chat.return_value = cohere_response("Generated text")

        prompt = StringPromptValue(text="Test prompt")
        result = await oci_wrapper.agenerate_text(prompt, n=1, temperature=0.5)

        assert isinstance(result, LLMResult)
        assert len(result.generations) == 1
        assert len(result.generations[0]) == 1
        assert result.generations[0][0].text == "Generated text"

    def test_is_finished(self, oci_wrapper):
        """Test is_finished method."""
        # Test with valid generations
        generations = [[Generation(text="Valid text")]]
        result = LLMResult(generations=generations)
        assert oci_wrapper.is_finished(result) is True

        # Test with empty text
        generations = [[Generation(text="")]]
        result = LLMResult(generations=generations)
        assert oci_wrapper.is_finished(result) is False

        # Test with whitespace only
        generations = [[Generation(text="   ")]]
        result = LLMResult(generations=generations)
        assert oci_wrapper.is_finished(result) is False

    def test_repr(self, oci_wrapper):
        """Test string representation."""
        repr_str = repr(oci_wrapper)
        assert "OCIGenAIWrapper" in repr_str
        assert "cohere.command-a-03-2025" in repr_str
        assert COMPARTMENT_ID in repr_str

    def test_import_error(self):
        """Test import error when OCI SDK is not available."""
        with pytest.raises(ImportError, match="OCI SDK not found"):
            with patch(
                "ragas.llms.oci_genai_wrapper.GenerativeAiInferenceClient", None
            ):
                OCIGenAIWrapper(
                    model_id="cohere.command-a-03-2025",
                    compartment_id=COMPARTMENT_ID,
                )

    def test_import_error_with_endpoint(self):
        """An endpoint does not replace the SDK, which builds every request."""
        with pytest.raises(ImportError, match="OCI SDK not found"):
            with patch(
                "ragas.llms.oci_genai_wrapper.GenerativeAiInferenceClient", None
            ):
                OCIGenAIWrapper(
                    model_id="cohere.command-a-03-2025",
                    compartment_id=COMPARTMENT_ID,
                    endpoint_id=ENDPOINT_ID,
                )


class TestOCIGenAIFactory:
    """Test cases for OCI Gen AI factory function."""

    @patch("ragas.llms.oci_genai_wrapper.OCIGenAIWrapper")
    def test_oci_genai_factory(self, mock_wrapper_class):
        """Test OCI Gen AI factory function."""
        mock_wrapper = Mock()
        mock_wrapper_class.return_value = mock_wrapper

        result = oci_genai_factory(
            model_id="cohere.command-a-03-2025",
            compartment_id="ocid1.compartment.oc1..example",
            endpoint_id="ocid1.endpoint.oc1..example",
        )

        mock_wrapper_class.assert_called_once_with(
            model_id="cohere.command-a-03-2025",
            compartment_id="ocid1.compartment.oc1..example",
            endpoint_id="ocid1.endpoint.oc1..example",
            config=None,
            run_config=None,
            cache=None,
            default_system_prompt=None,
            client=None,
        )
        assert result == mock_wrapper

    @patch("ragas.llms.oci_genai_wrapper.OCIGenAIWrapper")
    def test_oci_genai_factory_with_config(self, mock_wrapper_class):
        """Test OCI Gen AI factory with custom config."""
        config = {"user": "test_user", "key_file": "test_key.pem"}

        oci_genai_factory(
            model_id="cohere.command-a-03-2025",
            compartment_id="ocid1.compartment.oc1..example",
            config=config,
        )

        mock_wrapper_class.assert_called_once_with(
            model_id="cohere.command-a-03-2025",
            compartment_id="ocid1.compartment.oc1..example",
            endpoint_id=None,
            config=config,
            run_config=None,
            cache=None,
            default_system_prompt=None,
            client=None,
        )
