"""OCI Gen AI LLM wrapper implementation for Ragas."""

import asyncio
import logging
import typing as t
from typing import Dict, List

from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue

from ragas._analytics import LLMUsageEvent, track
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig

logger = logging.getLogger(__name__)

# Optional, module-level OCI imports to ease testing/mocking
try:  # pragma: no cover - environment dependent
    import oci as _oci  # type: ignore
except Exception:  # pragma: no cover - absence is okay
    _oci = None  # type: ignore

try:  # pragma: no cover - environment dependent
    from oci.generative_ai_inference import (
        GenerativeAiInferenceClient as _GenerativeAiInferenceClient,  # type: ignore
        models as _oci_models,  # type: ignore
    )
except Exception:  # pragma: no cover
    _GenerativeAiInferenceClient = None  # type: ignore
    _oci_models = None  # type: ignore

# Expose for tests to patch
oci = _oci  # type: ignore
GenerativeAiInferenceClient = _GenerativeAiInferenceClient  # type: ignore
oci_models = _oci_models  # type: ignore


class OCIGenAIWrapper(BaseRagasLLM):
    """
    OCI Gen AI LLM wrapper for Ragas.

    This wrapper provides direct integration with Oracle Cloud Infrastructure
    Generative AI services without requiring LangChain or LlamaIndex.
    """

    def __init__(
        self,
        model_id: str,
        compartment_id: str,
        config: t.Optional[t.Dict[str, t.Any]] = None,
        endpoint_id: t.Optional[str] = None,
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[t.Any] = None,
        default_system_prompt: t.Optional[str] = None,
        client: t.Optional[t.Any] = None,
    ):
        """
        Initialize OCI Gen AI wrapper.

        Args:
            model_id: The OCI model ID to use for generation. IDs starting with
                "cohere." use Cohere's chat format, all others the generic format.
                Keep this set when using endpoint_id, since it selects the format.
            compartment_id: The OCI compartment ID
            config: OCI configuration dictionary (optional, uses default if not provided)
            endpoint_id: Optional endpoint ID for dedicated serving
            run_config: Ragas run configuration
            cache: Optional cache backend
        """
        super().__init__(cache=cache)

        self.model_id = model_id
        self.compartment_id = compartment_id
        self.endpoint_id = endpoint_id
        self.default_system_prompt = default_system_prompt

        # Store client/config; perform lazy initialization to keep import-optional
        self.client = client
        self._oci_config = config
        # The SDK is required even with a custom client, as it builds the requests
        if GenerativeAiInferenceClient is None:  # type: ignore
            raise ImportError(
                "OCI SDK not found. Please install it with: pip install oci"
            )

        # Set run config
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

        # Track initialization
        track(
            LLMUsageEvent(
                provider="oci_genai",
                model=model_id,
                llm_type="oci_wrapper",
                num_requests=1,
                is_async=False,
            )
        )

    def _convert_prompt_to_messages(self, prompt: PromptValue) -> List[Dict[str, str]]:
        """Convert PromptValue to a list of role-aware messages for OCI.

        Supports system, user, and assistant roles when provided by the prompt.
        Falls back to a single user message when only a string is available.
        """
        oci_messages: List[Dict[str, str]] = []

        # Add default system prompt first if configured
        if self.default_system_prompt:
            oci_messages.append(
                {"role": "system", "content": self.default_system_prompt}
            )

        # If prompt can be converted to messages (LangChain chat-style)
        if hasattr(prompt, "to_messages"):
            try:
                lc_messages = prompt.to_messages()
                for m in lc_messages:
                    # Detect role from message type/name attributes
                    role = getattr(m, "role", None)
                    if role is None:
                        cls_name = m.__class__.__name__.lower()
                        if "system" in cls_name:
                            role = "system"
                        elif "human" in cls_name or "user" in cls_name:
                            role = "user"
                        elif "ai" in cls_name or "assistant" in cls_name:
                            role = "assistant"
                        else:
                            role = "user"
                    content = getattr(m, "content", str(m))
                    oci_messages.append({"role": role, "content": content})
                return oci_messages
            except Exception:
                # Fallback to string conversion below
                pass

        # If prompt can be converted to string
        if hasattr(prompt, "to_string"):
            return oci_messages + [{"role": "user", "content": prompt.to_string()}]

        # Generic fallback
        return oci_messages + [{"role": "user", "content": str(prompt)}]

    def _create_cohere_chat_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stop: t.Optional[t.List[str]],
    ) -> t.Any:
        """Create a chat request for Cohere models."""
        system = [m["content"] for m in messages if m["role"] == "system"]
        conversation = [m for m in messages if m["role"] != "system"]
        chat_history = [
            oci_models.CohereChatBotMessage(message=m["content"])  # type: ignore
            if m["role"] == "assistant"
            else oci_models.CohereUserMessage(message=m["content"])  # type: ignore
            for m in conversation[:-1]
        ]

        return oci_models.CohereChatRequest(  # type: ignore
            message=conversation[-1]["content"],
            chat_history=chat_history,
            preamble_override="\n\n".join(system) or None,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop,
        )

    def _create_generic_chat_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stop: t.Optional[t.List[str]],
    ) -> t.Any:
        """Create a chat request for models using the generic format."""
        message_classes = {
            "system": oci_models.SystemMessage,  # type: ignore
            "assistant": oci_models.AssistantMessage,  # type: ignore
        }
        oci_messages = [
            message_classes.get(m["role"], oci_models.UserMessage)(  # type: ignore
                content=[oci_models.TextContent(text=m["content"])]  # type: ignore
            )
            for m in messages
        ]

        return oci_models.GenericChatRequest(  # type: ignore
            messages=oci_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )

    def _create_chat_details(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.01,
        max_tokens: t.Optional[int] = None,
        stop: t.Optional[t.List[str]] = None,
    ) -> t.Any:
        """Create the ChatDetails for OCI Gen AI using role-aware messages."""
        if self.endpoint_id:
            serving_mode = oci_models.DedicatedServingMode(  # type: ignore
                endpoint_id=self.endpoint_id
            )
        else:
            serving_mode = oci_models.OnDemandServingMode(  # type: ignore
                model_id=self.model_id
            )

        if self.model_id.startswith("cohere."):
            create_chat_request = self._create_cohere_chat_request
        else:
            create_chat_request = self._create_generic_chat_request

        return oci_models.ChatDetails(  # type: ignore
            compartment_id=self.compartment_id,
            serving_mode=serving_mode,
            chat_request=create_chat_request(
                messages, temperature, max_tokens or 1000, stop
            ),
        )

    def _extract_text(self, response: t.Any) -> str:
        """Extract the generated text from an OCI chat response."""
        chat_response = response.data.chat_response

        if chat_response.api_format == "COHERE":
            return chat_response.text or ""

        if not chat_response.choices:
            raise ValueError(f"OCI Gen AI returned no choices for {self.model_id}")

        content = chat_response.choices[0].message.content or []
        return "".join(part.text for part in content if part.type == "TEXT")

    def _get_client(self):
        """Lazily initialize and return the OCI client."""
        if self.client is not None:
            return self.client
        cfg = self._oci_config
        if cfg is None and oci is not None:  # type: ignore
            cfg = oci.config.from_file()  # type: ignore
        if cfg is None:
            cfg = {}
        self.client = GenerativeAiInferenceClient(cfg)  # type: ignore
        return self.client

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[t.Any] = None,
    ) -> LLMResult:
        """Generate text using OCI Gen AI."""
        if temperature is None:
            temperature = self.get_temperature(n)

        messages = self._convert_prompt_to_messages(prompt)
        generations = []

        try:
            for _ in range(n):
                chat_details = self._create_chat_details(
                    messages, temperature, stop=stop
                )

                response = self._get_client().chat(chat_details)

                generation = Generation(text=self._extract_text(response))
                generations.append([generation])

            # Track usage
            track(
                LLMUsageEvent(
                    provider="oci_genai",
                    model=self.model_id,
                    llm_type="oci_wrapper",
                    num_requests=n,
                    is_async=False,
                )
            )

            return LLMResult(generations=generations)

        except Exception as e:
            logger.error(f"Error generating text with OCI Gen AI: {e}")
            raise

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[t.Any] = None,
    ) -> LLMResult:
        """Generate text asynchronously using OCI Gen AI."""
        if temperature is None:
            temperature = self.get_temperature(n)

        messages = self._convert_prompt_to_messages(prompt)
        generations = []

        try:
            # Run synchronous calls in thread pool for async compatibility
            loop = asyncio.get_event_loop()

            for _ in range(n):
                chat_details = self._create_chat_details(
                    messages, temperature, stop=stop
                )

                response = await loop.run_in_executor(
                    None, lambda: self._get_client().chat(chat_details)
                )

                generation = Generation(text=self._extract_text(response))
                generations.append([generation])

            # Track usage
            track(
                LLMUsageEvent(
                    provider="oci_genai",
                    model=self.model_id,
                    llm_type="oci_wrapper",
                    num_requests=n,
                    is_async=True,
                )
            )

            return LLMResult(generations=generations)

        except Exception as e:
            logger.error(f"Error generating text with OCI Gen AI: {e}")
            raise

    def is_finished(self, response: LLMResult) -> bool:
        """Check if the LLM response is finished/complete."""
        # For OCI Gen AI, we assume the response is always finished
        # unless there's an explicit error or truncation
        try:
            for generation_list in response.generations:
                for generation in generation_list:
                    if not generation.text or generation.text.strip() == "":
                        return False
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id={self.model_id}, compartment_id={self.compartment_id})"


def oci_genai_factory(
    model_id: str,
    compartment_id: str,
    config: t.Optional[t.Dict[str, t.Any]] = None,
    endpoint_id: t.Optional[str] = None,
    run_config: t.Optional[RunConfig] = None,
    cache: t.Optional[t.Any] = None,
    default_system_prompt: t.Optional[str] = None,
    client: t.Optional[t.Any] = None,
) -> OCIGenAIWrapper:
    """
    Factory function to create an OCI Gen AI LLM instance.

    Args:
        model_id: The OCI model ID to use for generation
        compartment_id: The OCI compartment ID
        config: OCI configuration dictionary (optional)
        endpoint_id: Optional endpoint ID for the model
        run_config: Ragas run configuration
        **kwargs: Additional arguments passed to OCIGenAIWrapper

    Returns:
        OCIGenAIWrapper: An instance of the OCI Gen AI LLM wrapper

    Examples:
        # Basic usage with default config
        llm = oci_genai_factory(
            model_id="cohere.command-a-03-2025",
            compartment_id="ocid1.compartment.oc1..example"
        )

        # With custom config
        llm = oci_genai_factory(
            model_id="cohere.command-a-03-2025",
            compartment_id="ocid1.compartment.oc1..example",
            config={"user": "user_ocid", "key_file": "~/.oci/private_key.pem"}
        )
    """
    return OCIGenAIWrapper(
        model_id=model_id,
        compartment_id=compartment_id,
        config=config,
        endpoint_id=endpoint_id,
        run_config=run_config,
        cache=cache,
        default_system_prompt=default_system_prompt,
        client=client,
    )
