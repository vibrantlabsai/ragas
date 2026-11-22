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
    from oci.generative_ai import (
        GenerativeAiClient as _GenerativeAiClient,  # type: ignore
    )
except Exception:  # pragma: no cover
    _GenerativeAiClient = None  # type: ignore

# Expose for tests to patch
oci = _oci  # type: ignore
GenerativeAiClient = _GenerativeAiClient  # type: ignore


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
            model_id: The OCI model ID to use for generation
            compartment_id: The OCI compartment ID
            config: OCI configuration dictionary (optional, uses default if not provided)
            endpoint_id: Optional endpoint ID for the model
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
        # If no client and SDK not available and no endpoint fallback, raise early
        if (
            self.client is None
            and GenerativeAiClient is None
            and self.endpoint_id is None
        ):  # type: ignore
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

    def _cache_identity(self) -> t.Dict[str, t.Any]:
        """Cache identity for the OCI Gen AI wrapper.

        ``compartment_id`` and ``endpoint_id`` are excluded as deployment identity,
        and ``_oci_config`` because it carries credentials (key files, fingerprints).
        Both exclusions are deliberate - see issue #2983.
        """
        return {
            "class": type(self).__name__,
            "provider": "oci_genai",
            "model": self.model_id,
            "system_prompt": self.default_system_prompt,
        }

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

    def _create_generation_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.01,
        max_tokens: t.Optional[int] = None,
        stop: t.Optional[t.List[str]] = None,
    ) -> t.Dict[str, t.Any]:
        """Create generation request for OCI Gen AI using role-aware messages."""
        request = {
            "compartment_id": self.compartment_id,
            "serving_mode": {"model_id": self.model_id},
            "inference_request": {
                "messages": messages,
                "max_tokens": max_tokens or 1000,
                "temperature": temperature,
            },
        }

        if self.endpoint_id:
            request["serving_mode"] = {"endpoint_id": self.endpoint_id}

        if stop:
            request["inference_request"]["stop"] = stop

        return request

    def _get_client(self):
        """Lazily initialize and return the OCI client."""
        if self.client is not None:
            return self.client
        if GenerativeAiClient is None:  # type: ignore
            raise ImportError(
                "OCI SDK not found. Please install it with: pip install oci"
            )
        cfg = self._oci_config
        if cfg is None and oci is not None:  # type: ignore
            cfg = oci.config.from_file()  # type: ignore
        if cfg is None:
            cfg = {}
        self.client = GenerativeAiClient(cfg)  # type: ignore
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
                request = self._create_generation_request(
                    messages, temperature, stop=stop
                )

                response = self._get_client().generate_text(**request)

                # Extract text from response
                if hasattr(response.data, "choices") and response.data.choices:
                    text = response.data.choices[0].message.content
                elif hasattr(response.data, "text"):
                    text = response.data.text
                else:
                    text = str(response.data)

                generation = Generation(text=text)
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
                request = self._create_generation_request(
                    messages, temperature, stop=stop
                )

                response = await loop.run_in_executor(
                    None, lambda: self._get_client().generate_text(**request)
                )

                # Extract text from response
                if hasattr(response.data, "choices") and response.data.choices:
                    text = response.data.choices[0].message.content
                elif hasattr(response.data, "text"):
                    text = response.data.text
                else:
                    text = str(response.data)

                generation = Generation(text=text)
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
            model_id="cohere.command",
            compartment_id="ocid1.compartment.oc1..example"
        )

        # With custom config
        llm = oci_genai_factory(
            model_id="cohere.command",
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
