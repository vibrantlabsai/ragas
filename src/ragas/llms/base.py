from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import instructor
from langchain_community.chat_models.vertexai import ChatVertexAI
from langchain_community.llms import VertexAI
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_openai.llms import AzureOpenAI, OpenAI
from langchain_openai.llms.base import BaseOpenAI
from pydantic import BaseModel

from ragas._analytics import LLMUsageEvent, track
from ragas.cache import CacheInterface, cacher
from ragas.exceptions import LLMDidNotFinishException
from ragas.run_config import RunConfig, add_async_retry

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from langchain_core.messages import BaseMessage
    from langchain_core.prompt_values import PromptValue
    from llama_index.core.base.llms.base import BaseLLM


logger = logging.getLogger(__name__)

# TypeVar for Instructor LLM response models
InstructorTypeVar = t.TypeVar("T", bound=BaseModel)  # type: ignore

MULTIPLE_COMPLETION_SUPPORTED = [
    OpenAI,
    ChatOpenAI,
    AzureOpenAI,
    AzureChatOpenAI,
    ChatVertexAI,
    VertexAI,
]


def is_multiple_completion_supported(llm: BaseLanguageModel) -> bool:
    """Return whether the given LLM supports n-completion."""
    for llm_type in MULTIPLE_COMPLETION_SUPPORTED:
        if isinstance(llm, llm_type):
            return True
    return False


@dataclass
class BaseRagasLLM(ABC):
    run_config: RunConfig = field(default_factory=RunConfig, repr=False)
    multiple_completion_supported: bool = field(default=False, repr=False)
    cache: t.Optional[CacheInterface] = field(default=None, repr=False)

    def __post_init__(self):
        # If a cache_backend is provided, wrap the implementation methods at construction time.
        if self.cache is not None:
            self.generate_text = cacher(cache_backend=self.cache)(self.generate_text)
            self.agenerate_text = cacher(cache_backend=self.cache)(self.agenerate_text)

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

    def get_temperature(self, n: int) -> float:
        """Return the temperature to use for completion based on n."""
        return 0.3 if n > 1 else 0.01

    @abstractmethod
    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult: ...

    @abstractmethod
    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult: ...

    @abstractmethod
    def is_finished(self, response: LLMResult) -> bool:
        """Check if the LLM response is finished/complete."""
        ...

    async def generate(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """Generate text using the given event loop."""

        if temperature is None:
            temperature = self.get_temperature(n)

        agenerate_text_with_retry = add_async_retry(
            self.agenerate_text, self.run_config
        )
        result = await agenerate_text_with_retry(
            prompt=prompt,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )

        # check there are no max_token issues
        if not self.is_finished(result):
            raise LLMDidNotFinishException()
        return result


class LangchainLLMWrapper(BaseRagasLLM):
    """
    A simple base class for RagasLLMs that is based on Langchain's BaseLanguageModel
    interface. it implements 2 functions:
    - generate_text: for generating text from a given PromptValue
    - agenerate_text: for generating text from a given PromptValue asynchronously

    # TODO: Revisit deprecation warning
    # .. deprecated::
    #     LangchainLLMWrapper is deprecated and will be removed in a future version.
    #     Use llm_factory instead:
    #     from openai import OpenAI
    #     from ragas.llms import llm_factory
    #     client = OpenAI(api_key="...")
    #     llm = llm_factory("gpt-4o-mini", client=client)
    """

    def __init__(
        self,
        langchain_llm: BaseLanguageModel,
        run_config: t.Optional[RunConfig] = None,
        is_finished_parser: t.Optional[t.Callable[[LLMResult], bool]] = None,
        cache: t.Optional[CacheInterface] = None,
        bypass_temperature: bool = False,
        bypass_n: bool = False,
    ):
        import warnings

        warnings.warn(
            "LangchainLLMWrapper is deprecated and will be removed in a future version. "
            "Use llm_factory instead: "
            "from openai import OpenAI; from ragas.llms import llm_factory; "
            "client = OpenAI(api_key='...'); llm = llm_factory('gpt-4o-mini', client=client)",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(cache=cache)
        self.langchain_llm = langchain_llm
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)
        self.is_finished_parser = is_finished_parser
        # Certain LLMs (e.g., OpenAI o1 series) do not support temperature
        self.bypass_temperature = bypass_temperature
        # Certain reasoning LLMs (e.g., OpenAI o1 series) do not support n parameter for
        self.bypass_n = bypass_n

    def is_finished(self, response: LLMResult) -> bool:
        """
        Parse the response to check if the LLM finished by checking the finish_reason
        or stop_reason. Supports OpenAI and Vertex AI models.
        """
        if self.is_finished_parser is not None:
            return self.is_finished_parser(response)
        # if no parser is provided default to our own

        is_finished_list = []
        for g in response.flatten():
            resp = g.generations[0][0]
            if resp.generation_info is not None:
                # generation_info is provided - so we parse that
                finish_reason = resp.generation_info.get("finish_reason")
                if finish_reason is not None:
                    # OpenAI uses "stop"
                    # Vertex AI uses "STOP" or "MAX_TOKENS"
                    # WatsonX AI uses "eos_token"
                    is_finished_list.append(
                        finish_reason in ["stop", "STOP", "MAX_TOKENS", "eos_token"]
                    )

                # provied more conditions here
                # https://github.com/vibrantlabsai/ragas/issues/1548

            # if generation_info is empty, we parse the response_metadata
            # this is less reliable

            elif (
                isinstance(resp, ChatGeneration)
                and t.cast(ChatGeneration, resp).message is not None
            ):
                resp_message: BaseMessage = t.cast(ChatGeneration, resp).message
                if resp_message.response_metadata.get("finish_reason") is not None:
                    finish_reason = resp_message.response_metadata.get("finish_reason")
                    is_finished_list.append(
                        finish_reason in ["stop", "STOP", "MAX_TOKENS", "eos_token"]
                    )
                elif resp_message.response_metadata.get("stop_reason") is not None:
                    stop_reason = resp_message.response_metadata.get("stop_reason")
                    is_finished_list.append(
                        stop_reason
                        in ["end_turn", "stop", "STOP", "MAX_TOKENS", "eos_token"]
                    )
            # default to True
            else:
                is_finished_list.append(True)
        return all(is_finished_list)

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        # figure out the temperature to set
        old_temperature: float | None = None
        if temperature is None:
            temperature = self.get_temperature(n=n)
        if hasattr(self.langchain_llm, "temperature"):
            old_temperature = self.langchain_llm.temperature  # type: ignore
            self.langchain_llm.temperature = temperature  # type: ignore

        if is_multiple_completion_supported(self.langchain_llm) and not self.bypass_n:
            result = self.langchain_llm.generate_prompt(
                prompts=[prompt],
                n=n,
                stop=stop,
                callbacks=callbacks,
            )
        else:
            result = self.langchain_llm.generate_prompt(
                prompts=[prompt] * n,
                stop=stop,
                callbacks=callbacks,
            )
            # make LLMResult.generation appear as if it was n_completions
            # note that LLMResult.runs is still a list that represents each run
            generations = [[g[0] for g in result.generations]]
            result.generations = generations

        # reset the temperature to the original value
        if old_temperature is not None:
            self.langchain_llm.temperature = old_temperature  # type: ignore

        # Track the usage
        track(
            LLMUsageEvent(
                provider="langchain",
                model=getattr(self.langchain_llm, "model_name", None)
                or getattr(self.langchain_llm, "model", None),
                llm_type="langchain_wrapper",
                num_requests=n,
                is_async=False,
            )
        )

        return result

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        # handle temperature
        old_temperature: float | None = None
        if temperature is None:
            temperature = self.get_temperature(n=n)
        if hasattr(self.langchain_llm, "temperature") and not self.bypass_temperature:
            old_temperature = self.langchain_llm.temperature  # type: ignore
            self.langchain_llm.temperature = temperature  # type: ignore

        # handle n
        if hasattr(self.langchain_llm, "n") and not self.bypass_n:
            self.langchain_llm.n = n  # type: ignore
            result = await self.langchain_llm.agenerate_prompt(
                prompts=[prompt],
                stop=stop,
                callbacks=callbacks,
            )
        else:
            result = await self.langchain_llm.agenerate_prompt(
                prompts=[prompt] * n,
                stop=stop,
                callbacks=callbacks,
            )
            # make LLMResult.generation appear as if it was n_completions
            # note that LLMResult.runs is still a list that represents each run
            generations = [[g[0] for g in result.generations]]
            result.generations = generations

        # reset the temperature to the original value
        if old_temperature is not None:
            self.langchain_llm.temperature = old_temperature  # type: ignore

        # Track the usage
        track(
            LLMUsageEvent(
                provider="langchain",
                model=getattr(self.langchain_llm, "model_name", None)
                or getattr(self.langchain_llm, "model", None),
                llm_type="langchain_wrapper",
                num_requests=n,
                is_async=True,
            )
        )

        return result

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

        # configure if using OpenAI API
        if isinstance(self.langchain_llm, BaseOpenAI) or isinstance(
            self.langchain_llm, ChatOpenAI
        ):
            try:
                from openai import RateLimitError
            except ImportError:
                raise ImportError(
                    "openai.error.RateLimitError not found. Please install openai package as `pip install openai`"
                )
            self.langchain_llm.request_timeout = run_config.timeout
            self.run_config.exception_types = RateLimitError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(langchain_llm={self.langchain_llm.__class__.__name__}(...))"


class LlamaIndexLLMWrapper(BaseRagasLLM):
    """
    A Adaptor for LlamaIndex LLMs

    # TODO: Revisit deprecation warning
    # .. deprecated::
    #     LlamaIndexLLMWrapper is deprecated and will be removed in a future version.
    #     Use llm_factory instead:
    #     from openai import OpenAI
    #     from ragas.llms import llm_factory
    #     client = OpenAI(api_key="...")
    #     llm = llm_factory("gpt-4o-mini", client=client)
    """

    def __init__(
        self,
        llm: BaseLLM,
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[CacheInterface] = None,
        bypass_temperature: bool = False,
    ):
        import warnings

        warnings.warn(
            "LlamaIndexLLMWrapper is deprecated and will be removed in a future version. "
            "Use llm_factory instead: "
            "from openai import OpenAI; from ragas.llms import llm_factory; "
            "client = OpenAI(api_key='...'); llm = llm_factory('gpt-4o-mini', client=client)",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(cache=cache)
        self.llm = llm
        # Certain LLMs (e.g., OpenAI o1 series) do not support temperature
        self.bypass_temperature = bypass_temperature

        try:
            self._signature = type(self.llm).__name__.lower()
        except AttributeError:
            self._signature = ""

        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def check_args(
        self,
        n: int,
        temperature: float,
        stop: t.Optional[t.List[str]],
        callbacks: Callbacks,
    ) -> dict[str, t.Any]:
        if n != 1:
            logger.warning("n values greater than 1 not support for LlamaIndex LLMs")
        if temperature != 0.01:
            logger.info("temperature kwarg passed to LlamaIndex LLM")
        if stop is not None:
            logger.info("stop kwarg passed to LlamaIndex LLM")
        if callbacks is not None:
            logger.info(
                "callbacks not supported for LlamaIndex LLMs, ignoring callbacks"
            )
        if self._signature in ["anthropic", "bedrock"]:
            return {"temperature": temperature}
        else:
            return {
                "n": n,
                "temperature": temperature,
                "stop": stop,
            }

    def is_finished(self, response: LLMResult) -> bool:
        return True

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        kwargs = self.check_args(n, temperature, stop, callbacks)
        li_response = self.llm.complete(prompt.to_string(), **kwargs)

        return LLMResult(generations=[[Generation(text=li_response.text)]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        if temperature is None:
            temperature = self.get_temperature(n)

        kwargs = self.check_args(n, temperature, stop, callbacks)

        if self.bypass_temperature:
            kwargs.pop("temperature", None)

        li_response = await self.llm.acomplete(prompt.to_string(), **kwargs)

        return LLMResult(generations=[[Generation(text=li_response.text)]])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(llm={self.llm.__class__.__name__}(...))"


def _patch_client_for_provider(client: t.Any, provider: str) -> t.Any:
    """
    Patch a client with Instructor for generic providers.

    Maps provider names to Provider enum and instantiates Instructor/AsyncInstructor.
    Supports anthropic, google, and any other provider Instructor recognizes.
    """
    from instructor import Provider

    provider_map = {
        "anthropic": Provider.ANTHROPIC,
        "google": Provider.GENAI,
        "gemini": Provider.GENAI,
        "azure": Provider.OPENAI,
        "groq": Provider.GROQ,
        "mistral": Provider.MISTRAL,
        "cohere": Provider.COHERE,
        "xai": Provider.XAI,
        "bedrock": Provider.BEDROCK,
        "deepseek": Provider.DEEPSEEK,
    }

    provider_enum = provider_map.get(provider, Provider.OPENAI)

    # Use JSON mode to avoid issues with Dict types in function calling
    if hasattr(client, "acompletion"):
        return instructor.AsyncInstructor(
            client=client,
            create=client.messages.create,
            provider=provider_enum,
            mode=instructor.Mode.JSON,
        )
    else:
        return instructor.Instructor(
            client=client,
            create=client.messages.create,
            provider=provider_enum,
            mode=instructor.Mode.JSON,
        )


def _is_new_google_genai_client(client: t.Any) -> bool:
    """
    Detect if client is from the new google-genai SDK vs old google-generativeai.

    New SDK (google-genai):
        - Import: from google import genai / import google.genai
        - Client: genai.Client(api_key="...")
        - Module: google.genai.client.Client

    Old SDK (google-generativeai):
        - Import: import google.generativeai as genai
        - Client: genai.GenerativeModel("model-name")
        - Module: google.generativeai.generative_models.GenerativeModel

    Note: The old SDK is deprecated (support ends Aug 2025). The new SDK is recommended
    but has a known upstream instructor issue with safety settings. See:
    https://github.com/567-labs/instructor/issues/1658
    """
    client_module = getattr(client, "__module__", "") or ""
    client_class = client.__class__.__name__

    # New SDK: google.genai.client.Client or similar
    if "google.genai" in client_module and "generativeai" not in client_module:
        return True

    # Check class name as fallback (new SDK uses Client, old uses GenerativeModel)
    if client_class == "Client" and "genai" in client_module.lower():
        return True

    return False


def _get_instructor_client(client: t.Any, provider: str) -> t.Any:
    """
    Get an instructor-patched client for the specified provider.

    Uses provider-specific methods when available, falls back to generic patcher.

    Note: For OpenAI, we use Mode.JSON instead of the default Mode.TOOLS because
    OpenAI's function calling (TOOLS mode) has issues with Dict type annotations
    in Pydantic models - it returns empty objects `{}` instead of proper structured
    data. Mode.JSON works correctly with all Pydantic types including Dict.
    See: https://github.com/vibrantlabsai/ragas/issues/2490

    For Google/Gemini, supports both SDKs:
    - New SDK (google-genai): Uses instructor.from_genai()
    - Old SDK (google-generativeai): Uses instructor.from_gemini()
    """
    provider_lower = provider.lower()

    if provider_lower == "openai":
        # Use JSON mode to avoid issues with Dict types in function calling
        return instructor.from_openai(client, mode=instructor.Mode.JSON)
    elif provider_lower == "anthropic":
        return instructor.from_anthropic(client)
    elif provider_lower in ("google", "gemini"):
        # Detect which Google SDK is being used
        if _is_new_google_genai_client(client):
            # New google-genai SDK - uses instructor.from_genai()
            # WARNING: Known upstream issue with instructor sending invalid safety
            # settings (HARM_CATEGORY_JAILBREAK). Track: github.com/567-labs/instructor/issues/1658
            # Workaround: Use OpenAI-compatible endpoint with Gemini base URL instead.
            return instructor.from_genai(client)
        else:
            # Old google-generativeai SDK (deprecated, support ends Aug 2025)
            return instructor.from_gemini(client)
    elif provider_lower == "litellm":
        # Use JSON mode to avoid issues with Dict types in function calling
        return instructor.from_litellm(client, mode=instructor.Mode.JSON)
    elif provider_lower == "perplexity":
        return instructor.from_perplexity(client)
    else:
        return _patch_client_for_provider(client, provider_lower)


def llm_factory(
    model: str,
    provider: str = "openai",
    client: t.Optional[t.Any] = None,
    adapter: str = "auto",
    cache: t.Optional[CacheInterface] = None,
    **kwargs: t.Any,
) -> InstructorBaseRagasLLM:
    """
    Create an LLM instance for structured output generation with automatic adapter selection.

    Supports multiple LLM providers and structured output backends with unified interface
    for both sync and async operations. Returns instances with .generate() and .agenerate()
    methods that accept Pydantic models for structured outputs.

    Auto-detects the best adapter for your provider:
    - Google Gemini → uses LiteLLM adapter
    - Other providers → uses Instructor adapter (default)
    - Explicit control available via adapter parameter

    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-sonnet", "gemini-2.0-flash").
        provider: LLM provider (default: "openai").
                 Examples: openai, anthropic, google, groq, mistral, etc.
        client: Pre-initialized client instance (required). For OpenAI, can be
               OpenAI(...) or AsyncOpenAI(...).
        adapter: Structured output adapter to use (default: "auto").
                - "auto": Auto-detect based on provider/client (recommended)
                - "instructor": Use Instructor library
                - "litellm": Use LiteLLM (supports 100+ providers)
        cache: Optional cache backend for caching LLM responses.
               Pass DiskCacheBackend() for persistent caching across runs.
               Saves costs and speeds up repeated evaluations by 60x.
        **kwargs: Additional model arguments (temperature, max_tokens, top_p, etc).

    Returns:
        InstructorBaseRagasLLM: Instance with generate() and agenerate() methods.

    Raises:
        ValueError: If client is missing, provider is unsupported, model is invalid,
                   or adapter initialization fails.

    Examples:
        from openai import OpenAI

        # Basic usage
        client = OpenAI(api_key="...")
        llm = llm_factory("gpt-4o-mini", client=client)
        response = llm.generate(prompt, ResponseModel)

        # With caching (recommended for experiments)
        from ragas.cache import DiskCacheBackend
        cache = DiskCacheBackend()
        llm = llm_factory("gpt-4o-mini", client=client, cache=cache)

        # Anthropic
        from anthropic import Anthropic
        client = Anthropic(api_key="...")
        llm = llm_factory("claude-3-sonnet", provider="anthropic", client=client)

        # Google Gemini (auto-detects litellm adapter)
        from litellm import OpenAI as LiteLLMClient
        client = LiteLLMClient(api_key="...", model="gemini-2.0-flash")
        llm = llm_factory("gemini-2.0-flash", client=client)

        # Explicit adapter selection
        llm = llm_factory("gemini-2.0-flash", client=client, adapter="litellm")

        # Async
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key="...")
        llm = llm_factory("gpt-4o-mini", client=client)
        response = await llm.agenerate(prompt, ResponseModel)
    """
    if client is None:
        raise ValueError(
            "llm_factory() requires a client instance. "
            "Text-only mode has been removed.\n\n"
            "To migrate:\n"
            "  from openai import OpenAI\n"
            "  client = OpenAI(api_key='...')\n"
            "  llm = llm_factory('gpt-4o-mini', client=client)\n\n"
            "For more details: https://docs.ragas.io/en/latest/llm-factory"
        )

    if not model:
        raise ValueError("model parameter is required")

    provider_lower = provider.lower()

    # Auto-detect adapter if needed
    if adapter == "auto":
        from ragas.llms.adapters import auto_detect_adapter

        adapter = auto_detect_adapter(client, provider_lower)

    # Create LLM using selected adapter
    from ragas.llms.adapters import get_adapter

    try:
        adapter_instance = get_adapter(adapter)
        llm = adapter_instance.create_llm(
            client, model, provider_lower, cache=cache, **kwargs
        )
    except ValueError as e:
        # Re-raise ValueError from get_adapter for unknown adapter names
        # Also handle adapter initialization failures
        if "Unknown adapter" in str(e):
            raise
        # Adapter-specific failures get wrapped
        raise ValueError(
            f"Failed to initialize {provider} client with {adapter} adapter. "
            f"Ensure you've created a valid {provider} client.\n"
            f"Error: {str(e)}"
        )
    except Exception as e:
        raise ValueError(
            f"Failed to initialize {provider} client with {adapter} adapter. "
            f"Ensure you've created a valid {provider} client.\n"
            f"Error: {str(e)}"
        )

    track(
        LLMUsageEvent(
            provider=provider,
            model=model,
            llm_type="llm_factory",
            num_requests=1,
            is_async=False,
        )
    )

    return llm


# Experimental LLM classes migrated from ragas.experimental.llms


class InstructorModelArgs(BaseModel):
    """Simple model arguments configuration for instructor LLMs

    Note: For GPT-5 and o-series models, you may need to increase max_tokens
    to 4096+ for structured output to work properly. See documentation for details.
    """

    temperature: float = 0.01
    top_p: float = 0.1
    max_tokens: int = 1024
    system_prompt: t.Optional[str] = None


class InstructorBaseRagasLLM(ABC):
    """Base class for LLMs using the Instructor library pattern."""

    @abstractmethod
    def generate(
        self, prompt: str, response_model: t.Type[InstructorTypeVar]
    ) -> InstructorTypeVar:
        """Generate a response using the configured LLM.

        For async clients, this will run the async method in the appropriate event loop.
        """

    @abstractmethod
    async def agenerate(
        self,
        prompt: str,
        response_model: t.Type[InstructorTypeVar],
    ) -> InstructorTypeVar:
        """Asynchronously generate a response using the configured LLM."""


class InstructorLLM(InstructorBaseRagasLLM):
    """LLM wrapper using the Instructor library for structured outputs."""

    def __init__(
        self,
        client: t.Any,
        model: str,
        provider: str,
        model_args: t.Optional[InstructorModelArgs] = None,
        cache: t.Optional[CacheInterface] = None,
        **kwargs,
    ):
        self.client = client
        self.model = model
        self.provider = provider

        # Use deterministic defaults if no model_args provided
        if model_args is None:
            model_args = InstructorModelArgs()

        # Convert to dict and merge with any additional kwargs
        self.model_args = {**model_args.model_dump(), **kwargs}

        # Extract system_prompt separately (not passed to LLM API)
        self.system_prompt = self.model_args.pop("system_prompt", None)

        self.cache = cache

        # Check if client is async-capable at initialization
        self.is_async = self._check_client_async()

        if self.cache is not None:
            self.generate = cacher(cache_backend=self.cache)(self.generate)  # type: ignore
            self.agenerate = cacher(cache_backend=self.cache)(self.agenerate)  # type: ignore

    def _map_provider_params(self) -> t.Dict[str, t.Any]:
        """Route to provider-specific parameter mapping.

        Each provider may have different parameter requirements:
        - Google: Wraps parameters in generation_config and renames max_tokens
        - OpenAI/Azure: Maps max_tokens to max_completion_tokens for o-series models
        - Anthropic: No special handling required (pass-through)
        - LiteLLM: No special handling required (routes internally, pass-through)
        """
        provider_lower = self.provider.lower()

        if provider_lower == "google":
            return self._map_google_params()
        elif provider_lower in ("openai", "azure"):
            return self._map_openai_params()
        else:
            # Anthropic, LiteLLM, and other providers - pass through unchanged
            return self.model_args.copy()

    def _map_openai_params(self) -> t.Dict[str, t.Any]:
        """Map parameters for OpenAI/Azure reasoning models with special constraints.

        Reasoning models (o-series and gpt-5 series) have unique requirements:
        1. max_tokens must be mapped to max_completion_tokens
        2. temperature must be set to 1.0 (only supported value)
        3. top_p parameter must be removed (not supported)

        Legacy OpenAI/Azure models (gpt-4, gpt-4o, etc.) continue to use max_tokens unchanged.

        Note on Azure deployments: Some Azure deployments restrict temperature to 1.0.
        If your Azure deployment has this constraint, pass temperature=1.0 explicitly:
        llm_factory("gpt-4o-mini", provider="azure", client=client, temperature=1.0)

        For GPT-5 and o-series models with structured output (Pydantic models):
        - Default max_tokens=1024 may not be sufficient
        - Consider increasing to 4096+ via: llm_factory(..., max_tokens=4096)
        - If structured output is truncated, increase max_tokens further

        Pattern-based matching for future-proof coverage:
        - O-series: o1, o2, o3, o4, o5, ... (all reasoning versions)
        - GPT-5 series: gpt-5, gpt-5-*, gpt-6, gpt-7, ... (all GPT-5+ models)
        - Other: codex-mini
        """
        mapped_args = self.model_args.copy()

        model_lower = self.model.lower()

        # Pattern-based detection for reasoning models that require max_completion_tokens
        # Uses prefix matching to cover current and future model variants
        def is_reasoning_model(model_str: str) -> bool:
            """Check if model is a reasoning model requiring max_completion_tokens."""
            # O-series reasoning models (o1, o1-mini, o1-2024-12-17, o2, o3, o4, o5, o6, o7, o8, o9)
            # Pattern: "o" followed by single digit 1-9, then optional "-" or end of string
            # TODO: Update to support o10+ when OpenAI releases models beyond o9
            if (
                len(model_str) >= 2
                and model_str[0] == "o"
                and model_str[1] in "123456789"
            ):
                # Allow single digit o-series: o1, o2, ..., o9
                if len(model_str) == 2 or model_str[2] in ("-", "_"):
                    return True

            # GPT-5 and newer generation models (gpt-5, gpt-5-*, gpt-6, gpt-7, ..., gpt-19)
            # Pattern: "gpt-" followed by single or double digit >= 5, max 19
            # TODO: Update to support gpt-20+ when OpenAI releases models beyond gpt-19
            if model_str.startswith("gpt-"):
                version_str = (
                    model_str[4:].split("-")[0].split("_")[0]
                )  # Get version number
                try:
                    version = int(version_str)
                    if 5 <= version <= 19:
                        return True
                except ValueError:
                    pass

            # Other specific reasoning models
            if model_str == "codex-mini":
                return True

            return False

        requires_max_completion_tokens = is_reasoning_model(model_lower)

        # If max_tokens is provided and model requires max_completion_tokens, map it
        if requires_max_completion_tokens and "max_tokens" in mapped_args:
            mapped_args["max_completion_tokens"] = mapped_args.pop("max_tokens")

        # Handle parameter constraints for reasoning models (GPT-5 and o-series)
        if requires_max_completion_tokens:
            # GPT-5 and o-series models have strict parameter requirements:
            # 1. Temperature must be exactly 1.0 (only supported value)
            # 2. top_p parameter is not supported and must be removed
            mapped_args["temperature"] = 1.0
            mapped_args.pop("top_p", None)

        return mapped_args

    def _map_google_params(self) -> t.Dict[str, t.Any]:
        """Map parameters for Google Gemini models.

        Google models require parameters to be wrapped in a generation_config dict,
        and max_tokens is renamed to max_output_tokens.
        """
        google_kwargs = {}
        generation_config_keys = {"temperature", "max_tokens", "top_p", "top_k"}
        generation_config = {}

        for key, value in self.model_args.items():
            if key in generation_config_keys:
                if key == "max_tokens":
                    generation_config["max_output_tokens"] = value
                else:
                    generation_config[key] = value
            else:
                google_kwargs[key] = value

        if generation_config:
            google_kwargs["generation_config"] = generation_config

        return google_kwargs

    def _check_client_async(self) -> bool:
        """Determine if the client is async-capable.

        Handles multiple cases:
        1. Instructor-wrapped AsyncInstructor clients (OpenAI/Anthropic/etc)
        2. Instructor-wrapped Instructor clients that wrap async underlying clients
        3. Direct async clients with chat.completions.create
        4. Instructor-wrapped clients where the underlying client is in a closure
        """
        try:
            # Check if this is an AsyncInstructor wrapper
            if self.client.__class__.__name__ == "AsyncInstructor":
                return True

            # Check if this is a sync Instructor wrapper that wraps an async client
            if hasattr(self.client, "client"):
                underlying = self.client.client
                # For OpenAI/Anthropic async clients
                if hasattr(underlying, "chat") and hasattr(
                    underlying.chat, "completions"
                ):
                    if hasattr(underlying.chat.completions, "create"):
                        if inspect.iscoroutinefunction(
                            underlying.chat.completions.create
                        ):
                            return True

            # Check if this is an async client by checking for a coroutine method
            if hasattr(self.client, "chat") and hasattr(
                self.client.chat, "completions"
            ):
                if hasattr(self.client.chat.completions, "create"):
                    return inspect.iscoroutinefunction(
                        self.client.chat.completions.create
                    )

            # For instructor-wrapped clients, also check the closure of create_fn
            # This handles cases where the underlying client is stored in a closure
            if (
                hasattr(self.client, "create_fn")
                and hasattr(self.client.create_fn, "__closure__")
                and self.client.create_fn.__closure__
            ):
                for cell in self.client.create_fn.__closure__:
                    try:
                        obj = cell.cell_contents
                        # Check if the closure object is an async client
                        if hasattr(obj, "chat") and hasattr(obj.chat, "completions"):
                            if hasattr(obj.chat.completions, "create"):
                                if inspect.iscoroutinefunction(
                                    obj.chat.completions.create
                                ):
                                    return True
                        # Also check for acompletion (e.g., litellm Router)
                        if hasattr(obj, "acompletion"):
                            if inspect.iscoroutinefunction(obj.acompletion):
                                return True
                    except (ValueError, AttributeError):
                        # cell_contents might not be accessible
                        pass

            return False
        except (AttributeError, TypeError):
            return False

    def _run_async_in_current_loop(self, coro: t.Awaitable[t.Any]) -> t.Any:
        """Run an async coroutine in the current event loop if possible.

        This handles Jupyter environments correctly by using a separate thread
        when a running event loop is detected.
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()

            if loop.is_running():
                # If the loop is already running (like in Jupyter notebooks),
                # we run the coroutine in a separate thread with its own event loop
                result_container: t.Dict[str, t.Any] = {
                    "result": None,
                    "exception": None,
                }

                def run_in_thread():
                    # Create a new event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        # Run the coroutine in this thread's event loop
                        result_container["result"] = new_loop.run_until_complete(coro)
                    except Exception as e:
                        # Capture any exceptions to re-raise in the main thread
                        result_container["exception"] = e
                    finally:
                        # Clean up the event loop
                        new_loop.close()

                # Start the thread and wait for it to complete
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()

                # Re-raise any exceptions that occurred in the thread
                if result_container["exception"]:
                    raise result_container["exception"]

                return result_container["result"]
            else:
                # Standard case - event loop exists but isn't running
                return loop.run_until_complete(coro)

        except RuntimeError:
            # If we get a runtime error about no event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                # Clean up
                loop.close()
                asyncio.set_event_loop(None)

    def generate(
        self, prompt: str, response_model: t.Type[InstructorTypeVar]
    ) -> InstructorTypeVar:
        """Generate a response using the configured LLM.

        For async clients, this will run the async method in the appropriate event loop.
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        # If client is async, use the appropriate method to run it
        if self.is_async:
            result = self._run_async_in_current_loop(
                self.agenerate(prompt, response_model)
            )
        else:
            # Map parameters based on provider requirements
            provider_kwargs = self._map_provider_params()

            if self.provider.lower() == "google":
                result = self.client.create(
                    model=self.model,
                    messages=messages,
                    response_model=response_model,
                    **provider_kwargs,
                )
            else:
                # OpenAI, Anthropic, LiteLLM
                result = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_model=response_model,
                    **provider_kwargs,
                )

        # Track the usage
        track(
            LLMUsageEvent(
                provider=self.provider,
                model=self.model,
                llm_type="instructor",
                num_requests=1,
                is_async=self.is_async,
            )
        )
        return result

    async def agenerate(
        self,
        prompt: str,
        response_model: t.Type[InstructorTypeVar],
    ) -> InstructorTypeVar:
        """Asynchronously generate a response using the configured LLM."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        # If client is not async, raise a helpful error
        if not self.is_async:
            raise TypeError(
                "Cannot use agenerate() with a synchronous client. Use generate() instead."
            )

        # Map parameters based on provider requirements
        provider_kwargs = self._map_provider_params()

        if self.provider.lower() == "google":
            result = await self.client.create(
                model=self.model,
                messages=messages,
                response_model=response_model,
                **provider_kwargs,
            )
        else:
            # OpenAI, Anthropic, LiteLLM
            result = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_model=response_model,
                **provider_kwargs,
            )

        # Track the usage
        track(
            LLMUsageEvent(
                provider=self.provider,
                model=self.model,
                llm_type="instructor",
                num_requests=1,
                is_async=True,
            )
        )
        return result

    def _get_client_info(self) -> str:
        """Get client type and async status information."""
        client_type = self.client.__class__.__name__
        async_status = "async" if self.is_async else "sync"
        return f"<{client_type}:{async_status}>"

    def _get_key_config(self) -> str:
        """Get key configuration parameters as a string."""
        config_parts = []

        # Show important model arguments
        important_args = [
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ]
        for arg in important_args:
            if arg in self.model_args:
                config_parts.append(f"{arg}={self.model_args[arg]}")

        # Show count of other args if there are any
        other_args = len([k for k in self.model_args.keys() if k not in important_args])
        if other_args > 0:
            config_parts.append(f"+{other_args} more")

        return ", ".join(config_parts)

    def __repr__(self) -> str:
        """Return a detailed string representation of the LLM."""
        client_info = self._get_client_info()
        key_config = self._get_key_config()

        base_repr = f"InstructorLLM(provider='{self.provider}', model='{self.model}', client={client_info}"

        if key_config:
            base_repr += f", {key_config}"

        base_repr += ")"
        return base_repr

    __str__ = __repr__
