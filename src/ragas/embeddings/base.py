from __future__ import annotations

import asyncio
import inspect
import typing as t
import warnings
from abc import ABC, abstractmethod
from dataclasses import field

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic.dataclasses import dataclass
from pydantic_core import CoreSchema, core_schema

from ragas._analytics import EmbeddingUsageEvent, track
from ragas.cache import CacheInterface, cacher
from ragas.embeddings.utils import run_async_in_current_loop, validate_texts
from ragas.run_config import RunConfig, add_async_retry, add_retry

if t.TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from pydantic import GetCoreSchemaHandler


DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class BaseRagasEmbedding(ABC):
    """Modern abstract base class for Ragas embedding implementations.

    This class provides a consistent interface for embedding text using various
    providers. Implementations should provide both sync and async methods for
    embedding single texts, with batch methods automatically provided.
    """

    def __init__(self, cache: t.Optional[CacheInterface] = None):
        """Initialize embedding with optional caching.

        Args:
            cache: Optional cache backend for caching embeddings.
                Use DiskCacheBackend() for persistent caching.
        """
        self.cache = cache

        if self.cache is not None:
            self.embed_text = cacher(cache_backend=self.cache)(self.embed_text)
            self.aembed_text = cacher(cache_backend=self.cache)(self.aembed_text)

    @abstractmethod
    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Embed a single text.

        Args:
            text: The text to embed
            **kwargs: Additional arguments for the embedding call

        Returns:
            List of floats representing the embedding
        """
        pass

    @abstractmethod
    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Asynchronously embed a single text.

        Args:
            text: The text to embed
            **kwargs: Additional arguments for the embedding call

        Returns:
            List of floats representing the embedding
        """
        pass

    def embed_texts(self, texts: t.List[str], **kwargs: t.Any) -> t.List[t.List[float]]:
        """Embed multiple texts.

        Default implementation processes texts individually. Override for
        batch optimization.

        Args:
            texts: List of texts to embed
            **kwargs: Additional arguments for the embedding calls

        Returns:
            List of embeddings, one for each input text
        """
        texts = validate_texts(texts)
        return [self.embed_text(text, **kwargs) for text in texts]

    async def aembed_texts(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        """Asynchronously embed multiple texts.

        Default implementation processes texts concurrently. Override for
        batch optimization.

        Args:
            texts: List of texts to embed
            **kwargs: Additional arguments for the embedding calls

        Returns:
            List of embeddings, one for each input text
        """
        texts = validate_texts(texts)
        tasks = [self.aembed_text(text, **kwargs) for text in texts]
        return await asyncio.gather(*tasks)

    def _check_client_async(
        self, client: t.Any, method_path: str = "embeddings.create"
    ) -> bool:
        """Check if a client supports async operations.

        Args:
            client: The client to check
            method_path: Dot-separated path to the method to check

        Returns:
            True if the client supports async operations
        """
        try:
            obj = client
            for attr in method_path.split("."):
                obj = getattr(obj, attr)
            return inspect.iscoroutinefunction(obj)
        except (AttributeError, TypeError):
            return False

    def _run_async_in_current_loop(self, coro):
        """Run an async coroutine in the current event loop if possible.

        This handles Jupyter environments correctly by using a separate thread
        when a running event loop is detected.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine
        """
        return run_async_in_current_loop(coro)

    @classmethod
    def _from_factory(
        cls,
        model: t.Optional[str] = None,
        client: t.Optional[t.Any] = None,
        **kwargs: t.Any,
    ) -> "BaseRagasEmbedding":
        """Create an embedding instance from factory parameters with validation.

        This base implementation handles common validation patterns. Individual
        providers can override this for custom initialization logic.
        """
        # Validate client requirement
        if getattr(cls, "REQUIRES_CLIENT", False) and not client:
            provider_name = getattr(cls, "PROVIDER_NAME", cls.__name__)
            raise ValueError(f"{provider_name} provider requires a client instance")

        # Validate model requirement
        if getattr(cls, "REQUIRES_MODEL", False) and not model:
            provider_name = getattr(cls, "PROVIDER_NAME", cls.__name__)
            raise ValueError(f"{provider_name} provider requires a model name")

        # Use default model if available and not provided
        if not model:
            model = getattr(cls, "DEFAULT_MODEL", None)

        # Construct instance - let providers handle their own parameters
        # Build constructor arguments based on provider requirements
        init_kwargs = kwargs.copy()
        if model is not None:
            init_kwargs["model"] = model
        if getattr(cls, "REQUIRES_CLIENT", False) and client is not None:
            init_kwargs["client"] = client

        return cls(**init_kwargs)


class BaseRagasEmbeddings(Embeddings, ABC):
    """
    Abstract base class for Ragas embeddings.

    This class extends the Embeddings class and provides methods for embedding
    text and managing run configurations.

    Attributes:
        run_config (RunConfig): Configuration for running the embedding operations.

    """

    run_config: RunConfig
    cache: t.Optional[CacheInterface] = None

    def __init__(self, cache: t.Optional[CacheInterface] = None):
        super().__init__()
        self.cache = cache
        if self.cache is not None:
            self.embed_query = cacher(cache_backend=self.cache)(self.embed_query)
            self.embed_documents = cacher(cache_backend=self.cache)(
                self.embed_documents
            )
            self.aembed_query = cacher(cache_backend=self.cache)(self.aembed_query)
            self.aembed_documents = cacher(cache_backend=self.cache)(
                self.aembed_documents
            )

    async def embed_text(self, text: str, is_async=True) -> t.List[float]:
        """
        Embed a single text string.
        """
        embs = await self.embed_texts([text], is_async=is_async)
        return embs[0]

    async def embed_texts(
        self, texts: t.List[str], is_async: bool = True
    ) -> t.List[t.List[float]]:
        """
        Embed multiple texts.
        """
        if is_async:
            aembed_documents_with_retry = add_async_retry(
                self.aembed_documents, self.run_config
            )
            return await aembed_documents_with_retry(texts)
        else:
            loop = asyncio.get_event_loop()
            embed_documents_with_retry = add_retry(
                self.embed_documents, self.run_config
            )
            return await loop.run_in_executor(None, embed_documents_with_retry, texts)

    @abstractmethod
    async def aembed_query(self, text: str) -> t.List[float]: ...

    @abstractmethod
    async def aembed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]: ...

    def set_run_config(self, run_config: RunConfig):
        """
        Set the run configuration for the embedding operations.
        """
        self.run_config = run_config

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: t.Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """
        Define how Pydantic generates a schema for BaseRagasEmbeddings.
        """
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.is_instance_schema(cls),  # The validator function
        )


class LangchainEmbeddingsWrapper(BaseRagasEmbeddings):
    """
    Wrapper for any embeddings from langchain.

    # TODO: Revisit deprecation warning
    # .. deprecated::
    #     LangchainEmbeddingsWrapper is deprecated and will be removed in a future version.
    #     Use the modern embedding providers directly with embedding_factory() instead:
    #
    #     # Instead of:
    #     # embedder = LangchainEmbeddingsWrapper(langchain_embeddings)
    #
    #     # Use:
    #     # embedder = embedding_factory("openai", model="text-embedding-3-small", client=openai_client)
    #     # embedder = embedding_factory("huggingface", model="sentence-transformers/all-MiniLM-L6-v2")
    #     # embedder = embedding_factory("google", client=vertex_client)
    """

    def __init__(
        self,
        embeddings: Embeddings,
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[CacheInterface] = None,
    ):
        warnings.warn(
            "LangchainEmbeddingsWrapper is deprecated and will be removed in a future version. "
            "Use the modern embedding providers instead: "
            "embedding_factory('openai', model='text-embedding-3-small', client=openai_client) "
            "or from ragas.embeddings import OpenAIEmbeddings, GoogleEmbeddings, HuggingFaceEmbeddings",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(cache=cache)
        self.embeddings = embeddings
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def embed_query(self, text: str) -> t.List[float]:
        """
        Embed a single query text.
        """
        result = self.embeddings.embed_query(text)

        # Track usage
        track(
            EmbeddingUsageEvent(
                provider="langchain",
                model=getattr(self.embeddings, "model", None),
                embedding_type="legacy",
                num_requests=1,
                is_async=False,
            )
        )
        return result

    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        """
        Embed multiple documents.
        """
        result = self.embeddings.embed_documents(texts)

        # Track usage
        track(
            EmbeddingUsageEvent(
                provider="langchain",
                model=getattr(self.embeddings, "model", None),
                embedding_type="legacy",
                num_requests=len(texts),
                is_async=False,
            )
        )
        return result

    async def aembed_query(self, text: str) -> t.List[float]:
        """
        Asynchronously embed a single query text.
        """
        result = await self.embeddings.aembed_query(text)

        # Track usage
        track(
            EmbeddingUsageEvent(
                provider="langchain",
                model=getattr(self.embeddings, "model", None),
                embedding_type="legacy",
                num_requests=1,
                is_async=True,
            )
        )
        return result

    async def aembed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        """
        Asynchronously embed multiple documents.
        """
        result = await self.embeddings.aembed_documents(texts)

        # Track usage
        track(
            EmbeddingUsageEvent(
                provider="langchain",
                model=getattr(self.embeddings, "model", None),
                embedding_type="legacy",
                num_requests=len(texts),
                is_async=True,
            )
        )
        return result

    def set_run_config(self, run_config: RunConfig):
        """
        Set the run configuration for the embedding operations.
        """
        self.run_config = run_config

        # run configurations specially for OpenAI
        if isinstance(self.embeddings, OpenAIEmbeddings):
            try:
                from openai import RateLimitError
            except ImportError:
                raise ImportError(
                    "openai.error.RateLimitError not found. Please install openai package as `pip install openai`"
                )
            self.embeddings.request_timeout = run_config.timeout
            self.run_config.exception_types = RateLimitError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(embeddings={self.embeddings.__class__.__name__}(...))"


@dataclass
class HuggingfaceEmbeddings(BaseRagasEmbeddings):
    """
    Hugging Face embeddings class for generating embeddings using pre-trained models.

    This class provides functionality to load and use Hugging Face models for
    generating embeddings of text inputs.

    Parameters
    ----------
    model_name : str, optional
        Name of the pre-trained model to use, by default DEFAULT_MODEL_NAME.
    cache_folder : str, optional
        Path to store downloaded models. Can also be set by SENTENCE_TRANSFORMERS_HOME
        environment variable.
    model_kwargs : dict, optional
        Additional keyword arguments to pass to the model.
    encode_kwargs : dict, optional
        Additional keyword arguments to pass to the encoding method.

    Attributes
    ----------
    model : Union[SentenceTransformer, CrossEncoder]
        The loaded Hugging Face model.
    is_cross_encoder : bool
        Flag indicating whether the model is a cross-encoder.

    Methods
    -------
    embed_query(text)
        Embed a single query text.
    embed_documents(texts)
        Embed multiple documents.
    predict(texts)
        Make predictions using a cross-encoder model.

    Notes
    -----
    This class requires the `sentence_transformers` and `transformers` packages
    to be installed.

    Examples
    --------
    >>> embeddings = HuggingfaceEmbeddings(model_name="bert-base-uncased")
    >>> query_embedding = embeddings.embed_query("What is the capital of France?")
    >>> doc_embeddings = embeddings.embed_documents(["Paris is the capital of France.", "London is the capital of the UK."])
    """

    model_name: str = DEFAULT_MODEL_NAME
    cache_folder: t.Optional[str] = None
    model_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    encode_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    cache: t.Optional[CacheInterface] = None

    def __post_init__(self):
        """
        Initialize the model after the object is created.
        """
        super().__init__(cache=self.cache)
        try:
            import sentence_transformers
            from transformers import AutoConfig  # type: ignore
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc
        config = AutoConfig.from_pretrained(self.model_name)
        self.is_cross_encoder = bool(
            np.intersect1d(
                list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values()),
                config.architectures or [],
            ).size
            != 0
        )

        if self.is_cross_encoder:
            self.model = sentence_transformers.CrossEncoder(
                self.model_name, **self.model_kwargs
            )
        else:
            self.model = sentence_transformers.SentenceTransformer(  # type: ignore
                self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
            )
        # ensure outputs are tensors
        if "convert_to_tensor" not in self.encode_kwargs:
            self.encode_kwargs["convert_to_tensor"] = True

        if self.cache is not None:
            self.predict = cacher(cache_backend=self.cache)(self.predict)

    def embed_query(self, text: str) -> t.List[float]:
        """
        Embed a single query text.
        """
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        """
        Embed multiple documents.
        """
        from sentence_transformers.SentenceTransformer import SentenceTransformer
        from torch import Tensor

        assert isinstance(self.model, SentenceTransformer), (
            "Model is not of the type Bi-encoder"
        )
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, **self.encode_kwargs
        )

        assert isinstance(embeddings, Tensor)
        return embeddings.tolist()

    def predict(self, texts: t.List[t.List[str]]) -> t.List[t.List[float]]:
        """
        Make predictions using a cross-encoder model.
        """
        from sentence_transformers.cross_encoder import CrossEncoder
        from torch import Tensor

        assert isinstance(self.model, CrossEncoder), (
            "Model is not of the type CrossEncoder"
        )

        predictions = self.model.predict(texts, **self.encode_kwargs)

        assert isinstance(predictions, Tensor)
        return predictions.tolist()


class LlamaIndexEmbeddingsWrapper(BaseRagasEmbeddings):
    """
    Wrapper for any embeddings from llama-index.

    # TODO: Revisit deprecation warning
    # .. deprecated::
    #     LlamaIndexEmbeddingsWrapper is deprecated and will be removed in a future version.
    #     Use the modern embedding providers directly with embedding_factory() instead:
    #
    #     # Instead of:
    #     # embedder = LlamaIndexEmbeddingsWrapper(llama_index_embeddings)
    #
    #     # Use:
    #     # embedder = embedding_factory("openai", model="text-embedding-3-small", client=openai_client)
    #     # embedder = embedding_factory("huggingface", model="sentence-transformers/all-MiniLM-L6-v2")
    #     # embedder = embedding_factory("google", client=vertex_client)

    This class provides a wrapper for llama-index embeddings, allowing them to be used
    within the Ragas framework. It supports both synchronous and asynchronous embedding
    operations for queries and documents.

    Parameters
    ----------
    embeddings : BaseEmbedding
        The llama-index embedding model to be wrapped.
    run_config : RunConfig, optional
        Configuration for the run. If not provided, a default RunConfig will be used.

    Attributes
    ----------
    embeddings : BaseEmbedding
        The wrapped llama-index embedding model.

    Examples
    --------
    >>> from llama_index.embeddings import OpenAIEmbedding
    >>> from ragas.embeddings import LlamaIndexEmbeddingsWrapper
    >>> llama_embeddings = OpenAIEmbedding()
    >>> wrapped_embeddings = LlamaIndexEmbeddingsWrapper(llama_embeddings)
    >>> query_embedding = wrapped_embeddings.embed_query("What is the capital of France?")
    >>> document_embeddings = wrapped_embeddings.embed_documents(["Paris is the capital of France.", "London is the capital of the UK."])
    """

    def __init__(
        self,
        embeddings: BaseEmbedding,
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[CacheInterface] = None,
    ):
        warnings.warn(
            "LlamaIndexEmbeddingsWrapper is deprecated and will be removed in a future version. "
            "Use the modern embedding providers instead: "
            "embedding_factory('openai', model='text-embedding-3-small', client=openai_client) "
            "or from ragas.embeddings import OpenAIEmbeddings, GoogleEmbeddings, HuggingFaceEmbeddings",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(cache=cache)
        self.embeddings = embeddings
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def embed_query(self, text: str) -> t.List[float]:
        return self.embeddings.get_query_embedding(text)

    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        return self.embeddings.get_text_embedding_batch(texts)

    async def aembed_query(self, text: str) -> t.List[float]:
        return await self.embeddings.aget_query_embedding(text)

    async def aembed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        return await self.embeddings.aget_text_embedding_batch(texts)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(embeddings={self.embeddings.__class__.__name__}(...))"


def _infer_embedding_provider_from_llm(llm: t.Any) -> str:
    """
    Infer the embedding provider from an LLM instance.

    This function attempts to extract the provider information from an LLM object
    to allow intelligent default selection of matching embedding providers.

    Parameters
    ----------
    llm : Any
        The LLM instance to extract provider information from.

    Returns
    -------
    str
        The inferred provider name, defaults to "openai" if unable to determine.
    """
    if llm is None:
        return "openai"

    # Check for InstructorLLM with provider attribute
    if hasattr(llm, "provider"):
        provider = getattr(llm, "provider", "").lower()
        if provider:
            return provider

    # Check for other LLM types
    llm_class_name = llm.__class__.__name__.lower()

    # Map common LLM class patterns to providers
    provider_mapping = {
        "anthropic": "anthropic",
        "claude": "anthropic",
        "gemini": "google",
        "google": "google",
        "vertex": "google",
        "groq": "groq",
        "mistral": "mistral",
        "cohere": "cohere",
        "openai": "openai",
        "azure": "azure",
    }

    for pattern, provider_name in provider_mapping.items():
        if pattern in llm_class_name:
            return provider_name

    # Default to OpenAI if unable to determine
    return "openai"


def embedding_factory(
    provider: str = "openai",
    model: t.Optional[str] = None,
    run_config: t.Optional[RunConfig] = None,
    client: t.Optional[t.Any] = None,
    interface: str = "auto",
    base_url: t.Optional[str] = None,
    cache: t.Optional[CacheInterface] = None,
    **kwargs: t.Any,
) -> t.Union[BaseRagasEmbeddings, BaseRagasEmbedding]:
    """
    Create and return an embeddings instance. Unified factory supporting both legacy and modern interfaces.

    This factory function automatically detects whether to use legacy or modern interfaces based on
    the parameters provided, while maintaining full backward compatibility.

    Parameters
    ----------
    provider : str, optional
        Provider name or provider/model string (e.g., "openai", "openai/text-embedding-3-small").
        For backward compatibility, also accepts model names directly.
        Default is "openai".
    model : str, optional
        The embedding model name. If not provided, uses provider defaults.
        For legacy calls, defaults to "text-embedding-ada-002".
    run_config : RunConfig, optional
        Configuration for the run, by default None.
    client : Any, optional
        Pre-initialized client for modern providers. When provided, uses modern interface.
    interface : str, optional
        Interface type: "legacy", "modern", or "auto" (default).
        "auto" detects based on parameters.
    base_url : str, optional
        Base URL for the API, by default None.
    cache : CacheInterface, optional
        Optional cache backend for caching embeddings.
        Use DiskCacheBackend() for persistent caching across runs.
        Saves costs and speeds up repeated embedding calls.
    **kwargs : Any
        Additional provider-specific arguments.

    Returns
    -------
    BaseRagasEmbeddings or BaseRagasEmbedding
        An instance of the requested embedding interface.

    Examples
    --------
    # Legacy usage (backward compatible)
    embedder = embedding_factory()
    embedder = embedding_factory("text-embedding-ada-002")

    # Modern usage
    embedder = embedding_factory("openai", "text-embedding-3-small", client=openai_client)
    embedder = embedding_factory("huggingface", "sentence-transformers/all-MiniLM-L6-v2")
    embedder = embedding_factory("google", client=vertex_client, project_id="my-project")

    # With caching
    from ragas.cache import DiskCacheBackend
    cache = DiskCacheBackend()
    embedder = embedding_factory("openai", client=openai_client, cache=cache)
    """
    # Detect if this is a legacy call for backward compatibility
    is_legacy_call = _is_legacy_embedding_call(provider, model, client, interface)

    if is_legacy_call:
        import warnings

        warnings.warn(
            "Legacy embedding_factory interface is deprecated and will be removed in a future version. "
            "Use the modern interface with explicit provider and client parameters: "
            "embedding_factory('openai', model='text-embedding-3-small', client=openai_client) "
            "or import providers directly: from ragas.embeddings import OpenAIEmbeddings, GoogleEmbeddings, HuggingFaceEmbeddings",
            DeprecationWarning,
            stacklevel=2,
        )
        # Legacy interface - treat provider as model name if it looks like a model
        model_name = (
            provider
            if _looks_like_model_name(provider)
            else (model or "text-embedding-ada-002")
        )
        openai_embeddings = OpenAIEmbeddings(model=model_name, base_url=base_url)
        if run_config is not None:
            openai_embeddings.request_timeout = run_config.timeout
        else:
            run_config = RunConfig()
        result = LangchainEmbeddingsWrapper(openai_embeddings, run_config=run_config)

        # Track factory usage (legacy)
        track(
            EmbeddingUsageEvent(
                provider="openai",
                model=model_name,
                embedding_type="factory_legacy",
                num_requests=1,
                is_async=False,
            )
        )
        return result

    # Modern interface - pass base_url and cache through kwargs for modern providers
    if base_url is not None:
        kwargs["base_url"] = base_url
    if cache is not None:
        kwargs["cache"] = cache
    result = _create_modern_embedding(provider, model, client, **kwargs)

    # Track factory usage (modern)
    track(
        EmbeddingUsageEvent(
            provider=provider,
            model=model,
            embedding_type="factory_modern",
            num_requests=1,
            is_async=False,
        )
    )
    return result


def _is_legacy_embedding_call(
    provider: str, model: t.Optional[str], client: t.Optional[t.Any], interface: str
) -> bool:
    """Detect if this is a legacy embedding factory call for backward compatibility."""
    # Explicit interface choice takes precedence
    if interface in ("legacy", "modern"):
        return interface == "legacy"

    # Auto-detection: legacy if no client AND (looks like model name OR is openai)
    return client is None and (_looks_like_model_name(provider) or provider == "openai")


# Model name patterns for backward compatibility detection
_LEGACY_MODEL_PATTERNS = {"text-embedding", "ada", "davinci", "gpt", "curie", "babbage"}


def _looks_like_model_name(name: str) -> bool:
    """Check if a string looks like an OpenAI model name rather than a provider name."""
    return any(pattern in name.lower() for pattern in _LEGACY_MODEL_PATTERNS)


def _get_provider_registry() -> t.Dict[str, t.Type[BaseRagasEmbedding]]:
    """Auto-discover available provider classes and build a registry.

    Returns:
        Dictionary mapping provider names to their classes.
    """
    from .google_provider import GoogleEmbeddings
    from .huggingface_provider import HuggingFaceEmbeddings
    from .litellm_provider import LiteLLMEmbeddings
    from .openai_provider import OpenAIEmbeddings

    providers = [
        OpenAIEmbeddings,
        GoogleEmbeddings,
        LiteLLMEmbeddings,
        HuggingFaceEmbeddings,
    ]

    return {
        cls.PROVIDER_NAME: cls for cls in providers if hasattr(cls, "PROVIDER_NAME")
    }


def _create_modern_embedding(
    provider: str, model: t.Optional[str], client: t.Optional[t.Any], **kwargs: t.Any
) -> BaseRagasEmbedding:
    """Create a modern embedding instance based on the provider."""
    cache = kwargs.pop("cache", None)

    # Handle provider/model string format
    if "/" in provider and model is None:
        provider_name, model_name = provider.split("/", 1)
        provider = provider_name
        model = model_name

    # Get provider registry and find the class
    registry = _get_provider_registry()
    provider_cls = registry.get(provider.lower())

    if not provider_cls:
        available = ", ".join(registry.keys())
        raise ValueError(
            f"Unsupported provider: {provider}. Supported providers: {available}"
        )

    # Let the provider class validate and construct itself
    return provider_cls._from_factory(model=model, client=client, cache=cache, **kwargs)


def modern_embedding_factory(
    provider: str,
    model: t.Optional[str] = None,
    client: t.Optional[t.Any] = None,
    **kwargs: t.Any,
) -> BaseRagasEmbedding:
    """
    Factory function to create a modern embedding instance based on the provider.

    DEPRECATED: Use embedding_factory() with interface="modern" or client parameter instead.
    This function is kept for backward compatibility and will be removed in a future version.

    Args:
        provider (str): The name of the embedding provider or provider/model string.
        model (str, optional): The model name to use for embeddings.
        client (Any, optional): Pre-initialized client for the provider.
        **kwargs: Additional arguments for the provider.

    Returns:
        BaseRagasEmbedding: An instance of the specified embedding provider.
    """
    result = embedding_factory(
        provider=provider, model=model, client=client, interface="modern", **kwargs
    )
    # Type narrowing: modern interface always returns BaseRagasEmbedding
    assert isinstance(result, BaseRagasEmbedding), (
        "Modern interface should always return BaseRagasEmbedding"
    )
    return result
