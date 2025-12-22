"""LiteLLM embeddings implementation for universal provider support."""

import typing as t

from ragas.cache import CacheInterface

from .base import BaseRagasEmbedding
from .utils import batch_texts, get_optimal_batch_size, safe_import, validate_texts


class LiteLLMEmbeddings(BaseRagasEmbedding):
    """Universal embedding interface using LiteLLM.

    Supports 100+ models across OpenAI, Azure, Google, Cohere, Anthropic, and more.
    Provides intelligent batching and provider-specific optimizations.
    """

    PROVIDER_NAME = "litellm"
    REQUIRES_MODEL = True

    def __init__(
        self,
        model: str,
        api_key: t.Optional[str] = None,
        api_base: t.Optional[str] = None,
        api_version: t.Optional[str] = None,
        timeout: int = 600,
        max_retries: int = 3,
        batch_size: t.Optional[int] = None,
        cache: t.Optional[CacheInterface] = None,
        **litellm_params: t.Any,
    ):
        super().__init__(cache=cache)
        self.litellm = safe_import("litellm", "litellm")
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size or get_optimal_batch_size("litellm", model)
        self.litellm_params = litellm_params

    def _prepare_kwargs(self, **kwargs: t.Any) -> t.Dict[str, t.Any]:
        """Prepare kwargs for LiteLLM call."""
        call_kwargs = {
            "model": self.model,
            "timeout": self.timeout,
            "num_retries": self.max_retries,
            **self.litellm_params,
            **kwargs,
        }

        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        if self.api_base:
            call_kwargs["api_base"] = self.api_base
        if self.api_version:
            call_kwargs["api_version"] = self.api_version

        return call_kwargs

    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Embed a single text using LiteLLM."""
        call_kwargs = self._prepare_kwargs(**kwargs)
        response = self.litellm.embedding(input=[text], **call_kwargs)
        return response.data[0]["embedding"]

    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Asynchronously embed a single text using LiteLLM."""
        call_kwargs = self._prepare_kwargs(**kwargs)
        response = await self.litellm.aembedding(input=[text], **call_kwargs)
        return response.data[0]["embedding"]

    def embed_texts(self, texts: t.List[str], **kwargs: t.Any) -> t.List[t.List[float]]:
        """Embed multiple texts using LiteLLM with intelligent batching."""
        texts = validate_texts(texts)
        if not texts:
            return []

        embeddings = []
        batches = batch_texts(texts, self.batch_size)

        for batch in batches:
            call_kwargs = self._prepare_kwargs(**kwargs)
            response = self.litellm.embedding(input=batch, **call_kwargs)
            embeddings.extend([item["embedding"] for item in response.data])

        return embeddings

    async def aembed_texts(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        """Asynchronously embed multiple texts using LiteLLM with intelligent batching."""
        texts = validate_texts(texts)
        if not texts:
            return []

        embeddings = []
        batches = batch_texts(texts, self.batch_size)

        for batch in batches:
            call_kwargs = self._prepare_kwargs(**kwargs)
            response = await self.litellm.aembedding(input=batch, **call_kwargs)
            embeddings.extend([item["embedding"] for item in response.data])

        return embeddings

    def _get_key_config(self) -> str:
        """Get key configuration parameters as a string."""
        config_parts = []

        if self.api_base:
            config_parts.append(f"api_base='{self.api_base}'")

        if self.batch_size != 10:  # Only show if different from default
            config_parts.append(f"batch_size={self.batch_size}")

        if self.timeout != 600:  # Only show if different from default
            config_parts.append(f"timeout={self.timeout}")

        if self.max_retries != 3:  # Only show if different from default
            config_parts.append(f"max_retries={self.max_retries}")

        # Show count of other litellm params if there are any
        if self.litellm_params:
            config_parts.append(f"+{len(self.litellm_params)} litellm_params")

        return ", ".join(config_parts)

    def __repr__(self) -> str:
        """Return a detailed string representation of the LiteLLM embeddings."""
        key_config = self._get_key_config()

        base_repr = f"LiteLLMEmbeddings(provider='litellm', model='{self.model}'"

        if key_config:
            base_repr += f", {key_config}"

        base_repr += ")"
        return base_repr

    __str__ = __repr__
