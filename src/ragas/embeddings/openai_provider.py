import typing as t

from ragas._analytics import EmbeddingUsageEvent, track
from ragas.cache import CacheInterface

from .base import BaseRagasEmbedding
from .utils import validate_texts


class OpenAIEmbeddings(BaseRagasEmbedding):
    """OpenAI embeddings implementation with batch optimization.

    Supports both sync and async OpenAI clients with automatic detection.
    Provides optimized batch processing for better performance.
    """

    PROVIDER_NAME = "openai"
    REQUIRES_CLIENT = True
    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        client: t.Any,
        model: str = "text-embedding-3-small",
        cache: t.Optional[CacheInterface] = None,
    ):
        super().__init__(cache=cache)
        self.client = client
        self.model = model
        self.is_async = self._check_client_async(client)

    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Embed a single text using OpenAI.

        For async clients, this will run the async method in the appropriate event loop.
        """
        if self.is_async:
            result = self._run_async_in_current_loop(self.aembed_text(text, **kwargs))
        else:
            response = self.client.embeddings.create(
                input=text, model=self.model, **kwargs
            )
            result = response.data[0].embedding

        # Track usage
        track(
            EmbeddingUsageEvent(
                provider="openai",
                model=self.model,
                embedding_type="modern",
                num_requests=1,
                is_async=self.is_async,
            )
        )
        return result

    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Asynchronously embed a single text using OpenAI."""
        if not self.is_async:
            raise TypeError(
                "Cannot use aembed_text() with a synchronous client. Use embed_text() instead."
            )

        response = await self.client.embeddings.create(
            input=text, model=self.model, **kwargs
        )
        result = response.data[0].embedding

        # Track usage
        track(
            EmbeddingUsageEvent(
                provider="openai",
                model=self.model,
                embedding_type="modern",
                num_requests=1,
                is_async=True,
            )
        )
        return result

    def embed_texts(self, texts: t.List[str], **kwargs: t.Any) -> t.List[t.List[float]]:
        """Embed multiple texts using OpenAI's batch API for optimization."""
        texts = validate_texts(texts)
        if not texts:
            return []

        if self.is_async:
            result = self._run_async_in_current_loop(self.aembed_texts(texts, **kwargs))
        else:
            # OpenAI supports batch embedding natively
            response = self.client.embeddings.create(
                input=texts, model=self.model, **kwargs
            )
            result = [item.embedding for item in response.data]

        # Track usage
        track(
            EmbeddingUsageEvent(
                provider="openai",
                model=self.model,
                embedding_type="modern",
                num_requests=len(texts),
                is_async=self.is_async,
            )
        )
        return result

    async def aembed_texts(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        """Asynchronously embed multiple texts using OpenAI's batch API."""
        texts = validate_texts(texts)
        if not texts:
            return []

        if not self.is_async:
            raise TypeError(
                "Cannot use aembed_texts() with a synchronous client. Use embed_texts() instead."
            )

        response = await self.client.embeddings.create(
            input=texts, model=self.model, **kwargs
        )
        result = [item.embedding for item in response.data]

        # Track usage
        track(
            EmbeddingUsageEvent(
                provider="openai",
                model=self.model,
                embedding_type="modern",
                num_requests=len(texts),
                is_async=True,
            )
        )
        return result

    def _get_client_info(self) -> str:
        """Get client type and async status information."""
        client_type = self.client.__class__.__name__
        async_status = "async" if self.is_async else "sync"
        return f"<{client_type}:{async_status}>"

    def __repr__(self) -> str:
        """Return a detailed string representation of the OpenAI embeddings."""
        client_info = self._get_client_info()
        return f"OpenAIEmbeddings(provider='openai', model='{self.model}', client={client_info})"

    __str__ = __repr__
