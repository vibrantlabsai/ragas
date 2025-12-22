"""HuggingFace embeddings implementation supporting both local and API-based models."""

import typing as t

from ragas.cache import CacheInterface

from .base import BaseRagasEmbedding
from .utils import batch_texts, run_sync_in_async, validate_texts


class HuggingFaceEmbeddings(BaseRagasEmbedding):
    """HuggingFace embeddings supporting both local and API-based models.

    Supports sentence-transformers for local models and HuggingFace API for
    hosted models. Provides efficient batch processing and caching.
    """

    PROVIDER_NAME = "huggingface"
    REQUIRES_MODEL = True

    def __init__(
        self,
        model: str,
        use_api: bool = False,
        api_key: t.Optional[str] = None,
        device: t.Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        cache: t.Optional[CacheInterface] = None,
        **model_kwargs: t.Any,
    ):
        super().__init__(cache=cache)
        self.model = model
        self.use_api = use_api
        self.api_key = api_key
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.model_kwargs = model_kwargs

        if use_api:
            self._setup_api_client()
        else:
            self._setup_local_model()

    def _setup_api_client(self):
        """Setup HuggingFace API client."""
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "HuggingFace API support requires huggingface-hub. "
                "Install with: pip install huggingface-hub"
            )

        self.client = InferenceClient(
            model=self.model,
            token=self.api_key,
        )

    def _setup_local_model(self):
        """Setup local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Local HuggingFace models require sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )

        self.model_instance = SentenceTransformer(
            self.model, device=self.device, **self.model_kwargs
        )

    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Embed a single text using HuggingFace."""
        if self.use_api:
            return self._embed_text_api(text, **kwargs)
        else:
            return self._embed_text_local(text, **kwargs)

    def _embed_text_api(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Embed text using HuggingFace API."""
        response = self.client.feature_extraction(text, **kwargs)
        # HuggingFace API returns nested list for single text
        if isinstance(response[0], list):
            return list(response[0])
        return list(response)

    def _embed_text_local(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Embed text using local sentence-transformers model."""
        embedding = self.model_instance.encode(
            text, normalize_embeddings=self.normalize_embeddings, **kwargs
        )
        return embedding.tolist()

    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Asynchronously embed a single text using HuggingFace."""
        if self.use_api:
            return await self._aembed_text_api(text, **kwargs)
        else:
            return await run_sync_in_async(self._embed_text_local, text, **kwargs)

    async def _aembed_text_api(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Asynchronously embed text using HuggingFace API."""
        # HuggingFace hub doesn't have native async support
        return await run_sync_in_async(self._embed_text_api, text, **kwargs)

    def embed_texts(self, texts: t.List[str], **kwargs: t.Any) -> t.List[t.List[float]]:
        """Embed multiple texts using HuggingFace with batching."""
        texts = validate_texts(texts)
        if not texts:
            return []

        if self.use_api:
            return self._embed_texts_api(texts, **kwargs)
        else:
            return self._embed_texts_local(texts, **kwargs)

    def _embed_texts_api(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        """Embed multiple texts using HuggingFace API with batching."""
        embeddings = []
        batches = batch_texts(texts, self.batch_size)

        for batch in batches:
            # HuggingFace API can handle batch processing
            batch_embeddings = []
            for text in batch:
                response = self.client.feature_extraction(text, **kwargs)
                if isinstance(response[0], list):
                    batch_embeddings.append(list(response[0]))
                else:
                    batch_embeddings.append(list(response))
            embeddings.extend(batch_embeddings)

        return embeddings

    def _embed_texts_local(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        """Embed multiple texts using local sentence-transformers model."""
        embeddings = self.model_instance.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            batch_size=self.batch_size,
            **kwargs,
        )
        return embeddings.tolist()

    async def aembed_texts(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        """Asynchronously embed multiple texts using HuggingFace."""
        texts = validate_texts(texts)
        if not texts:
            return []

        if self.use_api:
            return await run_sync_in_async(self._embed_texts_api, texts, **kwargs)
        else:
            return await run_sync_in_async(self._embed_texts_local, texts, **kwargs)

    def _get_client_info(self) -> str:
        """Get client type information."""
        if self.use_api:
            return "<HuggingFaceAPI>"
        else:
            return "<SentenceTransformer>"

    def _get_key_config(self) -> str:
        """Get key configuration parameters as a string."""
        config_parts = []

        config_parts.append(f"use_api={self.use_api}")

        if not self.use_api:
            if self.device:
                config_parts.append(f"device='{self.device}'")
            if not self.normalize_embeddings:
                config_parts.append(f"normalize_embeddings={self.normalize_embeddings}")

        if self.batch_size != 32:  # Only show if different from default
            config_parts.append(f"batch_size={self.batch_size}")

        # Show count of other model kwargs if there are any
        if self.model_kwargs:
            config_parts.append(f"+{len(self.model_kwargs)} model_kwargs")

        return ", ".join(config_parts)

    def __repr__(self) -> str:
        """Return a detailed string representation of the HuggingFace embeddings."""
        client_info = self._get_client_info()
        key_config = self._get_key_config()

        base_repr = f"HuggingFaceEmbeddings(provider='huggingface', model='{self.model}', client={client_info}"

        if key_config:
            base_repr += f", {key_config}"

        base_repr += ")"
        return base_repr

    __str__ = __repr__
