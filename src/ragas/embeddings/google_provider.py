"""Google embeddings implementation supporting both Vertex AI and Google AI (Gemini)."""

import sys
import typing as t

from .base import BaseRagasEmbedding
from .utils import run_sync_in_async, validate_texts


class GoogleEmbeddings(BaseRagasEmbedding):
    """Google embeddings using Vertex AI or Google AI (Gemini).

    Supports both Vertex AI and Google AI (Gemini) embedding models.
    For Vertex AI, requires google-cloud-aiplatform package.
    For Google AI, requires google-generativeai package.

    The client parameter is flexible:
    - For Gemini: Can be None (auto-imports genai), the genai module, or a GenerativeModel instance
    - For Vertex: Should be the configured vertex client

    Examples:
        # Gemini - auto-import (simplest)
        embeddings = GoogleEmbeddings(client=None, model="text-embedding-004")

        # Gemini - explicit genai module
        import google.generativeai as genai
        genai.configure(api_key="...")
        embeddings = GoogleEmbeddings(client=genai, model="text-embedding-004")

        # Gemini - from LLM client (auto-extracts genai module)
        llm_client = genai.GenerativeModel("gemini-2.0-flash")
        embeddings = GoogleEmbeddings(client=llm_client, model="text-embedding-004")
    """

    PROVIDER_NAME = "google"
    REQUIRES_CLIENT = False  # Client is optional for Gemini (can auto-import)
    DEFAULT_MODEL = "text-embedding-004"

    def __init__(
        self,
        client: t.Optional[t.Any] = None,
        model: str = "text-embedding-004",
        use_vertex: bool = False,
        project_id: t.Optional[str] = None,
        location: t.Optional[str] = "us-central1",
        **kwargs: t.Any,
    ):
        self._original_client = client
        self.model = model
        self.use_vertex = use_vertex
        self.project_id = project_id
        self.location = location
        self.kwargs = kwargs

        # Resolve the actual client to use
        self.client = self._resolve_client(client, use_vertex)

    def _resolve_client(self, client: t.Optional[t.Any], use_vertex: bool) -> t.Any:
        """Resolve the client to use for embeddings.

        For Vertex AI: Returns the client as-is (must be provided).
        For Gemini: Handles three scenarios:
            1. No client (None) - Auto-imports and returns genai module
            2. genai module - Returns as-is
            3. GenerativeModel instance - Extracts and returns genai module

        Args:
            client: The client provided by the user (can be None for Gemini)
            use_vertex: Whether using Vertex AI or Gemini

        Returns:
            The resolved client ready for use

        Raises:
            ValueError: If Vertex AI is used without a client, or if genai cannot be imported
        """
        if use_vertex:
            # Vertex AI requires an explicit client
            if client is None:
                raise ValueError(
                    "Vertex AI embeddings require a client. "
                    "Please provide a configured Vertex AI client."
                )
            return client

        # Gemini path - handle different client types
        if client is None:
            # Auto-import genai module
            return self._import_genai_module()

        # Check if client has embed_content method (it's the genai module or similar)
        if hasattr(client, "embed_content") and callable(
            getattr(client, "embed_content")
        ):
            return client

        # Check if it's a GenerativeModel instance - extract genai module from it
        client_module = client.__class__.__module__
        if "google.generativeai" in client_module or "google.genai" in client_module:
            # Extract base module name (google.generativeai or google.genai)
            if "google.generativeai" in client_module:
                base_module = "google.generativeai"
            else:
                base_module = "google.genai"

            # Try to get the module from sys.modules
            genai_module = sys.modules.get(base_module)
            if genai_module and hasattr(genai_module, "embed_content"):
                return genai_module

            # If not in sys.modules, try importing it
            try:
                import importlib

                genai_module = importlib.import_module(base_module)
                if hasattr(genai_module, "embed_content"):
                    return genai_module
            except ImportError:
                pass

        # If we couldn't resolve it, try importing genai as fallback
        return self._import_genai_module()

    def _import_genai_module(self) -> t.Any:
        """Import and return the google.generativeai module.

        Returns:
            The google.generativeai module

        Raises:
            ImportError: If google-generativeai is not installed
        """
        try:
            import google.generativeai as genai

            return genai
        except ImportError as e:
            raise ImportError(
                "Google AI (Gemini) embeddings require google-generativeai package. "
                "Install with: pip install google-generativeai"
            ) from e

    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Embed a single text using Google's embedding service."""
        if self.use_vertex:
            return self._embed_text_vertex(text, **kwargs)
        else:
            return self._embed_text_genai(text, **kwargs)

    def _embed_text_vertex(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Embed text using Vertex AI."""
        try:
            from vertexai.language_models import TextEmbeddingModel  # type: ignore
        except ImportError:
            raise ImportError(
                "Vertex AI support requires google-cloud-aiplatform. "
                "Install with: pip install google-cloud-aiplatform"
            )

        model = TextEmbeddingModel.from_pretrained(self.model)
        merged_kwargs = {**self.kwargs, **kwargs}
        embeddings = model.get_embeddings([text], **merged_kwargs)
        return embeddings[0].values

    def _embed_text_genai(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Embed text using Google AI (Gemini)."""
        merged_kwargs = {**self.kwargs, **kwargs}
        result = self.client.embed_content(
            model=f"models/{self.model}", content=text, **merged_kwargs
        )
        return result["embedding"]

    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Asynchronously embed a single text using Google's embedding service.

        Google's SDK doesn't provide native async support, so we use ThreadPoolExecutor.
        """
        return await run_sync_in_async(self.embed_text, text, **kwargs)

    def embed_texts(self, texts: t.List[str], **kwargs: t.Any) -> t.List[t.List[float]]:
        """Embed multiple texts using Google's embedding service."""
        texts = validate_texts(texts)
        if not texts:
            return []

        if self.use_vertex:
            return self._embed_texts_vertex(texts, **kwargs)
        else:
            return self._embed_texts_genai(texts, **kwargs)

    def _embed_texts_vertex(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        """Embed multiple texts using Vertex AI batch processing."""
        try:
            from vertexai.language_models import TextEmbeddingModel  # type: ignore
        except ImportError:
            raise ImportError(
                "Vertex AI support requires google-cloud-aiplatform. "
                "Install with: pip install google-cloud-aiplatform"
            )

        model = TextEmbeddingModel.from_pretrained(self.model)
        merged_kwargs = {**self.kwargs, **kwargs}
        embeddings = model.get_embeddings(texts, **merged_kwargs)
        return [emb.values for emb in embeddings]

    def _embed_texts_genai(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        """Embed multiple texts using Google AI (Gemini).

        Google AI doesn't support batch processing, so we process individually.
        """
        return [self._embed_text_genai(text, **kwargs) for text in texts]

    async def aembed_texts(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        """Asynchronously embed multiple texts using Google's embedding service."""
        texts = validate_texts(texts)
        if not texts:
            return []

        return await run_sync_in_async(self.embed_texts, texts, **kwargs)

    def _get_client_info(self) -> str:
        """Get client type information."""
        if self.use_vertex:
            return "<VertexAI>"
        else:
            client_type = self.client.__class__.__name__
            return f"<{client_type}>"

    def _get_key_config(self) -> str:
        """Get key configuration parameters as a string."""
        config_parts = []

        if self.use_vertex:
            config_parts.append(f"use_vertex={self.use_vertex}")
            if self.project_id:
                config_parts.append(f"project_id='{self.project_id}'")
            if self.location != "us-central1":
                config_parts.append(f"location='{self.location}'")
        else:
            config_parts.append(f"use_vertex={self.use_vertex}")

        return ", ".join(config_parts)

    def __repr__(self) -> str:
        """Return a detailed string representation of the Google embeddings."""
        client_info = self._get_client_info()
        key_config = self._get_key_config()

        base_repr = f"GoogleEmbeddings(provider='google', model='{self.model}', client={client_info}"

        if key_config:
            base_repr += f", {key_config}"

        base_repr += ")"
        return base_repr

    __str__ = __repr__
