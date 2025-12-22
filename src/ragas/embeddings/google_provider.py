"""Google embeddings implementation supporting both Vertex AI and Google AI (Gemini)."""

import sys
import typing as t

from ragas.cache import CacheInterface

from .base import BaseRagasEmbedding
from .utils import run_sync_in_async, validate_texts


class GoogleEmbeddings(BaseRagasEmbedding):
    """Google embeddings using Vertex AI or Google AI (Gemini).

    Supports both Vertex AI and Google AI (Gemini) embedding models.
    For Vertex AI, requires google-cloud-aiplatform package.
    For Google AI, supports both:
        - New SDK (google-genai): Recommended, uses genai.Client()
        - Old SDK (google-generativeai): Deprecated (support ends Aug 2025)

    The client parameter is flexible:
    - For new SDK: genai.Client(api_key="...") instance
    - For old SDK: None (auto-imports), the genai module, or a GenerativeModel instance
    - For Vertex: Should be the configured vertex client

    Note: Unlike LLM generation, embeddings work correctly with both SDKs.
    The known instructor safety settings issue (github.com/567-labs/instructor/issues/1658)
    only affects LLM generation, not embeddings.

    Examples:
        # New SDK (google-genai) - recommended
        from google import genai
        client = genai.Client(api_key="...")
        embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")

        # Old SDK (google-generativeai) - deprecated
        import google.generativeai as genai
        genai.configure(api_key="...")
        embeddings = GoogleEmbeddings(client=genai, model="text-embedding-004")

        # Auto-import (tries new SDK first, falls back to old)
        embeddings = GoogleEmbeddings(model="text-embedding-004")
    """

    PROVIDER_NAME = "google"
    REQUIRES_CLIENT = False  # Client is optional for Gemini (can auto-import)
    DEFAULT_MODEL = "gemini-embedding-001"

    def __init__(
        self,
        client: t.Optional[t.Any] = None,
        model: str = "gemini-embedding-001",
        use_vertex: bool = False,
        project_id: t.Optional[str] = None,
        location: t.Optional[str] = "us-central1",
        cache: t.Optional[CacheInterface] = None,
        **kwargs: t.Any,
    ):
        super().__init__(cache=cache)
        self._original_client = client
        self.model = model
        self.use_vertex = use_vertex
        self.project_id = project_id
        self.location = location
        self.kwargs = kwargs

        # Track which SDK is being used (new google-genai vs old google-generativeai)
        self._use_new_sdk = False

        # Resolve the actual client to use
        self.client = self._resolve_client(client, use_vertex)

    def _resolve_client(self, client: t.Optional[t.Any], use_vertex: bool) -> t.Any:
        """Resolve the client to use for embeddings.

        For Vertex AI: Returns the client as-is (must be provided).
        For Gemini: Handles multiple scenarios:
            - New SDK (google-genai): genai.Client() instance
            - Old SDK: None (auto-imports), genai module, or GenerativeModel instance

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

        # Check if it's the new google-genai SDK Client
        if client is not None and self._is_new_genai_client(client):
            self._use_new_sdk = True
            return client

        # Gemini path - handle different client types for old SDK
        if client is None:
            # Auto-import genai module (tries new SDK first, then old)
            return self._import_genai_module()

        # Check if client has embed_content method (it's the old genai module)
        if hasattr(client, "embed_content") and callable(
            getattr(client, "embed_content")
        ):
            self._use_new_sdk = False
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
                self._use_new_sdk = False
                return genai_module

            # If not in sys.modules, try importing it
            try:
                import importlib

                genai_module = importlib.import_module(base_module)
                if hasattr(genai_module, "embed_content"):
                    self._use_new_sdk = False
                    return genai_module
            except ImportError:
                pass

        # If we couldn't resolve it, try importing genai as fallback
        return self._import_genai_module()

    def _is_new_genai_client(self, client: t.Any) -> bool:
        """Check if client is from the new google-genai SDK.

        New SDK client is genai.Client() with client.models.embed_content() method.
        """
        client_module = getattr(client, "__module__", "") or ""
        client_class = client.__class__.__name__

        # New SDK: google.genai.client.Client
        if "google.genai" in client_module and "generativeai" not in client_module:
            # Verify it has the models.embed_content interface
            if hasattr(client, "models") and hasattr(client.models, "embed_content"):
                return True

        # Check class name as fallback
        if client_class == "Client" and hasattr(client, "models"):
            return True

        return False

    def _import_genai_module(self) -> t.Any:
        """Import and return the Google genai module.

        Tries new SDK (google-genai) first, falls back to old SDK (google-generativeai).

        Returns:
            The genai Client (new SDK) or module (old SDK)

        Raises:
            ImportError: If neither google-genai nor google-generativeai is installed
        """
        # Try new SDK first (google-genai)
        try:
            from google import genai  # type: ignore[attr-defined]

            # New SDK requires creating a Client instance
            client = genai.Client()
            self._use_new_sdk = True
            return client
        except ImportError:
            pass
        except Exception:
            # Client creation might fail without API key, fall back to old SDK
            pass

        # Fall back to old SDK (google-generativeai)
        try:
            import google.generativeai as genai  # type: ignore[import-untyped]

            self._use_new_sdk = False
            return genai
        except ImportError:
            pass

        raise ImportError(
            "Google AI (Gemini) embeddings require either:\n"
            "  - google-genai (recommended): pip install google-genai\n"
            "  - google-generativeai (deprecated): pip install google-generativeai"
        )

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
        """Embed text using Google AI (Gemini).

        Supports both new SDK (google-genai) and old SDK (google-generativeai).
        """
        merged_kwargs = {**self.kwargs, **kwargs}

        if self._use_new_sdk:
            # New SDK: client.models.embed_content(model="name", contents="text")
            result = self.client.models.embed_content(
                model=self.model, contents=text, **merged_kwargs
            )
            # New SDK returns result.embeddings[0].values
            return list(result.embeddings[0].values)
        else:
            # Old SDK: genai.embed_content(model="models/name", content="text")
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

        New SDK (google-genai) supports batch processing.
        Old SDK (google-generativeai) processes individually.
        """
        if self._use_new_sdk:
            # New SDK supports batch embedding
            merged_kwargs = {**self.kwargs, **kwargs}
            result = self.client.models.embed_content(
                model=self.model, contents=texts, **merged_kwargs
            )
            return [list(emb.values) for emb in result.embeddings]
        else:
            # Old SDK doesn't support batch processing
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
