import asyncio
import typing as t

import numpy as np

from ragas.cache import CacheInterface
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from haystack.components.embedders.azure_text_embedder import (
        AzureOpenAITextEmbedder,
    )
    from haystack.components.embedders.hugging_face_api_text_embedder import (
        HuggingFaceAPITextEmbedder,
    )
    from haystack.components.embedders.openai_text_embedder import (
        OpenAITextEmbedder,
    )
    from haystack.components.embedders.sentence_transformers_text_embedder import (
        SentenceTransformersTextEmbedder,
    )


class HaystackEmbeddingsWrapper(BaseRagasEmbeddings):
    """
    A wrapper for using Haystack embedders within the Ragas framework.

    This class allows you to use both synchronous and asynchronous methods
    (`embed_query`/`embed_documents` and `aembed_query`/`aembed_documents`)
    for generating embeddings through a Haystack embedder.

    Parameters
    ----------
    embedder : AzureOpenAITextEmbedder | HuggingFaceAPITextEmbedder | OpenAITextEmbedder | SentenceTransformersTextEmbedder
        An instance of a supported Haystack embedder class.
    run_config : RunConfig, optional
        A configuration object to manage embedding execution settings, by default None.
    cache : CacheInterface, optional
        A cache instance for storing and retrieving embedding results, by default None.
    """

    def __init__(
        self,
        embedder: t.Union[
            "AzureOpenAITextEmbedder",
            "HuggingFaceAPITextEmbedder",
            "OpenAITextEmbedder",
            "SentenceTransformersTextEmbedder",
        ],
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[CacheInterface] = None,
    ):
        super().__init__(cache=cache)

        # Lazy Import of required Haystack components
        try:
            from haystack import AsyncPipeline
            from haystack.components.embedders.azure_text_embedder import (
                AzureOpenAITextEmbedder,
            )
            from haystack.components.embedders.hugging_face_api_text_embedder import (
                HuggingFaceAPITextEmbedder,
            )
            from haystack.components.embedders.openai_text_embedder import (
                OpenAITextEmbedder,
            )
            from haystack.components.embedders.sentence_transformers_text_embedder import (
                SentenceTransformersTextEmbedder,
            )
        except ImportError as exc:
            raise ImportError(
                "Haystack is not installed. Please install it with `pip install haystack-ai`."
            ) from exc

        # Validate embedder type
        if not isinstance(
            embedder,
            (
                AzureOpenAITextEmbedder,
                HuggingFaceAPITextEmbedder,
                OpenAITextEmbedder,
                SentenceTransformersTextEmbedder,
            ),
        ):
            raise TypeError(
                "Expected 'embedder' to be one of: AzureOpenAITextEmbedder, "
                "HuggingFaceAPITextEmbedder, OpenAITextEmbedder, or "
                f"SentenceTransformersTextEmbedder, but got {type(embedder).__name__}."
            )

        self.embedder = embedder

        # Initialize an asynchronous pipeline and add the embedder component
        self.async_pipeline = AsyncPipeline()
        self.async_pipeline.add_component("embedder", self.embedder)  # type: ignore[reportArgumentType]

        # Set or create the run configuration
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def embed_query(self, text: str) -> t.List[float]:
        result = self.embedder.run(text=text)  # type: ignore[reportAttributeAccessIssue]
        embedding = result["embedding"]
        # Force conversion to float using NumPy's vectorized conversion.
        return t.cast(t.List[float], np.asarray(embedding, dtype=float).tolist())

    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        return [self.embed_query(text) for text in texts]

    async def aembed_query(self, text: str) -> t.List[float]:
        # Run the async pipeline with the input text
        output = await self.async_pipeline.run_async({"embedder": {"text": text}})
        return output.get("embedder", {}).get("embedding", [])

    async def aembed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        tasks = (self.aembed_query(text) for text in texts)
        results = await asyncio.gather(*tasks)
        return results

    def __repr__(self) -> str:
        try:
            from haystack.components.embedders.azure_text_embedder import (
                AzureOpenAITextEmbedder,
            )
            from haystack.components.embedders.hugging_face_api_text_embedder import (
                HuggingFaceAPITextEmbedder,
            )
            from haystack.components.embedders.openai_text_embedder import (
                OpenAITextEmbedder,
            )
            from haystack.components.embedders.sentence_transformers_text_embedder import (
                SentenceTransformersTextEmbedder,
            )
        except ImportError:
            return f"{self.__class__.__name__}(embeddings=Unknown(...))"

        if isinstance(
            self.embedder, (OpenAITextEmbedder, SentenceTransformersTextEmbedder)
        ):  # type: ignore
            model_info = self.embedder.model
        elif isinstance(self.embedder, AzureOpenAITextEmbedder):  # type: ignore
            model_info = self.embedder.azure_deployment
        elif isinstance(self.embedder, HuggingFaceAPITextEmbedder):  # type: ignore
            model_info = self.embedder.api_params
        else:
            model_info = "Unknown"

        return f"{self.__class__.__name__}(embeddings={model_info}(...))"
