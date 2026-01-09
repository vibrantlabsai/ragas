import typing as t

from langchain_core.callbacks import Callbacks
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue

from ragas.cache import CacheInterface
from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from haystack.components.generators.azure import AzureOpenAIGenerator
    from haystack.components.generators.hugging_face_api import (
        HuggingFaceAPIGenerator,
    )
    from haystack.components.generators.hugging_face_local import (
        HuggingFaceLocalGenerator,
    )
    from haystack.components.generators.openai import OpenAIGenerator


class HaystackLLMWrapper(BaseRagasLLM):
    """
    A wrapper class for using Haystack LLM generators within the Ragas framework.

    This class integrates Haystack's LLM components (e.g., `OpenAIGenerator`,
    `HuggingFaceAPIGenerator`, etc.) into Ragas, enabling both synchronous and
    asynchronous text generation.

    Parameters
    ----------
    haystack_generator : AzureOpenAIGenerator | HuggingFaceAPIGenerator | HuggingFaceLocalGenerator | OpenAIGenerator
        An instance of a Haystack generator.
    run_config : RunConfig, optional
        Configuration object to manage LLM execution settings, by default None.
    cache : CacheInterface, optional
        A cache instance for storing results, by default None.
    """

    def __init__(
        self,
        haystack_generator: t.Union[
            "AzureOpenAIGenerator",
            "HuggingFaceAPIGenerator",
            "HuggingFaceLocalGenerator",
            "OpenAIGenerator",
        ],
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[CacheInterface] = None,
    ):
        super().__init__(cache=cache)

        # Lazy Import of required Haystack components
        try:
            from haystack import AsyncPipeline
            from haystack.components.generators.azure import AzureOpenAIGenerator
            from haystack.components.generators.hugging_face_api import (
                HuggingFaceAPIGenerator,
            )
            from haystack.components.generators.hugging_face_local import (
                HuggingFaceLocalGenerator,
            )
            from haystack.components.generators.openai import OpenAIGenerator
        except ImportError as exc:
            raise ImportError(
                "Haystack is not installed. Please install it using `pip install haystack-ai`."
            ) from exc

        # Validate haystack_generator type
        if not isinstance(
            haystack_generator,
            (
                AzureOpenAIGenerator,
                HuggingFaceAPIGenerator,
                HuggingFaceLocalGenerator,
                OpenAIGenerator,
            ),
        ):
            raise TypeError(
                "Expected 'haystack_generator' to be one of: "
                "AzureOpenAIGenerator, HuggingFaceAPIGenerator, "
                "HuggingFaceLocalGenerator, or OpenAIGenerator, but received "
                f"{type(haystack_generator).__name__}."
            )

        # Set up Haystack pipeline and generator
        self.generator = haystack_generator
        self.async_pipeline = AsyncPipeline()
        self.async_pipeline.add_component("llm", self.generator)  # type: ignore[reportArgumentType]

        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def is_finished(self, response: LLMResult) -> bool:
        return True

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        component_output: t.Dict[str, t.Any] = self.generator.run(prompt.to_string())  # type: ignore[reportAttributeAccessIssue]
        replies = component_output.get("llm", {}).get("replies", [])
        output_text = replies[0] if replies else ""

        return LLMResult(generations=[[Generation(text=output_text)]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        # Prepare input parameters for the LLM component
        llm_input = {
            "prompt": prompt.to_string(),
            "generation_kwargs": {"temperature": temperature},
        }

        # Run the async pipeline with the LLM input
        pipeline_output = await self.async_pipeline.run_async(data={"llm": llm_input})
        replies = pipeline_output.get("llm", {}).get("replies", [])
        output_text = replies[0] if replies else ""

        return LLMResult(generations=[[Generation(text=output_text)]])

    def __repr__(self) -> str:
        try:
            from haystack.components.generators.azure import AzureOpenAIGenerator
            from haystack.components.generators.hugging_face_api import (
                HuggingFaceAPIGenerator,
            )
            from haystack.components.generators.hugging_face_local import (
                HuggingFaceLocalGenerator,
            )
            from haystack.components.generators.openai import OpenAIGenerator
        except ImportError:
            return f"{self.__class__.__name__}(llm=Unknown(...))"

        generator = self.generator

        if isinstance(generator, OpenAIGenerator):
            model_info = generator.model
        elif isinstance(generator, HuggingFaceLocalGenerator):
            model_info = generator.huggingface_pipeline_kwargs.get("model")
        elif isinstance(generator, HuggingFaceAPIGenerator):
            model_info = generator.api_params.get("model")
        elif isinstance(generator, AzureOpenAIGenerator):
            model_info = generator.azure_deployment
        else:
            model_info = "Unknown"

        return f"{self.__class__.__name__}(llm={model_info}(...))"
