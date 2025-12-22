import typing as t

from ragas.llms.adapters.base import StructuredOutputAdapter

if t.TYPE_CHECKING:
    from ragas.llms.litellm_llm import LiteLLMStructuredLLM


class LiteLLMAdapter(StructuredOutputAdapter):
    """
    Adapter using LiteLLM for structured outputs.

    Supports: All 100+ LiteLLM providers (Gemini, Ollama, vLLM, Groq, etc.)
    """

    def create_llm(
        self,
        client: t.Any,
        model: str,
        provider: str,
        **kwargs,
    ) -> "LiteLLMStructuredLLM":
        """
        Create LiteLLMStructuredLLM instance.

        Args:
            client: Pre-initialized client
            model: Model name
            provider: Provider name
            **kwargs: Additional model arguments

        Returns:
            LiteLLMStructuredLLM instance
        """
        from ragas.llms.litellm_llm import LiteLLMStructuredLLM

        cache = kwargs.pop("cache", None)

        return LiteLLMStructuredLLM(
            client=client,
            model=model,
            provider=provider,
            cache=cache,
            **kwargs,
        )
