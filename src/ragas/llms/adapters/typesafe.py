import typing as t

from ragas.llms.adapters.base import StructuredOutputAdapter

if t.TYPE_CHECKING:
    from ragas.llms.typesafe_llm import TypeSafeStructuredLLM


class TypeSafeAdapter(StructuredOutputAdapter):
    """
    Adapter using TypeSafe AI (Jev) for high-speed, calibrated System One evaluation.

    Specializes in sub-second decision judging, rubric scoring, and classification.
    """

    def create_llm(
        self,
        client: t.Any,
        model: str,
        provider: str,
        **kwargs,
    ) -> "TypeSafeStructuredLLM":
        """
        Create TypeSafeStructuredLLM instance.

        Args:
            client: Pre-initialized client (TypeSafeClient, AsyncTypeSafeClient, or None)
            model: Model name (e.g. "jev-latest")
            provider: Provider name ("typesafe")
            **kwargs: Additional model arguments (api_key, generator_llm, cache, etc.)

        Returns:
            TypeSafeStructuredLLM instance
        """
        from ragas.llms.typesafe_llm import TypeSafeStructuredLLM

        cache = kwargs.pop("cache", None)
        generator_llm = kwargs.pop("generator_llm", None)
        api_key = kwargs.pop("api_key", None)

        return TypeSafeStructuredLLM(
            client=client,
            model=model,
            api_key=api_key,
            generator_llm=generator_llm,
            cache=cache,
            **kwargs,
        )
