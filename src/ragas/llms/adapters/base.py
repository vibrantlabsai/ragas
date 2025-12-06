import typing as t
from abc import ABC, abstractmethod


class StructuredOutputAdapter(ABC):
    """
    Base class for structured output adapters.

    Provides a simple interface for adapters that support structured output
    from different backends (Instructor, LiteLLM, etc).
    """

    @abstractmethod
    def create_llm(
        self,
        client: t.Any,
        model: str,
        provider: str,
        **kwargs,
    ) -> t.Any:
        """
        Create an LLM instance with structured output support.

        Args:
            client: Pre-initialized client instance
            model: Model name (e.g., "gpt-4o", "gemini-2.0-flash")
            provider: Provider name (e.g., "openai", "google")
            **kwargs: Additional model arguments

        Returns:
            InstructorBaseRagasLLM-compatible instance
        """
        pass
