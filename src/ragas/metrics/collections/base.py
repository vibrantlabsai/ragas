"""Base class for collections metrics with modern component validation."""

import asyncio
import typing as t

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.base import SimpleBaseMetric
from ragas.metrics.result import MetricResult
from ragas.metrics.validators import NumericValidator


def validate_text_list(name: str, value: t.Any) -> None:
    """Reject a single string where a list of strings is expected.

    A string is iterable, so a metric that loops over it would score it one character at
    a time (for ContextPrecision, one LLM call per character) without raising. This
    usually happens when a list was stored as text, for example by the local/csv backend.
    """
    if not isinstance(value, (str, bytes)):
        return
    hint = "wrap it in a list"
    if str(value).lstrip().startswith("["):
        hint = (
            "it looks like a list that was saved as text (the local/csv backend stores "
            "every value as a string); parse it back into a list, e.g. with "
            "ast.literal_eval, or use the local/jsonl backend"
        )
    raise ValueError(
        f"{name} must be a list of strings, not a single {type(value).__name__}: {hint}."
    )


class BaseMetric(SimpleBaseMetric, NumericValidator):
    """
    Base class for metrics collections with modern component validation.

    This class inherits from SimpleBaseMetric and NumericValidator to provide:
    - All the base metric functionality (ascore, abatch_score, score, batch_score)
    - Numeric validation with configurable ranges
    - Modern LLM and embedding component validation (when defined by subclass)
    - Rejection of legacy wrappers with helpful error messages
    - Consistent error handling and type safety

    Attributes:
        name: The metric name
        allowed_values: Score range for numeric validation (tuple of min, max)

    Note: Subclasses define llm and/or embeddings fields only if they need them.
    The base classes handle all the core metric functionality - we just add modern component validation.
    """

    def __init__(
        self,
        name: str = "base_metric",
        allowed_values: t.Tuple[float, float] = (0.0, 1.0),
        **kwargs,
    ):
        """Initialize the base metric with validation."""
        super().__init__(name=name, allowed_values=allowed_values)

        # Validate components only if the metric defines them
        # Check if this instance has these attributes after initialization
        if hasattr(self, "llm"):
            self._validate_llm()
        if hasattr(self, "embeddings"):
            self._validate_embeddings()

    async def ascore(self, **kwargs) -> MetricResult:
        """
        Default async scoring method - subclasses should override this.

        This base implementation just returns a placeholder result.
        Subclasses should override this method with their specific logic.

        The base class handles component validation in __post_init__.
        """
        return MetricResult(
            value=0.0, reason="Base metric placeholder - override ascore() in subclass"
        )

    def score(self, **kwargs) -> MetricResult:
        """
        Synchronous scoring method that wraps ascore().

        This is a convenience method for backward compatibility and sync usage.
        For better performance, prefer using ascore() directly in async contexts.

        Returns:
            MetricResult object
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            # If we get here, there's already a running loop
            raise RuntimeError(
                "Cannot call sync score() from an async context. Use ascore() instead."
            )
        except RuntimeError as e:
            if "Use ascore() instead" in str(e):
                raise  # Re-raise our custom error
            # No running loop found, safe to use asyncio.run()
            return asyncio.run(self.ascore(**kwargs))

    def batch_score(
        self,
        inputs: t.List[t.Dict[str, t.Any]],
    ) -> t.List[MetricResult]:
        """
        Synchronous batch scoring that wraps abatch_score().

        This is a convenience method for backward compatibility and sync usage.
        For better performance, prefer using abatch_score() directly in async contexts.

        Args:
            inputs: List of input dictionaries for scoring

        Returns:
            List of MetricResult objects
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            # If we get here, there's already a running loop
            raise RuntimeError(
                "Cannot call sync batch_score() from an async context. Use abatch_score() instead."
            )
        except RuntimeError as e:
            if "Use abatch_score() instead" in str(e):
                raise  # Re-raise our custom error
            # No running loop found, safe to use asyncio.run()
            return asyncio.run(self.abatch_score(inputs))

    def _validate_llm(self):
        """Validate that a modern InstructorLLM is provided."""
        llm = getattr(self, "llm", None)

        if not isinstance(llm, InstructorBaseRagasLLM):
            raise ValueError(
                f"Collections metrics only support modern InstructorLLM. Found: {type(llm).__name__}. "
                f"Use: llm_factory('gpt-4o-mini', client=openai_client)"
            )

    def _validate_embeddings(self):
        """Validate that modern embeddings are provided."""
        embeddings = getattr(self, "embeddings", None)

        if not isinstance(embeddings, BaseRagasEmbedding):
            raise ValueError(
                f"Collections metrics only support modern embeddings. Found: {type(embeddings).__name__}. "
                f"Use: embedding_factory('openai', model='text-embedding-ada-002', client=openai_client, interface='modern')"
            )
