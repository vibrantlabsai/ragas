"""BLEU Score metric v2 - Class-based implementation with automatic validation."""

import typing as t

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult


class BleuScore(BaseMetric):
    """
    Calculate BLEU score between reference and response texts.

    This implementation provides automatic validation and pure async design
    without requiring LLM or embedding components. Uses sacrebleu library.

    Usage:
        >>> from ragas.metrics.collections import BleuScore
        >>>
        >>> metric = BleuScore()
        >>>
        >>> result = await metric.ascore(
        ...     reference="The capital of France is Paris.",
        ...     response="Paris is the capital of France."
        ... )
        >>> print(f"Score: {result.value}")
        >>>
        >>> results = await metric.abatch_score([
        ...     {"reference": "Text 1", "response": "Response 1"},
        ...     {"reference": "Text 2", "response": "Response 2"},
        ... ])

    Attributes:
        name: The metric name
        kwargs: Additional arguments to pass to sacrebleu.corpus_bleu
        allowed_values: Score range (0.0 to 1.0)
    """

    def __init__(
        self,
        name: str = "bleu_score",
        kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        **base_kwargs,
    ):
        """Initialize BleuScore metric."""
        super().__init__(name=name, **base_kwargs)
        self.kwargs = kwargs or {}

    async def ascore(
        self,
        reference: str,
        response: str,
    ) -> MetricResult:
        """
        Calculate BLEU score asynchronously.

        Args:
            reference: The reference/ground truth text
            response: The response text to evaluate

        Returns:
            MetricResult with BLEU score (0.0-1.0)
        """
        try:
            from sacrebleu import corpus_bleu
        except ImportError:
            raise ImportError(
                "sacrebleu is required for BLEU score calculation. "
                "Please install it using `pip install sacrebleu`"
            )

        assert isinstance(reference, str), "BleuScore expects a valid reference string"
        assert isinstance(response, str), "BleuScore expects a valid response string"

        # Pass full texts as a single segment (matches ChrfScore). Sentence-splitting
        # into N streams of length 1 made sacrebleu only score the first sentence.
        score = corpus_bleu([response], [[reference]], **self.kwargs).score / 100

        assert isinstance(score, float), "Expecting a float"
        return MetricResult(value=float(score))
