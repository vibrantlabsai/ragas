"""CHRFScore metric - Modern collections implementation."""

import typing as t

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult


class CHRFScore(BaseMetric):
    """
    Calculate CHRF (Character F-score) between reference and response texts.

    CHRF is a character n-gram F-score metric that correlates well with human
    judgments for machine translation quality. Unlike BLEU which operates on
    words, CHRF operates on character-level n-grams, making it more robust
    to morphological variations and better suited for morphologically rich languages.

    This implementation uses the sacrebleu library for consistent and reproducible
    scoring.

    Usage:
        >>> from ragas.metrics.collections import CHRFScore
        >>>
        >>> metric = CHRFScore()
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
        name: The metric name (default: "chrf_score")
        kwargs: Additional arguments to pass to sacrebleu.corpus_chrf
            (e.g., char_order, word_order, beta, eps_smoothing)
        allowed_values: Score range (0.0 to 1.0)
    """

    def __init__(
        self,
        name: str = "chrf_score",
        kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        **base_kwargs,
    ):
        """Initialize CHRFScore metric."""
        super().__init__(name=name, **base_kwargs)
        self.kwargs = kwargs or {}

    async def ascore(
        self,
        reference: str,
        response: str,
    ) -> MetricResult:
        """
        Calculate CHRF score asynchronously.

        Args:
            reference: The reference/ground truth text
            response: The response text to evaluate

        Returns:
            MetricResult with CHRF score (0.0-1.0)
        """
        try:
            from sacrebleu import corpus_chrf
        except ImportError:
            raise ImportError(
                "sacrebleu is required for CHRF score calculation. "
                "Please install it using `pip install sacrebleu`"
            )

        if not isinstance(reference, str) or not isinstance(response, str):
            return MetricResult(
                value=0.0,
                reason="Invalid input: reference and response must be strings",
            )

        if not reference.strip() or not response.strip():
            return MetricResult(
                value=0.0,
                reason="Empty input: reference or response is empty",
            )

        # corpus_chrf expects hypotheses as list of strings and references as list of list of strings
        references = [[reference]]
        hypotheses = [response]

        score = corpus_chrf(hypotheses, references, **self.kwargs).score / 100

        return MetricResult(value=float(score))
