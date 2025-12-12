"""QuotedSpansAlignment metric - Modern collections implementation."""

import typing as t

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import count_matched_spans, extract_quoted_spans


class QuotedSpansAlignment(BaseMetric):
    """
    Measure citation alignment for quoted spans in model-generated answers.

    This metric computes the fraction of quoted spans appearing verbatim in any
    of the provided source passages. If an answer quotes facts that cannot be
    found in the sources, the metric will reflect that drift.

    The metric performs light normalization by collapsing whitespace and
    lower-casing strings. You can adjust the minimum length of a quoted span
    and choose to disable case folding if desired.

    Usage:
        >>> from ragas.metrics.collections import QuotedSpansAlignment
        >>>
        >>> metric = QuotedSpansAlignment()
        >>>
        >>> result = await metric.ascore(
        ...     response='The study found that "machine learning models improve accuracy".',
        ...     retrieved_contexts=["Machine learning models improve accuracy by 15%."]
        ... )
        >>> print(f"Score: {result.value}")
        >>>
        >>> results = await metric.abatch_score([
        ...     {
        ...         "response": 'He said "the results are significant".',
        ...         "retrieved_contexts": ["The results are significant according to the paper."]
        ...     },
        ... ])

    Attributes:
        name: The metric name (default: "quoted_spans_alignment")
        casefold: Whether to normalize text by lower-casing before matching.
        min_span_words: Minimum number of words in a quoted span.
        allowed_values: Score range (0.0 to 1.0)
    """

    def __init__(
        self,
        name: str = "quoted_spans_alignment",
        casefold: bool = True,
        min_span_words: int = 3,
        **base_kwargs,
    ):
        """
        Initialize QuotedSpansAlignment metric.

        Args:
            name: The metric name.
            casefold: Whether to normalize text by lower-casing before matching.
            min_span_words: Minimum number of words in a quoted span.
            **base_kwargs: Additional arguments passed to BaseMetric.
        """
        super().__init__(name=name, **base_kwargs)
        self.casefold = casefold
        self.min_span_words = min_span_words

    async def ascore(
        self,
        response: str,
        retrieved_contexts: t.List[str],
    ) -> MetricResult:
        """
        Calculate quoted spans alignment score asynchronously.

        Args:
            response: The model response containing quoted spans.
            retrieved_contexts: List of source passages to check against.

        Returns:
            MetricResult with alignment score (0.0-1.0) and metadata containing
            matched and total counts.
        """
        if not isinstance(response, str):
            return MetricResult(
                value=0.0,
                reason="Invalid input: response must be a string",
            )

        if not isinstance(retrieved_contexts, list):
            return MetricResult(
                value=0.0,
                reason="Invalid input: retrieved_contexts must be a list of strings",
            )

        spans = extract_quoted_spans(response, min_len=self.min_span_words)

        if not spans:
            return MetricResult(
                value=1.0,
                reason="No quoted spans found in response",
            )

        matched, total = count_matched_spans(
            spans, retrieved_contexts, casefold=self.casefold
        )

        score = matched / total if total > 0 else 0.0

        reason = f"Matched {matched}/{total} quoted spans"
        return MetricResult(value=float(score), reason=reason)
