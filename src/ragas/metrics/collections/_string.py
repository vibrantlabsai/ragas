"""String-based metrics v2 - Class-based implementations with automatic validation."""

from enum import Enum

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult


class DistanceMeasure(Enum):
    LEVENSHTEIN = "levenshtein"
    HAMMING = "hamming"
    JARO = "jaro"
    JARO_WINKLER = "jaro_winkler"


class ExactMatch(BaseMetric):
    """
    Check if reference and response are exactly identical.

    This implementation provides automatic validation and pure async design
    without requiring LLM or embedding components.

    Usage:
        >>> from ragas.metrics.collections import ExactMatch
        >>>
        >>> metric = ExactMatch()
        >>>
        >>> result = await metric.ascore(
        ...     reference="Hello World",
        ...     response="Hello World"
        ... )
        >>> print(f"Score: {result.value}")  # 1.0
        >>>
        >>> results = await metric.abatch_score([
        ...     {"reference": "Text 1", "response": "Text 1"},
        ...     {"reference": "Text 2", "response": "Different"},
        ... ])

    Attributes:
        name: The metric name
        allowed_values: Score range (0.0 to 1.0)
    """

    def __init__(
        self,
        name: str = "exact_match",
        **base_kwargs,
    ):
        """Initialize ExactMatch metric."""
        super().__init__(name=name, **base_kwargs)

    async def ascore(
        self,
        reference: str,
        response: str,
    ) -> MetricResult:
        """
        Check if reference and response match exactly.

        Args:
            reference: The reference/ground truth text
            response: The response text to evaluate

        Returns:
            MetricResult with 1.0 if exact match, 0.0 otherwise
        """
        assert isinstance(reference, str), (
            "ExactMatch expects a valid reference string"
        )
        assert isinstance(response, str), (
            "ExactMatch expects a valid response string"
        )

        score = float(reference == response)
        return MetricResult(value=score)


class StringPresence(BaseMetric):
    """
    Check if reference string is present in the response.

    This implementation provides automatic validation and pure async design
    without requiring LLM or embedding components.

    Usage:
        >>> from ragas.metrics.collections import StringPresence
        >>>
        >>> metric = StringPresence()
        >>>
        >>> result = await metric.ascore(
        ...     reference="Paris",
        ...     response="The capital of France is Paris."
        ... )
        >>> print(f"Score: {result.value}")  # 1.0
        >>>
        >>> results = await metric.abatch_score([
        ...     {"reference": "cat", "response": "The cat sat on the mat"},
        ...     {"reference": "dog", "response": "The cat sat on the mat"},
        ... ])

    Attributes:
        name: The metric name
        allowed_values: Score range (0.0 to 1.0)
    """

    def __init__(
        self,
        name: str = "string_present",
        **base_kwargs,
    ):
        """Initialize StringPresence metric."""
        super().__init__(name=name, **base_kwargs)

    async def ascore(
        self,
        reference: str,
        response: str,
    ) -> MetricResult:
        """
        Check if reference is present in response.

        Args:
            reference: The reference string to search for
            response: The response text to search in

        Returns:
            MetricResult with 1.0 if reference is in response, 0.0 otherwise
        """
        assert isinstance(reference, str), (
            "StringPresence expects a valid reference string"
        )
        assert isinstance(response, str), (
            "StringPresence expects a valid response string"
        )

        score = float(reference in response)
        return MetricResult(value=score)


class NonLLMStringSimilarity(BaseMetric):
    """
    Calculate string similarity between reference and response using various distance measures.

    This implementation provides automatic validation and pure async design
    without requiring LLM or embedding components. Uses rapidfuzz library.

    Usage:
        >>> from ragas.metrics.collections import NonLLMStringSimilarity, DistanceMeasure
        >>>
        >>> metric = NonLLMStringSimilarity(distance_measure=DistanceMeasure.LEVENSHTEIN)
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
        distance_measure: The distance measure to use (default: LEVENSHTEIN)
        allowed_values: Score range (0.0 to 1.0)
    """

    def __init__(
        self,
        name: str = "non_llm_string_similarity",
        distance_measure: DistanceMeasure = DistanceMeasure.LEVENSHTEIN,
        **base_kwargs,
    ):
        """Initialize NonLLMStringSimilarity metric."""
        super().__init__(name=name, **base_kwargs)
        self.distance_measure = distance_measure

        try:
            from rapidfuzz import distance
        except ImportError:
            raise ImportError(
                "rapidfuzz is required for string distance. "
                "Please install it using `pip install rapidfuzz`"
            )

        self.distance_measure_map = {
            DistanceMeasure.LEVENSHTEIN: distance.Levenshtein,
            DistanceMeasure.HAMMING: distance.Hamming,
            DistanceMeasure.JARO: distance.Jaro,
            DistanceMeasure.JARO_WINKLER: distance.JaroWinkler,
        }

    async def ascore(
        self,
        reference: str,
        response: str,
    ) -> MetricResult:
        """
        Calculate string similarity score asynchronously.

        Args:
            reference: The reference/ground truth text
            response: The response text to evaluate

        Returns:
            MetricResult with similarity score (0.0-1.0)
        """
        assert isinstance(reference, str), (
            "NonLLMStringSimilarity expects a valid reference string"
        )
        assert isinstance(response, str), (
            "NonLLMStringSimilarity expects a valid response string"
        )

        score = 1 - self.distance_measure_map[
            self.distance_measure
        ].normalized_distance(reference, response)

        assert isinstance(score, float), "Expecting a float"
        return MetricResult(value=float(score))
