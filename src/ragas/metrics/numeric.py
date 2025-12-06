"""Base class for all numeric metrics"""

__all__ = ["numeric_metric", "NumericMetric"]

import typing as t
from dataclasses import dataclass

if t.TYPE_CHECKING:
    from ragas.metrics.base import EmbeddingModelType

from .base import SimpleLLMMetric
from .decorator import NumericMetricProtocol, create_metric_decorator
from .validators import NumericValidator


@dataclass(repr=False)
class NumericMetric(SimpleLLMMetric, NumericValidator):
    """
    Metric for continuous numeric evaluations within a specified range.

    This class is used for metrics that output numeric scores within a
    defined range, such as 0.0 to 1.0 for similarity scores or 1-10 ratings.
    Uses the instructor library for structured LLM outputs.

    Attributes
    ----------
    allowed_values : Union[Tuple[float, float], range]
        The valid range for metric outputs. Can be a tuple of (min, max) floats
        or a range object. Default is (0.0, 1.0).
    llm : Optional[BaseRagasLLM]
        The language model instance for evaluation. Can be created using llm_factory().
    prompt : Optional[Union[str, Prompt]]
        The prompt template for the metric. Should contain placeholders for
        evaluation inputs that will be formatted at runtime.

    Examples
    --------
    >>> from ragas.metrics import NumericMetric
    >>> from ragas.llms import llm_factory
    >>> from openai import OpenAI
    >>>
    >>> # Create an LLM instance
    >>> client = OpenAI(api_key="your-api-key")
    >>> llm = llm_factory("gpt-4o-mini", client=client)
    >>>
    >>> # Create a custom numeric metric with 0-10 range
    >>> metric = NumericMetric(
    ...     name="quality_score",
    ...     llm=llm,
    ...     prompt="Rate the quality of this response on a scale of 0-10: {response}",
    ...     allowed_values=(0.0, 10.0)
    ... )
    >>>
    >>> # Score with the metric
    >>> result = metric.score(
    ...     llm=llm,
    ...     response="This is a great response!"
    ... )
    >>> print(result.value)  # Output: a float between 0.0 and 10.0
    """

    allowed_values: t.Union[t.Tuple[float, float], range] = (0.0, 1.0)

    def __post_init__(self):
        super().__post_init__()
        # Use the factory to create and mark the model as auto-generated
        from ragas.metrics.base import create_auto_response_model

        self._response_model = create_auto_response_model(
            "NumericResponseModel", reason=(str, ...), value=(float, ...)
        )

    def get_correlation(
        self, gold_labels: t.List[str], predictions: t.List[str]
    ) -> float:
        """
        Calculate the correlation between gold labels and predictions.
        This is a placeholder method and should be implemented based on the specific metric.
        """
        try:
            from scipy.stats import pearsonr
        except ImportError:
            raise ImportError(
                "scipy is required for correlation calculation. "
                "Please install it with `pip install scipy`."
            )
        # Convert strings to floats for correlation calculation
        gold_floats = [float(x) for x in gold_labels]
        pred_floats = [float(x) for x in predictions]
        result = pearsonr(gold_floats, pred_floats)
        # pearsonr returns (correlation, p-value) tuple
        correlation = t.cast(float, result[0])
        return correlation

    @classmethod
    def load(
        cls, path: str, embedding_model: t.Optional["EmbeddingModelType"] = None
    ) -> "NumericMetric":
        """
        Load a NumericMetric from a JSON file.

        Parameters:
        -----------
        path : str
            File path to load from. Supports .gz compressed files.
        embedding_model : Optional[Any]
            Embedding model for DynamicFewShotPrompt. Required if the original used one.

        Returns:
        --------
        NumericMetric
            Loaded metric instance

        Raises:
        -------
        ValueError
            If file cannot be loaded or is not a NumericMetric
        """
        # Validate metric type before loading
        cls._validate_metric_type(path)

        # Load using parent class method
        metric = super().load(path, embedding_model=embedding_model)

        # Additional type check for safety
        if not isinstance(metric, cls):
            raise ValueError(f"Loaded metric is not a {cls.__name__}")

        # Convert allowed_values back to tuple if it's a list (due to JSON serialization)
        if hasattr(metric, "allowed_values") and isinstance(
            metric.allowed_values, list
        ):
            # Ensure it's a 2-element tuple for NumericMetric
            if len(metric.allowed_values) == 2:
                metric.allowed_values = (
                    metric.allowed_values[0],
                    metric.allowed_values[1],
                )
            else:
                metric.allowed_values = tuple(metric.allowed_values)  # type: ignore

        return metric


def numeric_metric(
    *,
    name: t.Optional[str] = None,
    allowed_values: t.Optional[t.Union[t.Tuple[float, float], range]] = None,
    **metric_params: t.Any,
) -> t.Callable[[t.Callable[..., t.Any]], NumericMetricProtocol]:
    """
    Decorator for creating numeric/continuous metrics.

    This decorator transforms a regular function into a NumericMetric instance
    that outputs continuous values within a specified range.

    Parameters
    ----------
    name : str, optional
        Name for the metric. If not provided, uses the function name.
    allowed_values : Union[Tuple[float, float], range], optional
        The valid range for metric outputs as (min, max) tuple or range object.
        Default is (0.0, 1.0).
    **metric_params : Any
        Additional parameters to pass to the metric initialization.

    Returns
    -------
    Callable[[Callable[..., Any]], NumericMetricProtocol]
        A decorator that transforms a function into a NumericMetric instance.

    Examples
    --------
    >>> from ragas.metrics import numeric_metric
    >>>
    >>> @numeric_metric(name="relevance_score", allowed_values=(0.0, 1.0))
    >>> def calculate_relevance(user_input: str, response: str) -> float:
    ...     '''Calculate relevance score between 0 and 1.'''
    ...     # Simple word overlap example
    ...     user_words = set(user_input.lower().split())
    ...     response_words = set(response.lower().split())
    ...     if not user_words:
    ...         return 0.0
    ...     overlap = len(user_words & response_words)
    ...     return overlap / len(user_words)
    >>>
    >>> result = calculate_relevance(
    ...     user_input="What is Python?",
    ...     response="Python is a programming language"
    ... )
    >>> print(result.value)  # Numeric score between 0.0 and 1.0
    """
    if allowed_values is None:
        allowed_values = (0.0, 1.0)

    decorator_factory = create_metric_decorator()
    return decorator_factory(name=name, allowed_values=allowed_values, **metric_params)  # type: ignore[return-value]
