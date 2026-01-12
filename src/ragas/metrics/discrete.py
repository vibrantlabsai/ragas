"""Base class from which all discrete metrics should inherit."""

__all__ = ["discrete_metric", "DiscreteMetric"]

import typing as t
from dataclasses import dataclass, field

from pydantic import Field

if t.TYPE_CHECKING:
    from ragas.metrics.base import EmbeddingModelType

from .base import SimpleLLMMetric
from .decorator import DiscreteMetricProtocol, create_metric_decorator
from .validators import DiscreteValidator


@dataclass(repr=False)
class DiscreteMetric(SimpleLLMMetric, DiscreteValidator):
    """
    Metric for categorical/discrete evaluations with predefined allowed values.

    This class is used for metrics that output categorical values like
    "pass/fail", "good/bad/excellent", or custom discrete categories.
    Uses the instructor library for structured LLM outputs.

    Attributes
    ----------
    allowed_values : List[str]
        List of allowed categorical values the metric can output.
        Default is ["pass", "fail"].
    prompt : Optional[Union[str, Prompt]]
        The prompt template for the metric. Should contain placeholders for
        evaluation inputs that will be formatted at runtime.

    Examples
    --------
    >>> from ragas.metrics import DiscreteMetric
    >>> from ragas.llms import llm_factory
    >>> from openai import OpenAI
    >>>
    >>> # Create an LLM instance
    >>> client = OpenAI(api_key="your-api-key")
    >>> llm = llm_factory("gpt-4o-mini", client=client)
    >>>
    >>> # Create a custom discrete metric
    >>> metric = DiscreteMetric(
    ...     name="quality_check",
    ...     prompt="Check the quality of the response: {response}. Return 'excellent', 'good', or 'poor'.",
    ...     allowed_values=["excellent", "good", "poor"]
    ... )
    >>>
    >>> # Score with the metric
    >>> result = metric.score(
    ...     llm=llm,
    ...     response="This is a great response!"
    ... )
    >>> print(result.value)  # Output: "excellent" or similar
    """

    allowed_values: t.List[str] = field(default_factory=lambda: ["pass", "fail"])

    def __post_init__(self):
        super().__post_init__()
        values = tuple(self.allowed_values)
        # Use the factory to create and mark the model as auto-generated
        from ragas.metrics.base import create_auto_response_model

        self._response_model = create_auto_response_model(
            "DiscreteResponseModel",
            reason=(str, Field(..., description="Reasoning for the value")),
            value=(t.Literal[values], Field(..., description="The value predicted")),
        )

    def get_correlation(
        self, gold_labels: t.List[str], predictions: t.List[str]
    ) -> float:
        """
        Calculate the correlation between gold labels and predictions.
        This is a placeholder method and should be implemented based on the specific metric.
        """
        try:
            from sklearn.metrics import cohen_kappa_score
        except ImportError:
            raise ImportError(
                "scikit-learn is required for correlation calculation. "
                "Please install it with `pip install scikit-learn`."
            )
        return cohen_kappa_score(gold_labels, predictions)

    @classmethod
    def load(
        cls, path: str, embedding_model: t.Optional["EmbeddingModelType"] = None
    ) -> "DiscreteMetric":
        """
        Load a DiscreteMetric from a JSON file.

        Parameters:
        -----------
        path : str
            File path to load from. Supports .gz compressed files.
        embedding_model : Optional[Any]
            Embedding model for DynamicFewShotPrompt. Required if the original used one.

        Returns:
        --------
        DiscreteMetric
            Loaded metric instance

        Raises:
        -------
        ValueError
            If file cannot be loaded or is not a DiscreteMetric
        """
        # Validate metric type before loading
        cls._validate_metric_type(path)

        # Load using parent class method
        metric = super().load(path, embedding_model=embedding_model)

        # Additional type check for safety
        if not isinstance(metric, cls):
            raise ValueError(f"Loaded metric is not a {cls.__name__}")

        return metric


def discrete_metric(
    *,
    name: t.Optional[str] = None,
    allowed_values: t.Optional[t.List[str]] = None,
    **metric_params: t.Any,
) -> t.Callable[[t.Callable[..., t.Any]], DiscreteMetricProtocol]:
    """
    Decorator for creating discrete/categorical metrics.

    This decorator transforms a regular function into a DiscreteMetric instance
    that can be used for evaluation with predefined categorical outputs.

    Parameters
    ----------
    name : str, optional
        Name for the metric. If not provided, uses the function name.
    allowed_values : List[str], optional
        List of allowed categorical values for the metric output.
        Default is ["pass", "fail"].
    **metric_params : Any
        Additional parameters to pass to the metric initialization.

    Returns
    -------
    Callable[[Callable[..., Any]], DiscreteMetricProtocol]
        A decorator that transforms a function into a DiscreteMetric instance.

    Examples
    --------
    >>> from ragas.metrics import discrete_metric
    >>>
    >>> @discrete_metric(name="sentiment", allowed_values=["positive", "neutral", "negative"])
    >>> def sentiment_analysis(user_input: str, response: str) -> str:
    ...     '''Analyze sentiment of the response.'''
    ...     if "great" in response.lower() or "good" in response.lower():
    ...         return "positive"
    ...     elif "bad" in response.lower() or "poor" in response.lower():
    ...         return "negative"
    ...     return "neutral"
    >>>
    >>> result = sentiment_analysis(
    ...     user_input="How was your day?",
    ...     response="It was great!"
    ... )
    >>> print(result.value)  # "positive"
    """
    if allowed_values is None:
        allowed_values = ["pass", "fail"]

    decorator_factory = create_metric_decorator()
    return decorator_factory(name=name, allowed_values=allowed_values, **metric_params)  # type: ignore[return-value]
