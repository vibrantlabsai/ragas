"""decorator factory for creating custom metrics"""

__all__ = [
    "create_metric_decorator",
    "BaseMetricProtocol",
    "DiscreteMetricProtocol",
    "NumericMetricProtocol",
    "RankingMetricProtocol",
]

import asyncio
import inspect
import typing as t
import warnings
from dataclasses import dataclass, field
from typing import get_args, get_origin, get_type_hints

from pydantic import ConfigDict, ValidationError, create_model

if t.TYPE_CHECKING:
    from typing_extensions import Protocol
else:
    try:
        from typing_extensions import Protocol
    except ImportError:
        from typing import Protocol

from .base import SimpleBaseMetric
from .result import MetricResult
from .validators import get_validator_for_allowed_values

# Type variables for generic typing
F = t.TypeVar("F", bound=t.Callable[..., t.Any])


# Protocol classes for type hints
class BaseMetricProtocol(Protocol):
    """Protocol defining the base metric interface."""

    name: str

    def score(self, **kwargs) -> MetricResult:
        """Synchronous scoring method."""
        ...

    async def ascore(self, **kwargs) -> MetricResult:
        """Asynchronous scoring method."""
        ...

    def batch_score(self, inputs: t.List[t.Dict[str, t.Any]]) -> t.List[MetricResult]:
        """Batch scoring method."""
        ...

    async def abatch_score(
        self, inputs: t.List[t.Dict[str, t.Any]]
    ) -> t.List[MetricResult]:
        """Asynchronous batch scoring method."""
        ...

    def __call__(self, *args, **kwargs):
        """Make the metric directly callable like the original function."""
        ...


class DiscreteMetricProtocol(BaseMetricProtocol, Protocol):
    """Protocol for discrete metrics with allowed values."""

    allowed_values: t.List[str]


class NumericMetricProtocol(BaseMetricProtocol, Protocol):
    """Protocol for numeric metrics with value ranges."""

    allowed_values: t.Tuple[float, float]


class RankingMetricProtocol(BaseMetricProtocol, Protocol):
    """Protocol for ranking metrics with list outputs."""

    allowed_values: int  # Expected list length


def create_metric_decorator():
    """
    Factory function that creates decorator factories for different metric types.

    Returns:
        A decorator factory function that determines the metric type based on allowed_values
    """

    def decorator_factory(
        name: t.Optional[str] = None,
        **metric_params,
    ):
        """
        Creates a decorator that wraps a function into a metric instance.

        Args:
            name: Optional name for the metric (defaults to function name)
            **metric_params: Additional parameters specific to the metric type
                (values for DiscreteMetrics, range for NumericMetrics, etc.)

        Returns:
            A decorator function
        """

        def decorator(func):
            # Get metric name and check if function is async
            metric_name = name or func.__name__
            is_async = inspect.iscoroutinefunction(func)
            sig = inspect.signature(func)

            # Determine the appropriate validator based on allowed_values
            allowed_values = metric_params.get("allowed_values")
            # If no allowed_values provided, default to discrete with pass/fail
            if allowed_values is None:
                allowed_values = ["pass", "fail"]
            validator_class = get_validator_for_allowed_values(allowed_values)

            # TODO: Move to dataclass type implementation
            @dataclass(repr=False)
            class CustomMetric(SimpleBaseMetric, validator_class):
                _func: t.Optional[t.Callable[..., t.Any]] = field(
                    default=None, init=False
                )
                _metric_params: t.Dict[str, t.Any] = field(
                    default_factory=dict, init=False
                )
                # Note: allowed_values is inherited from SimpleBaseMetric

                def _validate_result_value(self, result_value):
                    """Validate result value using the appropriate validator mixin."""
                    return self.validate_result_value(result_value)

                def _create_positional_error(self, args: tuple, kwargs: dict) -> str:
                    """Create error message for positional arguments."""
                    func_param_names = list(sig.parameters.keys())

                    msg = f"\n❌ {self.name}.score() requires keyword arguments, not positional.\n\n"
                    msg += (
                        f"   You provided: score({', '.join(repr(a) for a in args)})\n"
                    )
                    msg += "   Correct usage: score("

                    corrections = []
                    for i, param_name in enumerate(func_param_names):
                        if i < len(args):
                            corrections.append(f"{param_name}={repr(args[i])}")
                        else:
                            corrections.append(f"{param_name}=...")

                    msg += ", ".join(corrections) + ")\n\n"
                    msg += "   💡 Tip: Always use parameter names for clarity and future compatibility."

                    return msg

                def _create_pydantic_model(self):
                    """Create a Pydantic model dynamically from the function signature."""
                    try:
                        type_hints = get_type_hints(func)
                    except (NameError, AttributeError):
                        type_hints = {}

                    field_definitions = {}

                    for name, param in sig.parameters.items():
                        # Get type hint, default to Any if no hint available
                        type_hint = type_hints.get(name, param.annotation)
                        if type_hint == inspect.Parameter.empty:
                            if param.default != inspect.Parameter.empty:
                                type_hint = type(param.default)
                            else:
                                type_hint = t.Any

                        # Get default value
                        if param.default != inspect.Parameter.empty:
                            default = param.default
                        else:
                            # Check if it's an optional type
                            origin = get_origin(type_hint)
                            if origin is t.Union and type(None) in get_args(type_hint):
                                # Optional type, default to None
                                default = None
                            else:
                                # Required field
                                default = ...

                        field_definitions[name] = (type_hint, default)

                    # Create the dynamic model with arbitrary types allowed
                    model_name = f"{self.name}_ValidationModel"
                    return create_model(
                        model_name,
                        __config__=ConfigDict(arbitrary_types_allowed=True),
                        **field_definitions,
                    )

                def _format_pydantic_errors(
                    self, validation_error: ValidationError
                ) -> str:
                    """Format Pydantic validation errors into user-friendly messages."""
                    msg = f"\n❌ Type validation errors for {self.name}:\n\n"

                    for error in validation_error.errors():
                        field = error["loc"][0]
                        error_msg = error["msg"]
                        input_value = error.get("input", "N/A")

                        msg += f"   - {field}: {error_msg} (got: {repr(input_value)})\n"

                    return msg

                def _validate_inputs(self, args: tuple, kwargs: dict):
                    """Validate all inputs using Pydantic with helpful error messages."""
                    # Check for positional arguments (keep custom helpful error)
                    if args:
                        raise TypeError(self._create_positional_error(args, kwargs))

                    # Create dynamic Pydantic model from function signature
                    try:
                        pydantic_model = self._create_pydantic_model()
                    except Exception as e:
                        # Fallback if model creation fails
                        warnings.warn(
                            f"Could not create validation model: {e}", UserWarning
                        )
                        return

                    # Warn about unknown arguments (but continue processing)
                    valid_params = set(pydantic_model.model_fields.keys())
                    unknown = set(kwargs.keys()) - valid_params

                    if unknown:
                        warnings.warn(
                            f"⚠️  {self.name} received unknown arguments: {', '.join(sorted(unknown))}\n"
                            f"   Valid arguments: {', '.join(sorted(valid_params))}",
                            UserWarning,
                        )

                    # Validate using Pydantic (only for valid parameters)
                    valid_kwargs = {
                        k: v for k, v in kwargs.items() if k in valid_params
                    }

                    try:
                        # Pydantic handles missing required fields and type validation
                        validated_data = pydantic_model(**valid_kwargs)
                        # Store the validated data for use in execution
                        self._validated_data = validated_data.model_dump()
                    except ValidationError as e:
                        raise TypeError(self._format_pydantic_errors(e))

                def score(self, *args, **kwargs):
                    """Synchronous scoring method that wraps ascore()."""

                    # Use asyncio.run to execute the async method
                    async def _async_wrapper():
                        return await self.ascore(*args, **kwargs)

                    # Check if we're already in an event loop
                    try:
                        # If we're in a running event loop, we need nest_asyncio for compatibility
                        _ = asyncio.get_running_loop()
                        # Import nest_asyncio style runner from ragas
                        from ragas.async_utils import run

                        return run(_async_wrapper())
                    except RuntimeError:
                        # No running event loop, safe to use asyncio.run
                        return asyncio.run(_async_wrapper())

                async def ascore(self, *args, **kwargs):
                    """Asynchronous scoring method."""
                    # Validate inputs before execution
                    self._validate_inputs(args, kwargs)

                    try:
                        # Use validated data from Pydantic if available
                        func_kwargs = getattr(self, "_validated_data", {})

                        # Execute the function based on its type
                        if is_async:
                            # For async functions, await the result
                            result = await func(**func_kwargs)
                        else:
                            # For sync functions, run directly
                            result = func(**func_kwargs)

                        # Ensure result is a MetricResult
                        if not isinstance(result, MetricResult):
                            # Wrap plain values in MetricResult
                            result = MetricResult(value=result, reason=None)

                        # Validate the result based on metric type
                        validation_error = self._validate_result_value(result.value)
                        if validation_error:
                            return MetricResult(value=None, reason=validation_error)

                        return result

                    except Exception as e:
                        # Handle errors gracefully
                        error_msg = f"Error executing metric {self.name}: {str(e)}"
                        return MetricResult(value=None, reason=error_msg)

                def __call__(self, *args, **kwargs):
                    """Make the metric instance directly callable using the original function."""
                    if self._func is None:
                        raise RuntimeError(
                            "Original function not set on metric instance"
                        )

                    if is_async:
                        # For async functions, always return the coroutine
                        # Let the caller handle async context appropriately
                        return self._func(*args, **kwargs)
                    else:
                        # For sync functions, just call directly
                        return self._func(*args, **kwargs)

                def __repr__(self) -> str:
                    from ragas.metrics.validators import get_metric_type_name

                    param_names = list(sig.parameters.keys())
                    param_str = ", ".join(param_names)

                    metric_type = "CustomMetric"
                    if hasattr(self, "allowed_values"):
                        metric_type = get_metric_type_name(self.allowed_values)

                    allowed_values_str = ""
                    if hasattr(self, "allowed_values"):
                        allowed_values_str = f"[{self.allowed_values!r}]"

                    return (
                        f"{self.name}({param_str}) -> {metric_type}{allowed_values_str}"
                    )

            # Create the metric instance with all parameters
            metric_instance = CustomMetric(name=metric_name)

            # Store metric parameters and original function
            metric_instance._metric_params = metric_params
            metric_instance._func = func

            # Set allowed_values if provided
            if "allowed_values" in metric_params:
                metric_instance.allowed_values = metric_params["allowed_values"]

            # Preserve metadata
            metric_instance.__name__ = metric_name
            metric_instance.__doc__ = func.__doc__

            return metric_instance

        return decorator

    return decorator_factory
