"""Tool Call F1 metric - Modern collections implementation."""

import typing as t

from ragas.messages import AIMessage
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import calculate_f1_score, tool_call_to_hashable

if t.TYPE_CHECKING:
    from ragas.messages import HumanMessage, ToolCall, ToolMessage


class ToolCallF1(BaseMetric):
    """
    Modern implementation of Tool Call F1 metric.

    Measures the F1 score between predicted and reference tool calls. This metric
    treats tool calls as a set, comparing the exact match of tool names and their
    arguments using set-based precision and recall.

    The F1 score is calculated as:
    - Precision = TP / (TP + FP) where TP = true positives, FP = false positives
    - Recall = TP / (TP + FN) where FN = false negatives
    - F1 = 2 * (Precision * Recall) / (Precision + Recall)

    A tool call is considered a match only if both the tool name and all arguments
    match exactly between predicted and reference.

    Usage:
        >>> from ragas.metrics.collections import ToolCallF1
        >>> from ragas.messages import HumanMessage, AIMessage, ToolCall
        >>>
        >>> metric = ToolCallF1()
        >>>
        >>> result = await metric.ascore(
        ...     user_input=[
        ...         HumanMessage(content="What's the weather in Paris?"),
        ...         AIMessage(
        ...             content="Let me check",
        ...             tool_calls=[
        ...                 ToolCall(name="get_weather", args={"location": "Paris"}),
        ...                 ToolCall(name="get_uv_index", args={"location": "Paris"})
        ...             ]
        ...         )
        ...     ],
        ...     reference_tool_calls=[
        ...         ToolCall(name="get_weather", args={"location": "Paris"})
        ...     ]
        ... )
        >>> print(f"Tool Call F1: {result.value}")  # 0.67 (1 TP, 1 FP, 0 FN)

    Attributes:
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
    """

    def __init__(self, name: str = "tool_call_f1", **kwargs):
        """
        Initialize ToolCallF1 metric.

        Args:
            name: The metric name (default: "tool_call_f1")
            **kwargs: Additional arguments passed to BaseMetric
        """
        super().__init__(name=name, **kwargs)

    async def ascore(
        self,
        user_input: t.List[t.Union["HumanMessage", "AIMessage", "ToolMessage"]],
        reference_tool_calls: t.List["ToolCall"],
    ) -> MetricResult:
        """
        Calculate tool call F1 score asynchronously.

        Args:
            user_input: List of conversation messages (HumanMessage, AIMessage, ToolMessage)
            reference_tool_calls: List of expected tool calls

        Returns:
            MetricResult with F1 score (0.0-1.0, higher is better)
        """
        # Input validation
        if not isinstance(user_input, list):
            raise ValueError("user_input must be a list of messages")
        if not isinstance(reference_tool_calls, list):
            raise ValueError("reference_tool_calls must be a list")

        # Convert reference tool calls to set
        expected: t.Set[t.Tuple[str, t.FrozenSet]] = set()
        for call in reference_tool_calls:
            expected.add(tool_call_to_hashable(call))

        # Extract and convert predicted tool calls to set
        actual: t.Set[t.Tuple[str, t.FrozenSet]] = set()
        for msg in user_input:
            if isinstance(msg, AIMessage) and msg.tool_calls is not None:
                for call in msg.tool_calls:
                    actual.add(tool_call_to_hashable(call))

        # Calculate set-based metrics
        true_positives = len(actual & expected)
        false_positives = len(actual - expected)
        false_negatives = len(expected - actual)

        # Calculate F1 score
        f1_score = calculate_f1_score(true_positives, false_positives, false_negatives)

        return MetricResult(value=round(f1_score, 4))
