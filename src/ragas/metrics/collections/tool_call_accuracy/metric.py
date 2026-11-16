"""Tool Call Accuracy metric - Modern collections implementation."""

import typing as t
import warnings
from typing import List

from ragas.messages import AIMessage, ToolCall
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import exact_match_args, sorted_key_for_tool_call

if t.TYPE_CHECKING:
    from ragas.messages import HumanMessage, ToolMessage


class ToolCallAccuracy(BaseMetric):
    """
    Modern implementation of Tool Call Accuracy metric.

    Measures how accurately an LLM agent makes tool calls compared to reference tool calls.
    This is a rule-based metric that evaluates:
    1. Sequence alignment: Whether predicted and reference tool calls match in the required order
    2. Argument accuracy: How well tool call arguments match between predicted and reference

    The metric supports two evaluation modes:
    - Strict order (default): Tool calls must match exactly in sequence
    - Flexible order: Tool calls can be in any order (parallel evaluation)

    Score calculation:
    - If sequences don't align: score = 0
    - If sequences align: score = (average argument accuracy) * sequence_alignment_factor
    - Length mismatches apply proportional coverage penalty

    Usage:
        >>> from ragas.metrics.collections import ToolCallAccuracy
        >>> from ragas.messages import HumanMessage, AIMessage, ToolCall
        >>>
        >>> metric = ToolCallAccuracy(strict_order=True)
        >>>
        >>> result = await metric.ascore(
        ...     user_input=[
        ...         HumanMessage(content="What's the weather in Paris?"),
        ...         AIMessage(
        ...             content="Let me check",
        ...             tool_calls=[ToolCall(name="get_weather", args={"location": "Paris"})]
        ...         )
        ...     ],
        ...     reference_tool_calls=[
        ...         ToolCall(name="get_weather", args={"location": "Paris"})
        ...     ]
        ... )
        >>> print(f"Tool Call Accuracy: {result.value}")

    Attributes:
        strict_order: If True (default), tool calls must match exactly in sequence.
                     If False, tool calls can be in any order.
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
    """

    def __init__(
        self,
        strict_order: bool = True,
        name: str = "tool_call_accuracy",
        **kwargs,
    ):
        """
        Initialize ToolCallAccuracy metric.

        Args:
            strict_order: If True, tool calls must match exactly in sequence.
                         If False, tool calls can be in any order (default: True)
            name: The metric name (default: "tool_call_accuracy")
            **kwargs: Additional arguments passed to BaseMetric
        """
        self.strict_order = strict_order
        super().__init__(name=name, **kwargs)

    def _is_sequence_aligned(
        self, pred_sequence: List[str], ref_sequence: List[str]
    ) -> bool:
        """Check if tool call sequences are aligned.

        In strict mode, checks if reference is an exact match or a subsequence
        of predicted (to handle extra/retried tool calls).
        In flexible mode, checks multiset containment (all reference names
        appear in predicted with sufficient count).
        """
        if self.strict_order:
            if len(pred_sequence) == len(ref_sequence):
                return pred_sequence == ref_sequence
            # When lengths differ, check if reference is a subsequence of
            # predicted (handles extra/retried tool calls gracefully).
            return self._is_subsequence(ref_sequence, pred_sequence)
        else:
            from collections import Counter

            pred_counts = Counter(pred_sequence)
            ref_counts = Counter(ref_sequence)
            return all(
                pred_counts[name] >= count for name, count in ref_counts.items()
            )

    @staticmethod
    def _is_subsequence(subseq: List[str], seq: List[str]) -> bool:
        """Check if subseq appears in order within seq."""
        it = iter(seq)
        return all(item in it for item in subseq)

    async def ascore(
        self,
        user_input: List[t.Union["HumanMessage", "AIMessage", "ToolMessage"]],
        reference_tool_calls: List[ToolCall],
    ) -> MetricResult:
        """
        Calculate tool call accuracy score asynchronously.

        Args:
            user_input: List of conversation messages (HumanMessage, AIMessage, ToolMessage)
            reference_tool_calls: List of expected tool calls

        Returns:
            MetricResult with accuracy score (0.0-1.0, higher is better)
        """
        # Input validation
        if not isinstance(user_input, list):
            raise ValueError("user_input must be a list of messages")
        if not isinstance(reference_tool_calls, list):
            raise ValueError("reference_tool_calls must be a list")

        # Extract predicted tool calls from AI messages
        pred_tool_calls = []
        for item in user_input:
            if isinstance(item, AIMessage) and item.tool_calls is not None:
                pred_tool_calls.extend(item.tool_calls)

        # Handle edge cases
        if not pred_tool_calls and not reference_tool_calls:
            return MetricResult(value=1.0)
        elif not pred_tool_calls:
            warnings.warn("No tool calls found in the user input")
            return MetricResult(value=0.0)
        elif not reference_tool_calls:
            warnings.warn("Reference tool calls are empty but predictions exist")
            return MetricResult(value=0.0)

        # Sort tool calls if not using strict order
        if not self.strict_order:
            pred_tool_calls = sorted(pred_tool_calls, key=sorted_key_for_tool_call)
            reference_tool_calls = sorted(
                reference_tool_calls, key=sorted_key_for_tool_call
            )

        # Check for length mismatch
        if len(pred_tool_calls) != len(reference_tool_calls):
            warnings.warn(
                f"Length mismatch: predicted tool calls ({len(pred_tool_calls)}) "
                f"vs reference tool calls ({len(reference_tool_calls)}). "
            )

        # Extract sequences and check alignment
        tool_call_pred_sequence = [tc.name for tc in pred_tool_calls]
        tool_call_ref_sequence = [tc.name for tc in reference_tool_calls]

        sequence_aligned = int(
            self._is_sequence_aligned(tool_call_pred_sequence, tool_call_ref_sequence)
        )

        # Use best-match scoring: for each reference tool call, find the
        # predicted tool call with the highest argument score. Each predicted
        # call can only be matched once to avoid double-counting.
        score = 0.0
        remaining_preds = list(pred_tool_calls)

        for ref_tool_call in reference_tool_calls:
            best_score = 0.0
            best_idx = -1
            for idx, pred_tool_call in enumerate(remaining_preds):
                if pred_tool_call.name == ref_tool_call.name:
                    arg_score = exact_match_args(
                        pred_tool_call.args, ref_tool_call.args
                    )
                    if arg_score > best_score:
                        best_score = arg_score
                        best_idx = idx
            score += best_score
            if best_idx >= 0:
                remaining_preds.pop(best_idx)

        # Normalize by reference length
        score /= len(reference_tool_calls)

        # Apply coverage penalty when fewer predicted calls than reference
        if len(pred_tool_calls) < len(reference_tool_calls):
            coverage_penalty = len(pred_tool_calls) / len(reference_tool_calls)
            score *= coverage_penalty

        # Apply sequence alignment factor and clamp to valid range
        final_score = min(1.0, max(0.0, score * sequence_aligned))

        return MetricResult(value=float(final_score))
