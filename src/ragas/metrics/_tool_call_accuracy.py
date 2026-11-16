from __future__ import annotations

import typing as t
import warnings
from dataclasses import dataclass, field

from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.messages import AIMessage, ToolCall
from ragas.metrics._string import ExactMatch
from ragas.metrics.base import MetricType, MultiTurnMetric, SingleTurnMetric

if t.TYPE_CHECKING:
    from langchain_core.callbacks.base import Callbacks


@dataclass
class ToolCallAccuracy(MultiTurnMetric):
    """
    Tool Call Accuracy metric measures how accurately an LLM agent makes tool calls
    compared to reference tool calls.

    The metric supports two evaluation modes:
    1. Strict order (default): Tool calls must match exactly in sequence
    2. Flexible order: Tool calls can be in any order (parallel evaluation)

    The metric evaluates two aspects:
    1. Sequence alignment: Whether predicted and reference tool calls match in the required order
    2. Argument accuracy: How well tool call arguments match between predicted and reference

    Score calculation:
    - If sequences don't align: score = 0
    - If sequences align: score = (average argument accuracy) * sequence_alignment_factor
    - Length mismatches result in warnings and proportional penalty

    Edge cases:
    - No predicted tool calls: returns 0.0
    - Length mismatch: compares only the overlapping portion and applies coverage penalty
    - Missing arguments: contributes 0 to the argument score for that tool call

    The final score is always between 0.0 and 1.0.

    Args:
        strict_order: If True (default), tool calls must match exactly in sequence.
                     If False, tool calls can be in any order (parallel evaluation).
    """

    name: str = "tool_call_accuracy"
    strict_order: bool = True
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.MULTI_TURN: {
                "user_input",
                "reference_tool_calls",
            }
        }
    )

    arg_comparison_metric: SingleTurnMetric = field(
        default_factory=lambda: ExactMatch()
    )

    def init(self, run_config):
        pass

    async def _get_arg_score(
        self, preds: t.Dict[str, t.Any], refs: t.Dict[str, t.Any], callbacks: Callbacks
    ) -> float:
        if not refs and not preds:
            return 1.0
        if not refs:
            return 0.0

        score = 0.0
        for arg in refs.keys():
            if arg in preds:
                score += await self.arg_comparison_metric.single_turn_ascore(
                    SingleTurnSample(
                        response=str(preds[arg]), reference=str(refs[arg])
                    ),
                    callbacks,
                )

        return score / len(refs.keys())

    @staticmethod
    def _sorted_key_for_tool_call(tc: ToolCall) -> t.Tuple[str, ...]:
        """
        Generate a consistent sorting key for tool calls.

        This ensures tool calls with the same content are compared correctly
        regardless of argument order in the original call.
        """
        key_list = [tc.name]
        args = tc.args
        args_name = sorted(args)
        for name in args_name:
            key_list.append(name)
            key_list.append(str(args[name]))
        return tuple(key_list)

    def is_sequence_aligned(
        self, pred_sequence: t.List[str], ref_sequence: t.List[str]
    ) -> bool:
        if self.strict_order:
            if len(pred_sequence) == len(ref_sequence):
                return pred_sequence == ref_sequence
            # When lengths differ, check if reference is a subsequence of
            # predicted (handles extra/retried tool calls gracefully).
            return self._is_subsequence(ref_sequence, pred_sequence)
        else:
            # For non-strict mode, check that every reference tool name
            # has a matching predicted tool name (multiset containment).
            from collections import Counter

            pred_counts = Counter(pred_sequence)
            ref_counts = Counter(ref_sequence)
            return all(
                pred_counts[name] >= count for name, count in ref_counts.items()
            )

    @staticmethod
    def _is_subsequence(subseq: t.List[str], seq: t.List[str]) -> bool:
        """Check if subseq appears in order within seq (not necessarily contiguous)."""
        it = iter(seq)
        return all(item in it for item in subseq)

    async def _find_best_match_score(
        self,
        ref_tool_call: ToolCall,
        candidates: t.List[ToolCall],
        callbacks: Callbacks,
    ) -> t.Tuple[float, int]:
        """Find the best matching predicted tool call for a reference tool call.

        Returns:
            Tuple of (best_score, best_index) where best_index is the index
            in candidates. Returns (0.0, -1) if no match found.
        """
        best_score = 0.0
        best_idx = -1
        for idx, pred_tool_call in enumerate(candidates):
            if pred_tool_call.name == ref_tool_call.name:
                arg_score = await self._get_arg_score(
                    pred_tool_call.args, ref_tool_call.args, callbacks
                )
                if arg_score > best_score:
                    best_score = arg_score
                    best_idx = idx
        return best_score, best_idx

    async def _multi_turn_ascore(
        self, sample: MultiTurnSample, callbacks: Callbacks
    ) -> float:
        assert sample.reference_tool_calls is not None, (
            "Reference tool calls is not set"
        )

        pred_tool_calls = []
        for item in sample.user_input:
            if isinstance(item, AIMessage) and item.tool_calls is not None:
                pred_tool_calls.extend(item.tool_calls)

        reference_tool_calls = sample.reference_tool_calls

        # Handle edge cases
        if not pred_tool_calls and not reference_tool_calls:
            # Both empty - perfect match
            return 1.0
        elif not pred_tool_calls:
            warnings.warn("No tool calls found in the user input")
            return 0.0
        elif not reference_tool_calls:
            # Reference is empty but we have predictions - this is typically an error in test data
            warnings.warn("Reference tool calls are empty but predictions exist")
            return 0.0

        # Sort tool calls if not using strict order
        if not self.strict_order:
            pred_tool_calls = sorted(
                pred_tool_calls, key=self._sorted_key_for_tool_call
            )
            reference_tool_calls = sorted(
                reference_tool_calls, key=self._sorted_key_for_tool_call
            )

        # Check for length mismatch and warn user
        if len(pred_tool_calls) != len(reference_tool_calls):
            warnings.warn(
                f"Length mismatch: predicted tool calls ({len(pred_tool_calls)}) "
                f"vs reference tool calls ({len(reference_tool_calls)}). "
            )

        tool_call_pred_sequence = [tool_call.name for tool_call in pred_tool_calls]
        tool_call_ref_sequence = [tool_call.name for tool_call in reference_tool_calls]

        sequence_aligned = int(
            self.is_sequence_aligned(tool_call_pred_sequence, tool_call_ref_sequence)
        )

        # Use best-match scoring: for each reference tool call, find the
        # predicted tool call with the highest argument score. Each predicted
        # call can only be matched once to avoid double-counting.
        score = 0.0
        remaining_preds = list(pred_tool_calls)

        for ref_tool_call in reference_tool_calls:
            best_score, best_idx = await self._find_best_match_score(
                ref_tool_call, remaining_preds, callbacks
            )
            score += best_score
            if best_idx >= 0:
                remaining_preds.pop(best_idx)

        score /= len(reference_tool_calls)

        # Apply coverage penalty when fewer predicted calls than reference
        if len(pred_tool_calls) < len(reference_tool_calls):
            coverage_penalty = len(pred_tool_calls) / len(reference_tool_calls)
            score *= coverage_penalty

        # Clamp to valid range as a safety net
        return min(1.0, max(0.0, score * sequence_aligned))

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._multi_turn_ascore(MultiTurnSample(**row), callbacks)
