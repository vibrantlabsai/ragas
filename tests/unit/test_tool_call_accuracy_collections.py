"""Tests for ToolCallAccuracy metric (collections implementation)."""

import pytest

from ragas.messages import AIMessage, HumanMessage, ToolCall
from ragas.metrics.collections import ToolCallAccuracy


@pytest.fixture
def tool_call_accuracy():
    """Fixture providing ToolCallAccuracy instance."""
    return ToolCallAccuracy()


class TestToolCallAccuracyCollections:
    """Test cases for ToolCallAccuracy metric from collections."""

    @pytest.mark.asyncio
    async def test_perfect_match_scenario(self, tool_call_accuracy):
        """Test perfect match scenario with identical tool calls."""
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        user_input = [
            HumanMessage(content="Search for recent python articles"),
            AIMessage(content="I'll search for you", tool_calls=ref_tool_calls),
        ]

        result = await tool_call_accuracy.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_no_predicted_tool_calls(self, tool_call_accuracy):
        """Test case with no predicted tool calls."""
        ref_tool_calls = [ToolCall(name="search", args={"query": "python"})]

        user_input = [
            HumanMessage(content="Search something"),
            AIMessage(content="No tool calls here"),
        ]

        with pytest.warns(UserWarning, match="No tool calls found"):
            result = await tool_call_accuracy.ascore(
                user_input=user_input,
                reference_tool_calls=ref_tool_calls,
            )
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_sequence_misalignment_strict_order(self, tool_call_accuracy):
        """Test case where sequences don't align in strict order mode."""
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        pred_tool_calls = [
            ToolCall(name="filter", args={"type": "recent"}),
            ToolCall(name="search", args={"query": "python"}),
        ]

        user_input = [
            HumanMessage(content="Do a search"),
            AIMessage(content="Searching...", tool_calls=pred_tool_calls),
        ]

        result = await tool_call_accuracy.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_flexible_order_mode(self):
        """Test case with flexible order mode enabled."""
        metric = ToolCallAccuracy(strict_order=False)

        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        pred_tool_calls = [
            ToolCall(name="filter", args={"type": "recent"}),
            ToolCall(name="search", args={"query": "python"}),
        ]

        user_input = [
            HumanMessage(content="Do a search"),
            AIMessage(content="Searching...", tool_calls=pred_tool_calls),
        ]

        result = await metric.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_partial_argument_match(self, tool_call_accuracy):
        """Test case with partial argument matches."""
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python", "limit": 10}),
        ]

        pred_tool_calls = [
            ToolCall(name="search", args={"query": "python", "limit": 5}),
        ]

        user_input = [
            HumanMessage(content="Search"),
            AIMessage(content="Searching...", tool_calls=pred_tool_calls),
        ]

        result = await tool_call_accuracy.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        # Should be 0.5 because only 1 of 2 args match
        assert result.value == 0.5

    @pytest.mark.asyncio
    async def test_both_empty(self, tool_call_accuracy):
        """Test case with both predicted and reference empty."""
        user_input = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]

        result = await tool_call_accuracy.ascore(
            user_input=user_input,
            reference_tool_calls=[],
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_length_mismatch(self, tool_call_accuracy):
        """Test case with length mismatch."""
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        pred_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
        ]

        user_input = [
            HumanMessage(content="Search"),
            AIMessage(content="Searching...", tool_calls=pred_tool_calls),
        ]

        with pytest.warns(UserWarning, match="Length mismatch"):
            result = await tool_call_accuracy.ascore(
                user_input=user_input,
                reference_tool_calls=ref_tool_calls,
            )
        # Overlapping portion (first call) aligns and its args match:
        # mean arg score 1.0, coverage penalty 1/2 for the missing call.
        assert result.value == 0.5

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "strict_order,pred_calls,ref_calls,expected",
        [
            # equal length: unchanged behavior
            (
                True,
                [("a", {"x": 1}), ("b", {"y": 2})],
                [("a", {"x": 1}), ("b", {"y": 2})],
                1.0,
            ),
            (True, [("b", {}), ("a", {})], [("a", {}), ("b", {})], 0.0),
            (False, [("b", {}), ("a", {})], [("a", {}), ("b", {})], 1.0),
            # length mismatch: overlap aligned -> partial credit (#3079),
            # extra predictions still penalized (#1893)
            (True, [("a", {"x": 1}), ("b", {})], [("a", {"x": 1})], 0.5),
            (True, [("a", {"x": 1})], [("a", {"x": 1}), ("b", {})], 0.5),
            (False, [("a", {"x": 1}), ("b", {})], [("a", {"x": 1})], 0.5),
            # misaligned overlapping portion still scores 0
            (True, [("b", {}), ("a", {})], [("a", {})], 0.0),
            # duplicates
            (True, [("a", {}), ("a", {})], [("a", {})], 0.5),
            (True, [("a", {}), ("a", {})], [("a", {}), ("a", {})], 1.0),
            # flexible mode pairs positionally after sorting: a reference
            # subset past the first min-length positions does not match
            (False, [("a", {}), ("b", {}), ("c", {})], [("b", {}), ("c", {})], 0.0),
        ],
    )
    async def test_scoring_matrix(self, strict_order, pred_calls, ref_calls, expected):
        """Regression matrix for alignment, length mismatch, and duplicates."""
        metric = ToolCallAccuracy(strict_order=strict_order)
        pred = [ToolCall(name=name, args=args) for name, args in pred_calls]
        ref = [ToolCall(name=name, args=args) for name, args in ref_calls]
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = await metric.ascore(
                user_input=[AIMessage(content="Working...", tool_calls=pred)],
                reference_tool_calls=ref,
            )
        assert result.value == expected
