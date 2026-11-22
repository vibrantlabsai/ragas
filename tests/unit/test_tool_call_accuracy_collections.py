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
    async def test_score_clamped_to_max_one(self, tool_call_accuracy):
        """Test that score never exceeds 1.0 when multiple predictions match one reference.

        Regression test for https://github.com/explodinggradients/ragas/issues/2079
        """
        ref_tool_calls = [ToolCall(name="weather_check", args={"location": "Shanghai"})]

        pred_tool_calls = [
            ToolCall(
                name="weather_check", args={"location": "Shanghai", "time": "9:00"}
            ),
            ToolCall(
                name="weather_check", args={"location": "Shanghai", "time": "13:00"}
            ),
            ToolCall(
                name="weather_check", args={"location": "Shanghai", "time": "21:00"}
            ),
        ]

        user_input = [
            HumanMessage(content="What's the weather in Shanghai at different times?"),
            AIMessage(content="...", tool_calls=pred_tool_calls),
        ]

        with pytest.warns(UserWarning, match="Length mismatch"):
            result = await tool_call_accuracy.ascore(
                user_input=user_input,
                reference_tool_calls=ref_tool_calls,
            )

        assert 0.0 <= result.value <= 1.0, (
            f"Score should be between 0.0 and 1.0, got {result.value}"
        )

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
        # Sequences don't align (different lengths), so score is 0
        assert result.value == 0.0
