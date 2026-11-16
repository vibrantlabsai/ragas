"""Tests for ToolCallAccuracy metric."""

from unittest.mock import AsyncMock

import pytest

from ragas.dataset_schema import MultiTurnSample
from ragas.messages import AIMessage, ToolCall
from ragas.metrics import ToolCallAccuracy


@pytest.fixture
def tool_call_accuracy():
    """Fixture providing ToolCallAccuracy instance."""
    return ToolCallAccuracy()


@pytest.fixture
def mock_callbacks():
    """Fixture providing mock callbacks."""
    return AsyncMock()


class TestToolCallAccuracy:
    """Test cases for ToolCallAccuracy metric."""

    def test_is_sequence_aligned_perfect_match(self, tool_call_accuracy):
        """Test sequence alignment with perfect match."""
        pred_seq = ["func1", "func2", "func3"]
        ref_seq = ["func1", "func2", "func3"]
        assert tool_call_accuracy.is_sequence_aligned(pred_seq, ref_seq) is True

    def test_is_sequence_aligned_different_order(self, tool_call_accuracy):
        """Test sequence alignment with different order."""
        pred_seq = ["func1", "func3", "func2"]
        ref_seq = ["func1", "func2", "func3"]
        assert tool_call_accuracy.is_sequence_aligned(pred_seq, ref_seq) is False

    def test_is_sequence_aligned_different_length(self, tool_call_accuracy):
        """Test sequence alignment with different lengths."""
        pred_seq = ["func1", "func2"]
        ref_seq = ["func1", "func2", "func3"]
        assert tool_call_accuracy.is_sequence_aligned(pred_seq, ref_seq) is False

    def test_is_sequence_aligned_empty_sequences(self, tool_call_accuracy):
        """Test sequence alignment with empty sequences."""
        assert tool_call_accuracy.is_sequence_aligned([], []) is True

    @pytest.mark.asyncio
    async def test_perfect_match_scenario(self, tool_call_accuracy, mock_callbacks):
        """Test perfect match scenario with identical tool calls."""
        # Create reference tool calls
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        # Create predicted tool calls
        pred_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        # Create sample
        sample = MultiTurnSample(
            user_input=[
                AIMessage(content="I'll search for you", tool_calls=pred_tool_calls)
            ],
            reference_tool_calls=ref_tool_calls,
        )

        # Mock the arg comparison to return 1.0 for perfect matches
        tool_call_accuracy.arg_comparison_metric.single_turn_ascore = AsyncMock(
            return_value=1.0
        )

        score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_no_predicted_tool_calls(self, tool_call_accuracy, mock_callbacks):
        """Test case with no predicted tool calls."""
        ref_tool_calls = [ToolCall(name="search", args={"query": "python"})]

        sample = MultiTurnSample(
            user_input=[AIMessage(content="No tool calls here")],
            reference_tool_calls=ref_tool_calls,
        )

        with pytest.warns(UserWarning, match="No tool calls found"):
            score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_sequence_misalignment(self, tool_call_accuracy, mock_callbacks):
        """Test case where sequences don't align."""
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        # Different order - should result in score 0 due to sequence misalignment
        pred_tool_calls = [
            ToolCall(name="filter", args={"type": "recent"}),
            ToolCall(name="search", args={"query": "python"}),
        ]

        sample = MultiTurnSample(
            user_input=[AIMessage(content="Searching...", tool_calls=pred_tool_calls)],
            reference_tool_calls=ref_tool_calls,
        )

        tool_call_accuracy.arg_comparison_metric.single_turn_ascore = AsyncMock(
            return_value=1.0
        )

        score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_length_mismatch_more_predicted_same_name(
        self, tool_call_accuracy, mock_callbacks
    ):
        """Test case with more predicted tool calls than reference (retries).

        When predicted calls contain the reference calls as a subsequence,
        the metric should score based on best matches, not return 0.
        Fixes: https://github.com/vibrantlabsai/ragas/issues/2079
        """
        ref_tool_calls = [ToolCall(name="search", args={"query": "python"})]

        pred_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="search", args={"query": "python"}),  # retry
        ]

        sample = MultiTurnSample(
            user_input=[AIMessage(content="Searching...", tool_calls=pred_tool_calls)],
            reference_tool_calls=ref_tool_calls,
        )

        tool_call_accuracy.arg_comparison_metric.single_turn_ascore = AsyncMock(
            return_value=1.0
        )

        with pytest.warns(UserWarning, match="Length mismatch"):
            score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)

        # Reference "search" is a subsequence of predicted - should score well
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_length_mismatch_more_predicted_different_names(
        self, tool_call_accuracy, mock_callbacks
    ):
        """Test case with more predicted tool calls with different names."""
        ref_tool_calls = [ToolCall(name="search", args={"query": "python"})]

        pred_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),  # extra, different name
        ]

        sample = MultiTurnSample(
            user_input=[AIMessage(content="Searching...", tool_calls=pred_tool_calls)],
            reference_tool_calls=ref_tool_calls,
        )

        tool_call_accuracy.arg_comparison_metric.single_turn_ascore = AsyncMock(
            return_value=1.0
        )

        with pytest.warns(UserWarning, match="Length mismatch"):
            score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)

        # "search" is a subsequence of ["search", "filter"] - should match
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_length_mismatch_fewer_predicted(
        self, tool_call_accuracy, mock_callbacks
    ):
        """Test case with fewer predicted tool calls than reference."""
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        pred_tool_calls = [ToolCall(name="search", args={"query": "python"})]

        sample = MultiTurnSample(
            user_input=[AIMessage(content="Searching...", tool_calls=pred_tool_calls)],
            reference_tool_calls=ref_tool_calls,
        )

        tool_call_accuracy.arg_comparison_metric.single_turn_ascore = AsyncMock(
            return_value=1.0
        )

        with pytest.warns(UserWarning, match="Length mismatch"):
            score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)

        # Should be 0 because reference ["search", "filter"] is NOT a
        # subsequence of predicted ["search"] - missing "filter"
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_score_never_exceeds_one(self, tool_call_accuracy, mock_callbacks):
        """Test that score never exceeds 1.0 even with many duplicate predictions.

        Regression test for: https://github.com/vibrantlabsai/ragas/issues/2079
        """
        ref_tool_calls = [
            ToolCall(name="weather_check", args={"location": "Shanghai"})
        ]

        # Agent made 3 calls (e.g., checking weather at different times)
        pred_tool_calls = [
            ToolCall(name="weather_check", args={"location": "Shanghai", "time": "9:00"}),
            ToolCall(name="weather_check", args={"location": "Shanghai", "time": "13:00"}),
            ToolCall(name="weather_check", args={"location": "Shanghai", "time": "21:00"}),
        ]

        sample = MultiTurnSample(
            user_input=[AIMessage(content="Checking...", tool_calls=pred_tool_calls)],
            reference_tool_calls=ref_tool_calls,
        )

        tool_call_accuracy.arg_comparison_metric.single_turn_ascore = AsyncMock(
            return_value=1.0
        )

        with pytest.warns(UserWarning, match="Length mismatch"):
            score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)

        assert 0.0 <= score <= 1.0, f"Score {score} is out of valid range [0, 1]"

    @pytest.mark.asyncio
    async def test_partial_argument_match(self, tool_call_accuracy, mock_callbacks):
        """Test case with partial argument matches."""
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python", "limit": 10}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        pred_tool_calls = [
            ToolCall(
                name="search", args={"query": "python", "limit": 5}
            ),  # Wrong limit
            ToolCall(name="filter", args={"type": "recent"}),  # Perfect match
        ]

        sample = MultiTurnSample(
            user_input=[AIMessage(content="Searching...", tool_calls=pred_tool_calls)],
            reference_tool_calls=ref_tool_calls,
        )

        # Mock to return scores based on the argument comparison
        # For the "search" tool call: we need to call for each argument
        # For "python" vs "python": 1.0, for 5 vs 10: 0.0 -> average = 0.5
        # For the "filter" tool call: "recent" vs "recent": 1.0 -> average = 1.0
        tool_call_accuracy.arg_comparison_metric.single_turn_ascore = AsyncMock(
            side_effect=[1.0, 0.0, 1.0]  # query match, limit mismatch, type match
        )

        score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)
        assert score == 0.75  # (0.5 + 1.0) / 2

    @pytest.mark.asyncio
    async def test_wrong_tool_names(self, tool_call_accuracy, mock_callbacks):
        """Test case with wrong tool names."""
        ref_tool_calls = [ToolCall(name="search", args={"query": "python"})]

        pred_tool_calls = [ToolCall(name="wrong_tool", args={"query": "python"})]

        sample = MultiTurnSample(
            user_input=[AIMessage(content="Searching...", tool_calls=pred_tool_calls)],
            reference_tool_calls=ref_tool_calls,
        )

        score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)
        assert score == 0.0  # Wrong tool name should result in 0

    @pytest.mark.asyncio
    async def test_multiple_ai_messages(self, tool_call_accuracy, mock_callbacks):
        """Test case with multiple AI messages containing tool calls."""
        ref_tool_calls = [
            ToolCall(name="search", args={"query": "python"}),
            ToolCall(name="filter", args={"type": "recent"}),
        ]

        # Tool calls spread across multiple messages
        sample = MultiTurnSample(
            user_input=[
                AIMessage(
                    content="First",
                    tool_calls=[ToolCall(name="search", args={"query": "python"})],
                ),
                AIMessage(
                    content="Second",
                    tool_calls=[ToolCall(name="filter", args={"type": "recent"})],
                ),
            ],
            reference_tool_calls=ref_tool_calls,
        )

        tool_call_accuracy.arg_comparison_metric.single_turn_ascore = AsyncMock(
            return_value=1.0
        )

        score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_empty_reference_tool_calls(self, tool_call_accuracy, mock_callbacks):
        """Test case with empty reference tool calls and no predictions."""
        sample = MultiTurnSample(
            user_input=[AIMessage(content="No tools needed")],
            reference_tool_calls=[],
        )

        score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)
        assert score == 1.0  # Both empty should be perfect match

    @pytest.mark.asyncio
    async def test_empty_reference_with_predictions(
        self, tool_call_accuracy, mock_callbacks
    ):
        """Test case with empty reference but predictions exist."""
        sample = MultiTurnSample(
            user_input=[
                AIMessage(
                    content="Calling tool",
                    tool_calls=[ToolCall(name="unexpected", args={})],
                )
            ],
            reference_tool_calls=[],
        )

        with pytest.warns(UserWarning, match="Reference tool calls are empty"):
            score = await tool_call_accuracy._multi_turn_ascore(sample, mock_callbacks)
        assert score == 0.0

    def test_metric_name(self, tool_call_accuracy):
        """Test that metric has correct name."""
        assert tool_call_accuracy.name == "tool_call_accuracy"

    def test_required_columns(self, tool_call_accuracy):
        """Test that metric has correct required columns."""
        from ragas.metrics.base import MetricType

        required = tool_call_accuracy._required_columns[MetricType.MULTI_TURN]
        assert "user_input" in required
        assert "reference_tool_calls" in required

    def test_strict_order_parameter_default(self):
        """Test that strict_order defaults to True for backward compatibility."""
        metric = ToolCallAccuracy()
        assert metric.strict_order is True

    def test_strict_order_parameter_explicit(self):
        """Test explicit strict_order parameter setting."""
        strict_metric = ToolCallAccuracy(strict_order=True)
        flexible_metric = ToolCallAccuracy(strict_order=False)

        assert strict_metric.strict_order is True
        assert flexible_metric.strict_order is False

    def test_is_sequence_aligned_flexible_mode(self):
        """Test sequence alignment with flexible ordering."""
        flexible_metric = ToolCallAccuracy(strict_order=False)

        pred_seq = ["func2", "func1", "func3"]
        ref_seq = ["func1", "func2", "func3"]

        # Flexible mode should return True for same elements in different order
        assert flexible_metric.is_sequence_aligned(pred_seq, ref_seq) is True

        # Strict mode should return False for different order
        strict_metric = ToolCallAccuracy(strict_order=True)
        assert strict_metric.is_sequence_aligned(pred_seq, ref_seq) is False

    def test_is_sequence_aligned_strict_subsequence(self):
        """Test strict mode treats reference as subsequence of predicted."""
        metric = ToolCallAccuracy(strict_order=True)

        # Reference is a subsequence of predicted (extra retries)
        assert metric.is_sequence_aligned(
            ["search", "search", "filter"], ["search", "filter"]
        ) is True

        # Reference is NOT a subsequence (wrong order)
        assert metric.is_sequence_aligned(
            ["filter", "search"], ["search", "filter"]
        ) is False

        # Reference has item not in predicted
        assert metric.is_sequence_aligned(
            ["search"], ["search", "filter"]
        ) is False

    def test_is_sequence_aligned_flexible_extra_predictions(self):
        """Test flexible mode with extra predicted calls."""
        flexible = ToolCallAccuracy(strict_order=False)

        # More predicted than reference - all ref names present
        assert flexible.is_sequence_aligned(
            ["search", "search", "filter"], ["search", "filter"]
        ) is True

        # Reference needs 2 search but predicted only has 1
        assert flexible.is_sequence_aligned(
            ["search", "filter"], ["search", "search"]
        ) is False

    def test_flexible_order_sorting_behavior(self):
        """Test that flexible mode sorts tool calls before evaluation."""

        # Test that tool calls get sorted when not in strict order mode
        reference_calls = [
            ToolCall(name="WeatherForecast", args={"location": "Paris"}),
            ToolCall(name="UVIndex", args={"location": "Paris"}),
        ]

        predicted_calls = [
            ToolCall(name="UVIndex", args={"location": "Paris"}),
            ToolCall(name="WeatherForecast", args={"location": "Paris"}),
        ]

        # Test sequence alignment logic directly
        strict_metric = ToolCallAccuracy(strict_order=True)
        flexible_metric = ToolCallAccuracy(strict_order=False)

        # Sequence names for comparison
        pred_seq = [
            call.name for call in predicted_calls
        ]  # ["UVIndex", "WeatherForecast"]
        ref_seq = [
            call.name for call in reference_calls
        ]  # ["WeatherForecast", "UVIndex"]

        # Strict should fail on order
        strict_aligned = strict_metric.is_sequence_aligned(pred_seq, ref_seq)
        assert strict_aligned is False

        # Flexible should pass (sorts both before comparing)
        flexible_aligned = flexible_metric.is_sequence_aligned(pred_seq, ref_seq)
        assert flexible_aligned is True

    def test_sorted_key_for_tool_call(self):
        """Test the sorting key generation for tool calls."""
        tool_call_1 = ToolCall(
            name="WeatherForecast", args={"location": "Paris", "units": "metric"}
        )
        tool_call_2 = ToolCall(
            name="WeatherForecast", args={"units": "metric", "location": "Paris"}
        )

        key_1 = ToolCallAccuracy._sorted_key_for_tool_call(tool_call_1)
        key_2 = ToolCallAccuracy._sorted_key_for_tool_call(tool_call_2)

        # Same content with different arg order should produce same key
        assert key_1 == key_2

        # Different tool call should produce different key
        different_call = ToolCall(name="UVIndex", args={"location": "Paris"})
        key_3 = ToolCallAccuracy._sorted_key_for_tool_call(different_call)
        assert key_1 != key_3
