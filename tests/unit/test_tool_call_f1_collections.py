"""Tests for ToolCallF1 metric (collections implementation)."""

import pytest

from ragas.messages import AIMessage, HumanMessage, ToolCall
from ragas.metrics.collections.tool_call_f1 import ToolCallF1


@pytest.fixture
def tool_call_f1():
    """Fixture providing ToolCallF1 instance."""
    return ToolCallF1()


class TestToolCallF1Collections:
    """Test cases for ToolCallF1 metric from collections."""

    @pytest.mark.asyncio
    async def test_perfect_match(self, tool_call_f1):
        """Test perfect match scenario with identical tool calls."""
        ref_tool_calls = [
            ToolCall(name="WeatherForecast", args={"location": "Paris"}),
        ]

        user_input = [
            HumanMessage(content="What is the weather in Paris?"),
            AIMessage(
                content="Let me check the weather forecast",
                tool_calls=[
                    ToolCall(name="WeatherForecast", args={"location": "Paris"})
                ],
            ),
        ]

        result = await tool_call_f1.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_partial_match_missing_prediction(self, tool_call_f1):
        """Test case where prediction has fewer tool calls than reference."""
        ref_tool_calls = [
            ToolCall(name="WeatherForecast", args={"location": "Paris"}),
            ToolCall(name="UVIndex", args={"location": "Paris"}),
        ]

        user_input = [
            HumanMessage(content="Weather info please"),
            AIMessage(
                content="Checking",
                tool_calls=[
                    ToolCall(name="WeatherForecast", args={"location": "Paris"})
                ],
            ),
        ]

        result = await tool_call_f1.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        # TP=1, FP=0, FN=1 -> Precision=1.0, Recall=0.5, F1=0.67
        assert round(result.value, 2) == 0.67

    @pytest.mark.asyncio
    async def test_partial_match_extra_prediction(self, tool_call_f1):
        """Test case where prediction has more tool calls than reference."""
        ref_tool_calls = [
            ToolCall(name="WeatherForecast", args={"location": "Paris"}),
        ]

        user_input = [
            HumanMessage(content="Weather info"),
            AIMessage(
                content="Getting info",
                tool_calls=[
                    ToolCall(name="WeatherForecast", args={"location": "Paris"}),
                    ToolCall(name="AirQuality", args={"location": "Paris"}),
                ],
            ),
        ]

        result = await tool_call_f1.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        # TP=1, FP=1, FN=0 -> Precision=0.5, Recall=1.0, F1=0.67
        assert round(result.value, 2) == 0.67

    @pytest.mark.asyncio
    async def test_no_match(self, tool_call_f1):
        """Test case with no matching tool calls."""
        ref_tool_calls = [
            ToolCall(name="WeatherForecast", args={"location": "Paris"}),
        ]

        user_input = [
            HumanMessage(content="Weather"),
            AIMessage(
                content="Getting data",
                tool_calls=[ToolCall(name="AirQuality", args={"location": "Paris"})],
            ),
        ]

        result = await tool_call_f1.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        # TP=0, FP=1, FN=1 -> F1=0.0
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_multiple_messages(self, tool_call_f1):
        """Test with tool calls spread across multiple messages."""
        ref_tool_calls = [
            ToolCall(name="WeatherForecast", args={"location": "Paris"}),
            ToolCall(name="UVIndex", args={"location": "Paris"}),
        ]

        user_input = [
            HumanMessage(content="Get weather and UV info"),
            AIMessage(
                content="Getting weather",
                tool_calls=[
                    ToolCall(name="WeatherForecast", args={"location": "Paris"})
                ],
            ),
            AIMessage(
                content="Getting UV",
                tool_calls=[ToolCall(name="UVIndex", args={"location": "Paris"})],
            ),
        ]

        result = await tool_call_f1.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_both_empty(self, tool_call_f1):
        """Test case with no tool calls in both predicted and reference."""
        user_input = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]

        result = await tool_call_f1.ascore(
            user_input=user_input,
            reference_tool_calls=[],
        )
        # No predictions, no references -> F1=0.0
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_only_predicted_no_reference(self, tool_call_f1):
        """Test case with predicted tool calls but no reference."""
        user_input = [
            HumanMessage(content="Weather"),
            AIMessage(
                content="Checking",
                tool_calls=[
                    ToolCall(name="WeatherForecast", args={"location": "Paris"})
                ],
            ),
        ]

        result = await tool_call_f1.ascore(
            user_input=user_input,
            reference_tool_calls=[],
        )
        # TP=0, FP=1, FN=0 -> Precision=0.0 -> F1=0.0
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_only_reference_no_predicted(self, tool_call_f1):
        """Test case with reference tool calls but no predictions."""
        ref_tool_calls = [
            ToolCall(name="WeatherForecast", args={"location": "Paris"}),
        ]

        user_input = [
            HumanMessage(content="Weather"),
            AIMessage(content="I don't know"),
        ]

        result = await tool_call_f1.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        # TP=0, FP=0, FN=1 -> Recall=0.0 -> F1=0.0
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_argument_mismatch(self, tool_call_f1):
        """Test case where tool names match but arguments differ."""
        ref_tool_calls = [
            ToolCall(name="WeatherForecast", args={"location": "Paris"}),
        ]

        user_input = [
            HumanMessage(content="Weather"),
            AIMessage(
                content="Checking",
                tool_calls=[
                    ToolCall(name="WeatherForecast", args={"location": "London"})
                ],
            ),
        ]

        result = await tool_call_f1.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        # Different arguments means no match -> TP=0, FP=1, FN=1 -> F1=0.0
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_duplicate_tool_calls_in_prediction(self, tool_call_f1):
        """Test case with duplicate tool calls in prediction."""
        ref_tool_calls = [
            ToolCall(name="WeatherForecast", args={"location": "Paris"}),
        ]

        user_input = [
            HumanMessage(content="Weather"),
            AIMessage(
                content="Checking multiple times",
                tool_calls=[
                    ToolCall(name="WeatherForecast", args={"location": "Paris"}),
                    ToolCall(name="WeatherForecast", args={"location": "Paris"}),
                ],
            ),
        ]

        result = await tool_call_f1.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        # Sets will deduplicate, so TP=1, FP=0, FN=0 -> F1=1.0
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_complex_scenario(self, tool_call_f1):
        """Test complex scenario with multiple correct and incorrect calls."""
        ref_tool_calls = [
            ToolCall(name="WeatherForecast", args={"location": "Paris"}),
            ToolCall(name="UVIndex", args={"location": "Paris"}),
            ToolCall(name="AirQuality", args={"location": "Paris"}),
        ]

        user_input = [
            HumanMessage(content="Get all environmental data"),
            AIMessage(
                content="Fetching data",
                tool_calls=[
                    ToolCall(name="WeatherForecast", args={"location": "Paris"}),
                    ToolCall(name="UVIndex", args={"location": "Paris"}),
                    ToolCall(name="Humidity", args={"location": "Paris"}),
                ],
            ),
        ]

        result = await tool_call_f1.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        # TP=2 (Weather, UV), FP=1 (Humidity), FN=1 (AirQuality)
        # Precision=2/3, Recall=2/3, F1=2/3=0.6667
        assert round(result.value, 2) == 0.67

    @pytest.mark.asyncio
    async def test_input_validation(self, tool_call_f1):
        """Test input validation."""
        with pytest.raises(ValueError, match="user_input must be a list"):
            await tool_call_f1.ascore(
                user_input="not a list",
                reference_tool_calls=[],
            )

        with pytest.raises(ValueError, match="reference_tool_calls must be a list"):
            await tool_call_f1.ascore(
                user_input=[],
                reference_tool_calls="not a list",
            )

    @pytest.mark.asyncio
    async def test_nested_dict_in_args(self, tool_call_f1):
        """Test handling of nested dicts in tool call args (issue #2506)."""
        ref_tool_calls = [
            ToolCall(
                name="store_data",
                args={
                    "title": "Backend Engineer",
                    "kwargs": {},  # Nested empty dict
                },
            ),
        ]

        user_input = [
            HumanMessage(content="Store the data"),
            AIMessage(
                content="Storing...",
                tool_calls=[
                    ToolCall(
                        name="store_data",
                        args={
                            "title": "Backend Engineer",
                            "kwargs": {},
                        },
                    )
                ],
            ),
        ]

        result = await tool_call_f1.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_nested_list_in_args(self, tool_call_f1):
        """Test handling of nested lists in tool call args."""
        ref_tool_calls = [
            ToolCall(
                name="search",
                args={
                    "categories": ["a", "b"],
                    "filters": {"min": 10, "max": 100},
                },
            ),
        ]

        user_input = [
            HumanMessage(content="Search"),
            AIMessage(
                content="Searching...",
                tool_calls=[
                    ToolCall(
                        name="search",
                        args={
                            "categories": ["a", "b"],
                            "filters": {"min": 10, "max": 100},
                        },
                    )
                ],
            ),
        ]

        result = await tool_call_f1.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_deeply_nested_args(self, tool_call_f1):
        """Test handling of deeply nested structures in tool call args."""
        ref_tool_calls = [
            ToolCall(
                name="complex_tool",
                args={
                    "level1": {
                        "level2": {
                            "level3": ["x", "y", "z"],
                        }
                    }
                },
            ),
        ]

        user_input = [
            HumanMessage(content="Do something"),
            AIMessage(
                content="Processing...",
                tool_calls=[
                    ToolCall(
                        name="complex_tool",
                        args={
                            "level1": {
                                "level2": {
                                    "level3": ["x", "y", "z"],
                                }
                            }
                        },
                    )
                ],
            ),
        ]

        result = await tool_call_f1.ascore(
            user_input=user_input,
            reference_tool_calls=ref_tool_calls,
        )
        assert result.value == 1.0
