"""Tests for metric decorators (discrete_metric, numeric_metric, ranking_metric)

This module tests that the decorators can handle both:
1. Functions returning plain values (strings, floats, lists)
2. Functions returning MetricResult objects

Following TDD approach: Write failing tests first, then implement the fix.
"""

import typing as t
import warnings

import pydantic
import pytest

from ragas.metrics import MetricResult, discrete_metric, numeric_metric, ranking_metric


class TestDiscreteMetric:
    """Tests for discrete_metric decorator."""

    def test_discrete_metric_with_plain_string_return(self):
        """Test discrete metric with function returning plain string."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str) -> str:
            return "pass" if predicted.lower() == expected.lower() else "fail"

        # This should work without errors
        result = my_metric.score(predicted="test", expected="test")

        assert isinstance(result, MetricResult)
        assert result.value == "pass"
        assert result.reason is None  # Should be None for plain value returns

    def test_discrete_metric_with_plain_string_fail(self):
        """Test discrete metric returning 'fail'."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str) -> str:
            return "pass" if predicted.lower() == expected.lower() else "fail"

        result = my_metric.score(predicted="hello", expected="world")

        assert isinstance(result, MetricResult)
        assert result.value == "fail"
        assert result.reason is None

    def test_discrete_metric_with_metric_result_return(self):
        """Test discrete metric with function returning MetricResult."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str) -> MetricResult:
            value = "pass" if predicted.lower() == expected.lower() else "fail"
            reason = f"Compared '{predicted}' with '{expected}'"
            return MetricResult(value=value, reason=reason)

        result = my_metric.score(predicted="test", expected="test")

        assert isinstance(result, MetricResult)
        assert result.value == "pass"
        assert result.reason == "Compared 'test' with 'test'"

    def test_discrete_metric_validation_invalid_value(self):
        """Test discrete metric validation with invalid value."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str) -> str:
            return "maybe"  # Invalid value

        result = my_metric.score(predicted="test", expected="test")

        assert isinstance(result, MetricResult)
        assert result.value is None
        assert "expected one of ['pass', 'fail']" in result.reason

    @pytest.mark.asyncio
    async def test_discrete_metric_async_with_plain_return(self):
        """Test async discrete metric with plain string return."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        async def my_metric(predicted: str, expected: str) -> str:
            return "pass" if predicted.lower() == expected.lower() else "fail"

        result = await my_metric.ascore(predicted="test", expected="test")

        assert isinstance(result, MetricResult)
        assert result.value == "pass"
        assert result.reason is None


class TestNumericMetric:
    """Tests for numeric_metric decorator."""

    def test_numeric_metric_with_plain_float_return(self):
        """Test numeric metric with function returning plain float."""

        @numeric_metric(name="response_accuracy", allowed_values=(0, 1))
        def my_metric(predicted: float, expected: float) -> float:
            return abs(predicted - expected) / max(expected, 1e-5)

        result = my_metric.score(predicted=0.8, expected=1.0)

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, float)
        assert abs(result.value - 0.2) < 1e-10
        assert result.reason is None

    def test_numeric_metric_with_metric_result_return(self):
        """Test numeric metric with function returning MetricResult."""

        @numeric_metric(name="response_accuracy", allowed_values=(0, 1))
        def my_metric(predicted: float, expected: float) -> MetricResult:
            value = abs(predicted - expected) / max(expected, 1e-5)
            reason = f"Difference: {abs(predicted - expected)}"
            return MetricResult(value=value, reason=reason)

        result = my_metric.score(predicted=0.8, expected=1.0)

        assert isinstance(result, MetricResult)
        assert abs(result.value - 0.2) < 1e-10
        assert result.reason == "Difference: 0.19999999999999996"

    def test_numeric_metric_validation_out_of_range(self):
        """Test numeric metric validation with out-of-range value."""

        @numeric_metric(name="response_accuracy", allowed_values=(0, 1))
        def my_metric(predicted: float, expected: float) -> float:
            return 1.5  # Out of range

        result = my_metric.score(predicted=0.8, expected=1.0)

        assert isinstance(result, MetricResult)
        assert result.value is None
        assert "expected value in range (0, 1)" in result.reason

    @pytest.mark.asyncio
    async def test_numeric_metric_async_with_plain_return(self):
        """Test async numeric metric with plain float return."""

        @numeric_metric(name="response_accuracy", allowed_values=(0, 1))
        async def my_metric(predicted: float, expected: float) -> float:
            return abs(predicted - expected) / max(expected, 1e-5)

        result = await my_metric.ascore(predicted=0.8, expected=1.0)

        assert isinstance(result, MetricResult)
        assert abs(result.value - 0.2) < 1e-10
        assert result.reason is None


class TestRankingMetric:
    """Tests for ranking_metric decorator."""

    def test_ranking_metric_with_plain_list_return(self):
        """Test ranking metric with function returning plain list."""

        @ranking_metric(name="response_ranking", allowed_values=3)
        def my_metric(responses: list) -> list:
            response_lengths = [len(response) for response in responses]
            sorted_indices = sorted(
                range(len(response_lengths)), key=lambda i: response_lengths[i]
            )
            return sorted_indices

        result = my_metric.score(
            responses=["short", "a bit longer", "the longest response"]
        )

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, list)
        assert result.value == [0, 1, 2]  # indices sorted by length
        assert result.reason is None

    def test_ranking_metric_with_metric_result_return(self):
        """Test ranking metric with function returning MetricResult."""

        @ranking_metric(name="response_ranking", allowed_values=3)
        def my_metric(responses: list) -> MetricResult:
            response_lengths = [len(response) for response in responses]
            sorted_indices = sorted(
                range(len(response_lengths)), key=lambda i: response_lengths[i]
            )
            reason = f"Sorted by lengths: {response_lengths}"
            return MetricResult(value=sorted_indices, reason=reason)

        result = my_metric.score(
            responses=["short", "a bit longer", "the longest response"]
        )

        assert isinstance(result, MetricResult)
        assert result.value == [0, 1, 2]
        assert result.reason == "Sorted by lengths: [5, 12, 20]"

    def test_ranking_metric_validation_wrong_length(self):
        """Test ranking metric validation with wrong list length."""

        @ranking_metric(name="response_ranking", allowed_values=3)
        def my_metric(responses: list) -> list:
            return [0, 1]  # Wrong length - should be 3

        result = my_metric.score(responses=["short", "medium", "long"])

        assert isinstance(result, MetricResult)
        assert result.value is None
        assert "expected 3 items" in result.reason

    @pytest.mark.asyncio
    async def test_ranking_metric_async_with_plain_return(self):
        """Test async ranking metric with plain list return."""

        @ranking_metric(name="response_ranking", allowed_values=2)
        async def my_metric(responses: list) -> list:
            return [1, 0]  # Reverse order

        result = await my_metric.ascore(responses=["first", "second"])

        assert isinstance(result, MetricResult)
        assert result.value == [1, 0]
        assert result.reason is None


class TestDirectCallable:
    """Test that decorated metrics are directly callable using the original function."""

    def test_discrete_metric_direct_call_with_plain_return(self):
        """Test that decorated discrete metric can be called directly."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str) -> str:
            return "pass" if predicted.lower() == expected.lower() else "fail"

        # Direct call should work and return the raw function result
        result = my_metric("test", "test")
        assert result == "pass"  # Should return plain string, not MetricResult

        result = my_metric("hello", "world")
        assert result == "fail"

    def test_discrete_metric_direct_call_with_metric_result_return(self):
        """Test direct call when function returns MetricResult."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str) -> MetricResult:
            value = "pass" if predicted.lower() == expected.lower() else "fail"
            reason = f"Compared '{predicted}' with '{expected}'"
            return MetricResult(value=value, reason=reason)

        # Direct call should return MetricResult as the original function does
        result = my_metric("test", "test")
        assert isinstance(result, MetricResult)
        assert result.value == "pass"
        assert result.reason == "Compared 'test' with 'test'"

    def test_numeric_metric_direct_call(self):
        """Test that decorated numeric metric can be called directly."""

        @numeric_metric(name="response_accuracy", allowed_values=(0, 1))
        def my_metric(predicted: float, expected: float) -> float:
            return abs(predicted - expected) / max(expected, 1e-5)

        # Direct call should work and return the raw function result
        result = my_metric(0.8, 1.0)
        assert isinstance(result, float)
        assert abs(result - 0.2) < 1e-10

    def test_ranking_metric_direct_call(self):
        """Test that decorated ranking metric can be called directly."""

        @ranking_metric(name="response_ranking", allowed_values=3)
        def my_metric(responses: list) -> list:
            response_lengths = [len(response) for response in responses]
            sorted_indices = sorted(
                range(len(response_lengths)), key=lambda i: response_lengths[i]
            )
            return sorted_indices

        # Direct call should work and return the raw function result
        result = my_metric(["short", "a bit longer", "the longest response"])
        assert isinstance(result, list)
        assert result == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_async_discrete_metric_direct_call(self):
        """Test that decorated async metric can be called directly."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        async def my_metric(predicted: str, expected: str) -> str:
            return "pass" if predicted.lower() == expected.lower() else "fail"

        # Direct call should work and return a coroutine that can be awaited
        result = await my_metric("test", "test")
        assert result == "pass"

    def test_direct_call_vs_score_method(self):
        """Test that direct call returns raw result while score method returns MetricResult."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str) -> str:
            return "pass" if predicted.lower() == expected.lower() else "fail"

        # Direct call returns raw result
        direct_result = my_metric("test", "test")
        assert direct_result == "pass"
        assert not isinstance(direct_result, MetricResult)

        # Score method returns MetricResult
        score_result = my_metric.score(predicted="test", expected="test")
        assert isinstance(score_result, MetricResult)
        assert score_result.value == "pass"

    def test_direct_call_with_positional_args(self):
        """Test that direct call allows positional arguments like the original function."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str) -> str:
            return "pass" if predicted.lower() == expected.lower() else "fail"

        # Direct call should allow positional arguments
        result = my_metric("test", "test")
        assert result == "pass"

    def test_direct_call_handles_function_errors(self):
        """Test that direct call propagates function errors normally."""

        @discrete_metric(name="error_metric", allowed_values=["pass", "fail"])
        def error_metric(should_error: bool) -> str:
            if should_error:
                raise ValueError("Test error from original function")
            return "pass"

        # Direct call should propagate the error normally
        with pytest.raises(ValueError, match="Test error from original function"):
            error_metric(True)

        # Should work normally when no error
        result = error_metric(False)
        assert result == "pass"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_discrete_metric_with_custom_allowed_values(self):
        """Test discrete metric with custom allowed values."""

        @discrete_metric(
            name="sentiment", allowed_values=["positive", "negative", "neutral"]
        )
        def sentiment_metric(text: str) -> str:
            if "good" in text.lower():
                return "positive"
            elif "bad" in text.lower():
                return "negative"
            else:
                return "neutral"

        result = sentiment_metric.score(text="This is good")
        assert result.value == "positive"

        result = sentiment_metric.score(text="This is bad")
        assert result.value == "negative"

        result = sentiment_metric.score(text="This is okay")
        assert result.value == "neutral"

    def test_numeric_metric_with_range_type(self):
        """Test numeric metric with range type."""

        @numeric_metric(name="score", allowed_values=range(0, 11))  # 0-10
        def score_metric(value: int) -> int:
            return min(10, max(0, value))

        result = score_metric.score(value=5)
        assert result.value == 5

        result = score_metric.score(value=15)  # Should be clamped to 10
        assert result.value == 10

    def test_function_with_no_parameters(self):
        """Test metric function with no parameters."""

        @discrete_metric(name="constant", allowed_values=["always_pass"])
        def constant_metric() -> str:
            return "always_pass"

        result = constant_metric.score()
        assert result.value == "always_pass"

    def test_function_with_exception(self):
        """Test that exceptions are handled gracefully."""

        @discrete_metric(name="error_metric", allowed_values=["pass", "fail"])
        def error_metric(should_error: bool) -> str:
            if should_error:
                raise ValueError("Test error")
            return "pass"

        # Should not raise exception, should return error result
        result = error_metric.score(should_error=True)

        assert isinstance(result, MetricResult)
        assert result.value is None
        assert "Error executing metric" in result.reason
        assert "Test error" in result.reason


class TestErrorHandling:
    """Test comprehensive error handling and validation."""

    def test_positional_arguments_error(self):
        """Test that positional arguments give helpful error message."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str) -> str:
            return "pass" if predicted.lower() == expected.lower() else "fail"

        with pytest.raises(TypeError) as exc_info:
            my_metric.score("test", "test")

        error_msg = str(exc_info.value)
        assert "requires keyword arguments, not positional" in error_msg
        assert "You provided: score('test', 'test')" in error_msg
        assert "Correct usage: score(predicted='test', expected='test')" in error_msg
        assert "💡 Tip:" in error_msg

    def test_missing_required_arguments_error(self):
        """Test error message for missing required arguments."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str, context: str) -> str:
            return "pass"

        with pytest.raises(TypeError) as exc_info:
            my_metric.score(predicted="test")

        error_msg = str(exc_info.value)
        assert "Type validation errors" in error_msg
        assert "expected: Field required" in error_msg
        assert "context: Field required" in error_msg

    def test_missing_required_arguments_with_optional_arguments_error(self):
        """Test that Optional[T] parameters are treated as optional, not required."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(
            predicted: str, expected: str, context: t.Optional[str] = None
        ) -> str:
            return "pass"

        with pytest.raises(TypeError) as exc_info:
            my_metric.score(
                predicted="test"
            )  # missing 'expected' but 'context' is optional

        error_msg = str(exc_info.value)
        assert "Type validation errors" in error_msg
        assert "expected: Field required" in error_msg
        assert "context" not in error_msg  # context should not be listed as required

    def test_optional_type_annotation_without_default(self):
        """Test that t.Optional[T] without default value is still treated as optional."""

        @discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
        def my_metric(predicted: str, expected: str, context: t.Optional[str]) -> str:
            return "pass"

        # Should work without the optional parameter
        result = my_metric.score(predicted="test", expected="test")
        assert result.value == "pass"

        # Should also work with the optional parameter
        result = my_metric.score(
            predicted="test", expected="test", context="some context"
        )
        assert result.value == "pass"

        # Should also work with None for the optional parameter
        result = my_metric.score(predicted="test", expected="test", context=None)
        assert result.value == "pass"

    def test_mixed_required_optional_and_default_parameters(self):
        """Test complex scenario with required, optional, and default parameters."""

        @discrete_metric(name="complex_metric", allowed_values=["pass", "fail"])
        def my_metric(
            required1: str,
            required2: int,
            optional_typed: t.Optional[str],  # Optional type annotation
            with_default: float = 0.5,  # Has default value
            optional_with_default: t.Optional[
                str
            ] = None,  # Both optional and has default
        ) -> str:
            return "pass"

        # Test missing required arguments
        with pytest.raises(TypeError) as exc_info:
            my_metric.score(required1="test")  # missing required2

        error_msg = str(exc_info.value)
        assert "Type validation errors" in error_msg
        assert "required2: Field required" in error_msg
        assert "optional_typed" not in error_msg  # Should not be required
        assert "with_default" not in error_msg  # Should not be required
        assert "optional_with_default" not in error_msg  # Should not be required

        # Test that it works with just required arguments
        result = my_metric.score(required1="test", required2=42)
        assert result.value == "pass"

        # Test that it works with all arguments
        result = my_metric.score(
            required1="test",
            required2=42,
            optional_typed="optional",
            with_default=0.8,
            optional_with_default="also optional",
        )
        assert result.value == "pass"

    def test_unknown_arguments_warning(self):
        """Test that unknown arguments generate warnings."""

        @discrete_metric(name="simple", allowed_values=["pass", "fail"])
        def my_metric(text: str) -> str:
            return "pass"

        with pytest.warns(UserWarning, match="received unknown arguments"):
            result = my_metric.score(text="test", unknown_param="value")

        # Should still work despite unknown parameter
        assert result.value == "pass"

    def test_mixed_error_scenarios(self):
        """Test combinations of errors."""

        @discrete_metric(name="complex", allowed_values=["pass", "fail"])
        def my_metric(text: str, threshold: float = 0.5) -> str:
            return "pass"

        # Test positional + extra args
        with pytest.raises(TypeError, match="requires keyword arguments"):
            my_metric.score("text", 0.5, extra="unknown")

    def test_optional_parameters_work(self):
        """Test that optional parameters don't cause missing args error."""

        @discrete_metric(name="optional_test", allowed_values=["pass", "fail"])
        def my_metric(text: str, threshold: float = 0.5) -> str:
            return "pass" if len(text) > threshold else "fail"

        # Should work with just required parameter
        result = my_metric.score(text="hello")
        assert result.value == "pass"

        # Should also work with optional parameter
        result = my_metric.score(text="hi", threshold=5.0)
        assert result.value == "fail"

    @pytest.mark.asyncio
    async def test_async_error_handling(self):
        """Test that async methods also validate inputs."""

        @discrete_metric(name="async_metric", allowed_values=["pass", "fail"])
        async def my_metric(text: str) -> str:
            return "pass"

        # Test positional args error in async
        with pytest.raises(TypeError, match="requires keyword arguments"):
            await my_metric.ascore("test")

        # Test missing args error in async
        with pytest.raises(TypeError, match="Type validation errors"):
            await my_metric.ascore()

    def test_pydantic_validation_error_format(self):
        """Test that Pydantic validation errors are properly formatted."""

        @numeric_metric(name="complex_metric", allowed_values=(0, 10))
        def my_metric(score: int, weight: float, tags: list) -> float:
            return float(score * weight)

        with pytest.raises(TypeError) as exc_info:
            my_metric.score()  # Missing all args

        error_msg = str(exc_info.value)
        # Should show Pydantic validation errors
        assert "Type validation errors for complex_metric" in error_msg
        assert "score: Field required" in error_msg
        assert "weight: Field required" in error_msg
        assert "tags: Field required" in error_msg

    def test_no_type_hints_still_works(self):
        """Test that metrics work even without type hints."""

        @discrete_metric(name="no_hints", allowed_values=["pass", "fail"])
        def my_metric(text, threshold=0.5):  # No type hints
            return "pass"

        # Should still validate and work
        result = my_metric.score(text="hello")
        assert result.value == "pass"

        # Should still catch positional args
        with pytest.raises(TypeError, match="requires keyword arguments"):
            my_metric.score("hello", 0.8)

    def test_comprehensive_type_validation(self):
        """Test comprehensive type validation with Pydantic for all complex types."""

        @discrete_metric(name="complex_types", allowed_values=["pass", "fail"])
        def my_metric(
            simple_str: str,
            simple_int: int,
            optional_str: t.Optional[str] = None,
            list_of_strings: t.List[str] = None,
            union_type: t.Union[str, int] = "default",
        ) -> str:
            return "pass"

        # Test 1: Simple types validation
        with pytest.raises(TypeError) as exc_info:
            my_metric.score(simple_str=123, simple_int="not_int")

        error_msg = str(exc_info.value)
        assert "simple_str: Input should be a valid string" in error_msg
        assert "simple_int: Input should be a valid integer" in error_msg

        # Test 2: List type validation
        with pytest.raises(TypeError) as exc_info:
            my_metric.score(simple_str="ok", simple_int=1, list_of_strings="not_a_list")

        error_msg = str(exc_info.value)
        assert "list_of_strings: Input should be a valid list" in error_msg

        # Test 3: Union type validation - should accept both str and int
        result1 = my_metric.score(simple_str="ok", simple_int=1, union_type="string")
        result2 = my_metric.score(simple_str="ok", simple_int=1, union_type=42)
        assert result1.value == "pass"
        assert result2.value == "pass"

        # Test 4: Union type validation - should reject other types
        with pytest.raises(TypeError) as exc_info:
            my_metric.score(simple_str="ok", simple_int=1, union_type=[1, 2, 3])

        error_msg = str(exc_info.value)
        assert "union_type:" in error_msg  # Should show union validation error

        # Test 5: Optional types work correctly
        result = my_metric.score(
            simple_str="ok", simple_int=1
        )  # optional_str not provided
        assert result.value == "pass"


class TestCustomTypeValidation:
    """Tests for validation with custom types like InstructorLLM."""

    def test_custom_type_validation_should_work(self):
        """Test that metrics can accept custom class types without warnings."""

        # Create a mock custom class similar to InstructorLLM
        class MockInstructorLLM:
            def __init__(self, name="mock"):
                self.name = name

            def generate(self, prompt: str, response_model) -> str:
                return "pass"

        @discrete_metric(name="custom_type_metric", allowed_values=["pass", "fail"])
        def my_metric(input_text: str, llm: MockInstructorLLM) -> str:
            return llm.generate(f"Process: {input_text}", str)

        # This should work without warnings or errors
        mock_llm = MockInstructorLLM()

        # Capture warnings to ensure no validation warnings
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = my_metric.score(input_text="test", llm=mock_llm)

            # Should not have any warnings about "Could not create validation model"
            validation_warnings = [
                warning
                for warning in w
                if "Could not create validation model" in str(warning.message)
            ]
            assert len(validation_warnings) == 0, (
                f"Got validation warnings: {[str(w.message) for w in validation_warnings]}"
            )

        assert isinstance(result, MetricResult)
        assert result.value == "pass"

    def test_custom_type_validation_wrong_type_should_fail(self):
        """Test that wrong custom types are still caught."""

        class MockInstructorLLM:
            def generate(self, prompt: str, response_model) -> str:
                return "pass"

        class WrongType:
            pass

        @discrete_metric(name="custom_type_metric", allowed_values=["pass", "fail"])
        def my_metric(input_text: str, llm: MockInstructorLLM) -> str:
            return llm.generate(f"Process: {input_text}", str)

        wrong_obj = WrongType()

        # Should fail with type validation error
        with pytest.raises(TypeError) as exc_info:
            my_metric.score(input_text="test", llm=wrong_obj)

        error_msg = str(exc_info.value)
        assert "llm:" in error_msg  # Should show validation error for llm field

    def test_mixed_standard_and_custom_types(self):
        """Test validation with both standard Python types and custom types."""

        class MockLLM:
            def process(self, text: str) -> str:
                return "processed"

        @discrete_metric(name="mixed_type_metric", allowed_values=["pass", "fail"])
        def my_metric(
            text: str, count: int, llm: MockLLM, optional_flag: bool = False
        ) -> str:
            result = llm.process(text)
            return "pass" if count > 0 and result else "fail"

        mock_llm = MockLLM()

        # Should work with valid types
        result = my_metric.score(
            text="hello", count=5, llm=mock_llm, optional_flag=True
        )
        assert result.value == "pass"

        # Should fail with wrong standard type
        with pytest.raises(TypeError):
            my_metric.score(
                text="hello", count="not_int", llm=mock_llm
            )  # count should be int

        # Should fail with wrong custom type
        with pytest.raises(TypeError):
            my_metric.score(
                text="hello", count=5, llm="not_llm"
            )  # llm should be MockLLM

    def test_instructor_llm_like_usage(self):
        """Test the actual use case that was failing - InstructorLLM-like usage."""

        # Mock the InstructorLLM interface
        class MockInstructorLLM:
            def generate(self, prompt: str, response_model):
                if "accurate" in prompt:
                    return "pass"
                return "fail"

        @discrete_metric(name="summary_accuracy", allowed_values=["pass", "fail"])
        def summary_accuracy(
            user_input: str, response: str, llm: MockInstructorLLM
        ) -> str:
            prompt = f"Is the following summary accurate for the user's query: {user_input}? {response}"
            return llm.generate(prompt, response_model=str)

        # Test data similar to the failing case
        test_data = {
            "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024...",
            "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing...",
        }

        mock_llm = MockInstructorLLM()

        # This should work without warnings
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = summary_accuracy.score(
                user_input=test_data["user_input"],
                response=test_data["response"],
                llm=mock_llm,
            )

            # Should not have validation model warnings
            validation_warnings = [
                warning
                for warning in w
                if "Could not create validation model" in str(warning.message)
            ]
            assert len(validation_warnings) == 0

        assert isinstance(result, MetricResult)
        assert result.value in ["pass", "fail"]


class TestIDESupport:
    """Tests for IDE type support and Protocol compliance."""

    def test_discrete_metric_has_proper_methods(self):
        """Test that discrete metrics have all expected methods for IDE support."""

        @discrete_metric(name="ide_test", allowed_values=["pass", "fail"])
        def my_metric(text: str) -> str:
            return "pass"

        # Should have all protocol methods
        assert hasattr(my_metric, "score")
        assert hasattr(my_metric, "ascore")
        assert hasattr(my_metric, "batch_score")
        assert hasattr(my_metric, "abatch_score")
        assert hasattr(my_metric, "__call__")
        assert hasattr(my_metric, "name")
        assert hasattr(my_metric, "allowed_values")

        # Test that methods work
        result = my_metric.score(text="test")
        assert isinstance(result, MetricResult)
        assert result.value == "pass"

    def test_numeric_metric_has_proper_methods(self):
        """Test that numeric metrics have all expected methods for IDE support."""

        @numeric_metric(name="ide_numeric_test", allowed_values=(0.0, 1.0))
        def my_metric(value: float) -> float:
            return min(max(value, 0.0), 1.0)

        # Should have all protocol methods
        assert hasattr(my_metric, "score")
        assert hasattr(my_metric, "ascore")
        assert hasattr(my_metric, "batch_score")
        assert hasattr(my_metric, "abatch_score")
        assert hasattr(my_metric, "__call__")
        assert hasattr(my_metric, "name")
        assert hasattr(my_metric, "allowed_values")

        # Test that methods work
        result = my_metric.score(value=0.5)
        assert isinstance(result, MetricResult)
        assert result.value == 0.5

    def test_ranking_metric_has_proper_methods(self):
        """Test that ranking metrics have all expected methods for IDE support."""

        @ranking_metric(name="ide_ranking_test", allowed_values=2)
        def my_metric(items: list) -> list:
            return [1, 0]  # Simple reverse ranking

        # Should have all protocol methods
        assert hasattr(my_metric, "score")
        assert hasattr(my_metric, "ascore")
        assert hasattr(my_metric, "batch_score")
        assert hasattr(my_metric, "abatch_score")
        assert hasattr(my_metric, "__call__")
        assert hasattr(my_metric, "name")
        assert hasattr(my_metric, "allowed_values")

        # Test that methods work
        result = my_metric.score(items=["a", "b"])
        assert isinstance(result, MetricResult)
        assert result.value == [1, 0]

    def test_protocol_attributes_accessible(self):
        """Test that protocol attributes are properly accessible."""

        @discrete_metric(name="protocol_test", allowed_values=["yes", "no"])
        def test_metric(input_val: str) -> str:
            return "yes" if input_val else "no"

        # Protocol attributes should be accessible
        assert test_metric.name == "protocol_test"
        assert test_metric.allowed_values == ["yes", "no"]

        # Should work in both direct call and score method
        direct_result = test_metric("hello")
        assert direct_result == "yes"

        score_result = test_metric.score(input_val="hello")
        assert isinstance(score_result, MetricResult)
        assert score_result.value == "yes"


class TestDecoratorValidationFallback:
    """Tests for fallback behavior when dynamic Pydantic model creation fails."""

    @pytest.mark.asyncio
    async def test_kwargs_preserved_when_pydantic_model_creation_fails(
        self, monkeypatch
    ):
        """Test that kwargs are passed to the async function when model creation fails."""

        def failing_create_model(*args, **kwargs):
            raise ValueError("Simulated dynamic model creation failure")

        import ragas.metrics.decorator as dec_mod

        monkeypatch.setattr(dec_mod, "create_model", failing_create_model)
        monkeypatch.setattr(pydantic, "create_model", failing_create_model)

        @discrete_metric(name="test_fallback_async", allowed_values=["pass", "fail"])
        async def my_metric(question: str) -> str:
            return "pass" if question == "expected" else "fail"

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            result = await my_metric.ascore(question="expected")

        assert isinstance(result, MetricResult)
        assert result.value == "pass"
        assert result.reason is None

        # Verify warning was emitted explaining the bypass
        user_warnings = [
            w for w in recorded_warnings if issubclass(w.category, UserWarning)
        ]
        assert any(
            "Could not create validation model" in str(w.message)
            and "without Pydantic validation" in str(w.message)
            for w in user_warnings
        )

    def test_sync_score_path_also_preserves_kwargs(self, monkeypatch):
        """Test that kwargs are preserved on synchronous score() when model creation fails."""

        def failing_create_model(*args, **kwargs):
            raise ValueError("Simulated dynamic model creation failure")

        import ragas.metrics.decorator as dec_mod

        monkeypatch.setattr(dec_mod, "create_model", failing_create_model)
        monkeypatch.setattr(pydantic, "create_model", failing_create_model)

        @discrete_metric(name="test_fallback_sync", allowed_values=["pass", "fail"])
        def my_metric(question: str) -> str:
            return "pass" if question == "expected" else "fail"

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            result = my_metric.score(question="expected")

        assert isinstance(result, MetricResult)
        assert result.value == "pass"
        assert result.reason is None

        user_warnings = [
            w for w in recorded_warnings if issubclass(w.category, UserWarning)
        ]
        assert any(
            "Could not create validation model" in str(w.message)
            and "without Pydantic validation" in str(w.message)
            for w in user_warnings
        )

    def test_validation_model_creation_success_does_not_warn(self):
        """Sanity check: when model creation succeeds, no fallback warning is emitted."""

        @discrete_metric(name="test_normal", allowed_values=["pass", "fail"])
        def my_metric(question: str) -> str:
            return "pass" if question == "expected" else "fail"

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            result = my_metric.score(question="expected")

        assert isinstance(result, MetricResult)
        assert result.value == "pass"

        fallback_warnings = [
            w
            for w in recorded_warnings
            if "Could not create validation model" in str(w.message)
        ]
        assert len(fallback_warnings) == 0
