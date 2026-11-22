import typing as t
from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AspectCritic, MetricResult, SimpleCriteriaScore
from ragas.metrics.base import MetricType, SimpleLLMMetric as LLMMetric


def test_single_turn_metric():
    from ragas.metrics.base import SingleTurnMetric

    class FakeMetric(SingleTurnMetric):
        name = "fake_metric"  # type: ignore
        _required_columns = {MetricType.SINGLE_TURN: {"user_input", "response"}}

        def init(self, run_config):
            pass

        async def _ascore(self, row, callbacks) -> float:
            return 0

        async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks):
            return 0

    fm = FakeMetric()
    assert fm.single_turn_score(SingleTurnSample(user_input="a", response="b")) == 0


def test_required_columns():
    from ragas.metrics.base import MetricType, SingleTurnMetric

    @dataclass
    class FakeMetric(SingleTurnMetric):
        name = "fake_metric"  # type: ignore
        _required_columns: t.Dict[MetricType, t.Set[str]] = field(
            default_factory=lambda: {
                MetricType.SINGLE_TURN: {
                    "user_input",
                    "response",
                    "retrieved_contexts:optional",
                },
            }
        )

        def init(self, run_config):
            pass

        async def _ascore(self, row, callbacks) -> float:
            return 0

        async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks):
            return 0

    fm = FakeMetric()

    # only return required columns, don't include optional columns
    assert fm.required_columns[MetricType.SINGLE_TURN.name] == {
        "user_input",
        "response",
    }

    # check if optional columns are included
    assert fm.get_required_columns(with_optional=False)[
        MetricType.SINGLE_TURN.name
    ] == {
        "user_input",
        "response",
    }
    # check if optional columns are included
    assert fm.get_required_columns(with_optional=True)[MetricType.SINGLE_TURN.name] == {
        "user_input",
        "response",
        "retrieved_contexts",
    }

    # check if only required columns are returned
    assert (
        fm._only_required_columns_single_turn(
            SingleTurnSample(user_input="a", response="b", reference="c")
        ).to_dict()
        == SingleTurnSample(user_input="a", response="b").to_dict()
    )

    # check if optional columns are included if they are not none
    assert (
        fm._only_required_columns_single_turn(
            SingleTurnSample(user_input="a", response="b", retrieved_contexts=["c"])
        ).to_dict()
        == SingleTurnSample(
            user_input="a", response="b", retrieved_contexts=["c"]
        ).to_dict()
    )


@pytest.mark.parametrize("metric", [AspectCritic, SimpleCriteriaScore])
def test_metrics_with_definition(metric):
    """
    Test the general metrics like AspectCritic, SimpleCriteriaScore
    """

    m = metric(name="metric", definition="test")

    # check if the definition is set
    assert m.definition == "test"

    # check if the definition is updated and the instruction along with it
    m.definition = "this is a new definition"
    assert m.definition == "this is a new definition"
    assert "this is a new definition" in m.single_turn_prompt.instruction


@pytest.mark.parametrize("metric", [AspectCritic, SimpleCriteriaScore])
def test_metric_init_propagates_definition_to_prompts(metric):
    """Regression test for #1944: definition passed to __init__ must reach
    both single_turn_prompt.instruction and multi_turn_prompt.instruction,
    not just the private ``_definition`` attribute."""

    m = metric(name="metric", definition="Score 0 to 5 by similarity")

    assert "Score 0 to 5 by similarity" in m.single_turn_prompt.instruction
    assert "Score 0 to 5 by similarity" in m.multi_turn_prompt.instruction


@pytest.mark.parametrize("metric", [AspectCritic, SimpleCriteriaScore])
def test_metric_setter_updates_multi_turn_prompt(metric):
    """The ``definition`` setter must update *both* prompt instructions."""

    m = metric(name="metric", definition="original")
    m.definition = "updated definition"

    assert "updated definition" in m.single_turn_prompt.instruction
    assert "updated definition" in m.multi_turn_prompt.instruction
    assert "original" not in m.single_turn_prompt.instruction
    assert "original" not in m.multi_turn_prompt.instruction


def test_ignored_columns():
    """Test that :ignored suffixed columns are properly excluded from all column queries."""
    from ragas.metrics.base import MetricType, SingleTurnMetric

    @dataclass
    class TestMetricWithIgnored(SingleTurnMetric):
        name = "test_metric_with_ignored"  # type: ignore
        _required_columns: t.Dict[MetricType, t.Set[str]] = field(
            default_factory=lambda: {
                MetricType.SINGLE_TURN: {
                    "user_input",  # Required
                    "response",  # Required
                    "retrieved_contexts:optional",  # Optional - should be included when with_optional=True
                    "reference:ignored",  # Ignored
                    "rubric:ignored",  # Ignored
                },
            }
        )

        def init(self, run_config):
            pass

        async def _ascore(self, row, callbacks) -> float:
            return 0.5

        async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks):
            return 0.5

    metric = TestMetricWithIgnored()

    # Test required_columns property (should exclude both :optional and :ignored)
    required_cols = metric.required_columns[MetricType.SINGLE_TURN.name]
    expected_required = {"user_input", "response"}
    assert required_cols == expected_required, (
        f"Expected {expected_required}, got {required_cols}"
    )

    # Test get_required_columns(with_optional=False) - should exclude both :optional and :ignored
    required_cols_no_optional = metric.get_required_columns(with_optional=False)[
        MetricType.SINGLE_TURN.name
    ]
    assert required_cols_no_optional == expected_required, (
        f"Expected {expected_required}, got {required_cols_no_optional}"
    )

    # Test get_required_columns(with_optional=True) - should include :optional but exclude :ignored
    required_cols_with_optional = metric.get_required_columns(with_optional=True)[
        MetricType.SINGLE_TURN.name
    ]
    expected_with_optional = {"user_input", "response", "retrieved_contexts"}
    assert required_cols_with_optional == expected_with_optional, (
        f"Expected {expected_with_optional}, got {required_cols_with_optional}"
    )

    # Verify that ignored fields are never included anywhere
    all_results = [
        required_cols,
        required_cols_no_optional,
        required_cols_with_optional,
    ]
    for result in all_results:
        assert "reference" not in result, (
            f"Ignored field 'reference' found in result: {result}"
        )
        assert "rubric" not in result, (
            f"Ignored field 'rubric' found in result: {result}"
        )
        assert "reference:ignored" not in result, (
            f"Raw ignored field 'reference:ignored' found in result: {result}"
        )
        assert "rubric:ignored" not in result, (
            f"Raw ignored field 'rubric:ignored' found in result: {result}"
        )


def test_ignored_columns_validation():
    """Test that validation works correctly with :ignored suffixed columns."""
    from ragas.metrics.base import MetricType, SingleTurnMetric

    class TestMetric(SingleTurnMetric):
        name = "test_metric"  # type: ignore

        def init(self, run_config):
            pass

        async def _ascore(self, row, callbacks) -> float:
            return 0.5

        async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks):
            return 0.5

    metric = TestMetric()

    # Test that validation passes for valid columns with :ignored suffix
    valid_columns_with_ignored = {
        MetricType.SINGLE_TURN: {
            "user_input",
            "response",
            "reference:ignored",  # Valid base column with :ignored
            "retrieved_contexts:ignored",  # Valid base column with :ignored
        }
    }
    # This should not raise an error
    metric.required_columns = valid_columns_with_ignored

    # Test that validation fails for invalid base columns with :ignored suffix
    with pytest.raises(ValueError, match="Invalid column.*must be one of"):
        invalid_columns_with_ignored = {
            MetricType.SINGLE_TURN: {
                "user_input",
                "invalid_column:ignored",  # Invalid base column
            }
        }
        metric.required_columns = invalid_columns_with_ignored

    # Test mixed valid and invalid columns
    with pytest.raises(ValueError, match="Invalid column.*must be one of"):
        mixed_columns = {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response:optional",  # Valid
                "reference:ignored",  # Valid
                "bad_column:ignored",  # Invalid base column
            }
        }
        metric.required_columns = mixed_columns


# ====================
# Metric Base Tests (formerly test_metric_base.py)
# ====================


class MetricResponseModel(BaseModel):
    value: int
    reason: t.Optional[str] = None


@dataclass
class CustomMetric(LLMMetric):
    """Custom metric implementation for testing."""

    def __post_init__(self):
        super().__post_init__()
        self._response_model = MetricResponseModel

    def get_correlation(
        self, gold_labels: t.List[str], predictions: t.List[str]
    ) -> float:
        return 0.0  # Placeholder for correlation logic


@pytest.fixture
def mock_llm(mock_llm):
    """Use the mock LLM from conftest."""
    return mock_llm


def test_metric_creation():
    """Test creating a custom metric."""
    metric = CustomMetric(name="test_metric", prompt="What is the result of {input}?")

    assert metric.name == "test_metric"
    assert isinstance(metric.prompt, str) or hasattr(metric.prompt, "format")


def test_metric_get_variables():
    """Test extracting variables from prompt template."""
    metric = CustomMetric(
        name="test_metric",
        prompt="Evaluate the {question} given the {context} and {answer}",
    )

    variables = metric.get_variables()
    expected_vars = ["question", "context", "answer"]

    assert set(variables) == set(expected_vars)


def test_metric_score_single(mock_llm):
    """Test scoring with a single input."""
    metric = CustomMetric(name="test_metric", prompt="What is the result of {input}?")

    # Mock the LLM to return a valid response
    def mock_generate(prompt, response_model):
        return response_model(value=1, reason="test reason")

    mock_llm.generate = mock_generate

    result = metric.score(llm=mock_llm, input="test")

    assert isinstance(result, MetricResult)
    assert result.traces is not None
    assert "input" in result.traces


@pytest.mark.asyncio
async def test_metric_async_score(mock_llm):
    """Test async scoring functionality."""
    metric = CustomMetric(name="test_metric", prompt="What is the result of {input}?")

    # Mock the async LLM method
    async def mock_agenerate(prompt, response_model):
        return response_model(value=1, reason="test reason")

    mock_llm.agenerate = mock_agenerate

    result = await metric.ascore(llm=mock_llm, input="test")

    assert isinstance(result, MetricResult)
    assert result.traces is not None


def test_metric_response_model():
    """Test that metric has correct response model."""
    metric = CustomMetric(name="test_metric", prompt="What is the result of {input}?")

    assert metric._response_model == MetricResponseModel


def test_metric_prompt_conversion():
    """Test that string prompts are converted to Prompt objects."""
    metric = CustomMetric(name="test_metric", prompt="What is the result of {input}?")

    # After __post_init__, prompt should be converted to Prompt object
    assert hasattr(metric.prompt, "format")
