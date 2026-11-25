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


def _llm_returning(mock_llm, payload):
    """Point mock_llm's generate/agenerate at a fixed judge reply."""

    def mock_generate(prompt, response_model):
        return response_model(**payload)

    async def mock_agenerate(prompt, response_model):
        return response_model(**payload)

    mock_llm.generate = mock_generate
    mock_llm.agenerate = mock_agenerate
    return mock_llm


@pytest.mark.parametrize(
    "allowed_values, value",
    [((0.0, 10.0), 11.0), ((0.0, 10.0), -1.0), ((1.0, 5.0), 0.0), (range(1, 6), 7)],
)
def test_numeric_metric_score_rejects_out_of_range_value(
    mock_llm, allowed_values, value
):
    """Class-based NumericMetric validates like the @numeric_metric path."""
    from ragas.metrics import NumericMetric

    metric = NumericMetric(
        name="m", prompt="Rate {response}", allowed_values=allowed_values
    )
    llm = _llm_returning(mock_llm, {"value": value, "reason": "judge"})

    result = metric.score(llm=llm, response="x")

    assert result.value is None
    assert "expected value in range" in result.reason
    assert result.traces["output"]["value"] == value


@pytest.mark.asyncio
async def test_numeric_metric_ascore_rejects_out_of_range_value(mock_llm):
    from ragas.metrics import NumericMetric

    metric = NumericMetric(name="m", prompt="Rate {response}", allowed_values=(0, 10))
    llm = _llm_returning(mock_llm, {"value": 11.0, "reason": "judge"})

    result = await metric.ascore(llm=llm, response="x")

    assert result.value is None
    assert "expected value in range" in result.reason


@pytest.mark.parametrize("value", [0.0, 7.5, 10.0])
def test_numeric_metric_score_keeps_in_range_value(mock_llm, value):
    from ragas.metrics import NumericMetric

    metric = NumericMetric(name="m", prompt="Rate {response}", allowed_values=(0, 10))
    llm = _llm_returning(mock_llm, {"value": value, "reason": "judge"})

    result = metric.score(llm=llm, response="x")

    assert result.value == value
    assert result.reason == "judge"


def test_ranking_metric_score_rejects_wrong_length(mock_llm):
    from ragas.metrics import RankingMetric

    metric = RankingMetric(name="r", prompt="Rank {items}", allowed_values=3)
    llm = _llm_returning(mock_llm, {"value": ["a", "b"], "reason": "judge"})

    result = metric.score(llm=llm, items="a, b, c")

    assert result.value is None
    assert "expected 3 items" in result.reason


class _FakeGoldDataset:
    """Minimal stand-in for the Dataset API used by validate_alignment."""

    def __init__(self, rows):
        self.rows = rows

    def reload(self):
        pass

    def __iter__(self):
        return iter(self.rows)

    def get_row_value(self, row, key):
        return row[key]

    def to_pandas(self):
        import pandas as pd

        return pd.DataFrame(self.rows)


def test_numeric_metric_alignment_skips_rejected_scores_in_pairs(mock_llm):
    """A rejected score drops that row from both gold and predicted lists."""
    pytest.importorskip("scipy")
    from ragas.metrics import NumericMetric

    metric = NumericMetric(name="m", prompt="Rate {response}", allowed_values=(0, 10))
    replies = iter([1.0, 11.0, 3.0])

    def mock_generate(prompt, response_model):
        return response_model(value=next(replies), reason="judge")

    mock_llm.generate = mock_generate
    dataset = _FakeGoldDataset(
        [{"response": r, "m": g} for r, g in zip("abc", [1.0, 2.0, 3.0])]
    )

    result = metric.validate_alignment(llm=mock_llm, test_dataset=dataset)

    assert result["agreement_rate"] == 1.0
    assert result["df"]["m_pred"].isna().tolist() == [False, True, False]


def test_numeric_metric_alignment_needs_two_valid_pairs(mock_llm):
    """Too few valid pairs gives a clear error instead of a scipy one."""
    from ragas.metrics import NumericMetric

    metric = NumericMetric(name="m", prompt="Rate {response}", allowed_values=(0, 10))
    replies = iter([1.0, 11.0])

    def mock_generate(prompt, response_model):
        return response_model(value=next(replies), reason="judge")

    mock_llm.generate = mock_generate
    dataset = _FakeGoldDataset(
        [{"response": "a", "m": 1.0}, {"response": "b", "m": 2.0}]
    )

    with pytest.raises(ValueError, match="at least 2 rows"):
        metric.validate_alignment(llm=mock_llm, test_dataset=dataset)
