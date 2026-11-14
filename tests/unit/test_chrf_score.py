from unittest.mock import patch

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ChrfScore
from ragas.metrics.base import MetricType


@pytest.fixture
def mock_sacrebleu():
    """Mock sacrebleu corpus_chrf function."""
    with patch("sacrebleu.corpus_chrf") as mock:
        yield mock


def test_chrf_score_init_sacrebleu_import():
    """Test ChrfScore initialization with sacrebleu import."""
    metric = ChrfScore()
    assert hasattr(metric, "corpus_chrf")
    assert metric.name == "chrf_score"
    assert metric._required_columns == {
        MetricType.SINGLE_TURN: {"reference", "response"}
    }


def test_chrf_score_init_sacrebleu_import_error():
    """Test ChrfScore initialization raises ImportError if sacrebleu is missing."""
    with patch("builtins.__import__", side_effect=ImportError):
        with pytest.raises(ImportError, match="sacrebleu is required"):
            ChrfScore()


@pytest.mark.asyncio
async def test_chrf_score_single_turn_ascore(mock_sacrebleu):
    """Test single turn async score calculation."""
    metric = ChrfScore()

    mock_sacrebleu.return_value.score = 80

    sample = SingleTurnSample(
        reference="The Eiffel Tower is located in Paris.",
        response="The Eiffel Tower is located in India.",
    )
    score = await metric._single_turn_ascore(sample, None)

    assert isinstance(score, float)
    assert score == 0.80
    mock_sacrebleu.assert_called_once_with(
        ["The Eiffel Tower is located in India."],
        [["The Eiffel Tower is located in Paris."]],
        **metric.kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", [None, "", "   "])
async def test_chrf_score_rejects_missing_reference(mock_sacrebleu, reference):
    """Missing or empty ground truth must not become a zero score."""
    metric = ChrfScore()

    sample = SingleTurnSample(
        reference=reference,
        response="Hello there",
    )

    with pytest.raises(ValueError, match="non-empty reference string"):
        await metric._single_turn_ascore(sample, None)

    mock_sacrebleu.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [None, "", "   "])
async def test_chrf_score_returns_zero_for_empty_response(mock_sacrebleu, response):
    """An empty model response may still receive a zero score."""
    metric = ChrfScore()

    sample = SingleTurnSample(
        reference="Hello world",
        response=response,
    )

    score = await metric._single_turn_ascore(sample, None)

    assert score == 0.0
    mock_sacrebleu.assert_not_called()


@pytest.mark.asyncio
async def test_chrf_score_ascore(mock_sacrebleu):
    """Test async score calculation from dictionary row."""
    metric = ChrfScore()

    # Mock corpus_chrf to return a score object
    mock_sacrebleu.return_value.score = 75.0

    row = {"reference": "Hello world", "response": "Hello there"}
    score = await metric._ascore(row, None)

    assert isinstance(score, float)
    assert score == 0.75
    mock_sacrebleu.assert_called_once_with(
        ["Hello there"], [["Hello world"]], **metric.kwargs
    )
