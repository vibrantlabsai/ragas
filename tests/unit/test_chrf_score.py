from unittest.mock import patch

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ChrfScore
from ragas.metrics.base import MetricType


@pytest.fixture
def mock_sacrebleu():
    """Mock sacrebleu corpus_chrf function."""
    pytest.importorskip("sacrebleu")
    with patch("sacrebleu.corpus_chrf") as mock:
        yield mock


def test_chrf_score_init_sacrebleu_import():
    """Test ChrfScore initialization with sacrebleu import."""
    pytest.importorskip("sacrebleu")
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
async def test_chrf_score_single_turn_ascore_none_values(mock_sacrebleu):
    """Test single turn async score with None values."""
    metric = ChrfScore()

    sample = SingleTurnSample(reference=None, response="Hello there")
    score = await metric._single_turn_ascore(sample, None)
    assert score == 0.0

    sample = SingleTurnSample(reference="Hello world", response=None)
    score = await metric._single_turn_ascore(sample, None)
    assert score == 0.0


@pytest.mark.asyncio
async def test_chrf_score_ascore(mock_sacrebleu):
    """Test async score calculation from dictionary row."""
    metric = ChrfScore()

    mock_sacrebleu.return_value.score = 75.0

    row = {"reference": "Hello world", "response": "Hello there"}
    score = await metric._ascore(row, None)
    
    assert isinstance(score, float)
    assert score == 0.75
    mock_sacrebleu.assert_called_once_with(
        ["Hello there"], [["Hello world"]], **metric.kwargs
    )
