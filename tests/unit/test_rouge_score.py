from unittest.mock import MagicMock, patch

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._rouge_score import RougeScore
from ragas.metrics.base import MetricType


def test_rouge_score_init():
    """Test RougeScore initialization with rouge_score import."""
    metric = RougeScore()
    assert hasattr(metric, "rouge_scorer")
    assert metric.name == "rouge_score"
    assert metric.rouge_type == "rougeL"
    assert metric.mode == "fmeasure"
    assert metric._required_columns == {
        MetricType.SINGLE_TURN: {"reference", "response"}
    }


def test_rouge_score_init_custom_params():
    """Test RougeScore initialization with custom parameters."""
    metric = RougeScore(rouge_type="rouge1", mode="precision")
    assert metric.rouge_type == "rouge1"
    assert metric.mode == "precision"


def test_rouge_score_init_import_error():
    """Test RougeScore initialization raises ImportError if rouge_score is missing."""
    with patch("builtins.__import__", side_effect=ImportError("rouge_score")):
        with pytest.raises(ImportError, match="is required for rouge score"):
            RougeScore()


@pytest.mark.asyncio
async def test_rouge_score_single_turn_ascore():
    """Test single turn async score calculation."""
    metric = RougeScore()

    mock_score = MagicMock()
    mock_score.fmeasure = 0.85

    mock_scorer_instance = MagicMock()
    mock_scorer_instance.score.return_value = {"rougeL": mock_score}

    with patch.object(metric, "rouge_scorer") as mock_module:
        mock_module.RougeScorer.return_value = mock_scorer_instance
        sample = SingleTurnSample(
            reference="The cat sat on the mat.",
            response="The cat sat on the mat.",
        )
        score = await metric._single_turn_ascore(sample, None)

        assert isinstance(score, float)
        assert score == 0.85
        mock_module.RougeScorer.assert_called_once_with(["rougeL"], use_stemmer=True)


@pytest.mark.asyncio
async def test_rouge_score_rouge1_precision():
    """Test RougeScore with rouge1 type and precision mode."""
    metric = RougeScore(rouge_type="rouge1", mode="precision")

    mock_score = MagicMock()
    mock_score.precision = 0.90

    mock_scorer_instance = MagicMock()
    mock_scorer_instance.score.return_value = {"rouge1": mock_score}

    with patch.object(metric, "rouge_scorer") as mock_module:
        mock_module.RougeScorer.return_value = mock_scorer_instance
        sample = SingleTurnSample(
            reference="Hello world.",
            response="Hello there.",
        )
        score = await metric._single_turn_ascore(sample, None)

        assert isinstance(score, float)
        assert score == 0.90


@pytest.mark.asyncio
async def test_rouge_score_ascore():
    """Test async score calculation from dictionary row."""
    metric = RougeScore()

    mock_score = MagicMock()
    mock_score.fmeasure = 0.75

    mock_scorer_instance = MagicMock()
    mock_scorer_instance.score.return_value = {"rougeL": mock_score}

    with patch.object(metric, "rouge_scorer") as mock_module:
        mock_module.RougeScorer.return_value = mock_scorer_instance
        row = {"reference": "Hello world.", "response": "Hello there."}
        score = await metric._ascore(row, None)

        assert isinstance(score, float)
        assert score == 0.75


@pytest.mark.asyncio
async def test_rouge_score_assertion_error_on_none_reference():
    """Test that None reference raises AssertionError."""
    metric = RougeScore()
    sample = SingleTurnSample(reference=None, response="Hello there")
    with pytest.raises(AssertionError, match="reference must be a string"):
        await metric._single_turn_ascore(sample, None)


@pytest.mark.asyncio
async def test_rouge_score_assertion_error_on_none_response():
    """Test that None response raises AssertionError."""
    metric = RougeScore()
    sample = SingleTurnSample(reference="Hello world", response=None)
    with pytest.raises(AssertionError, match="response must be a string"):
        await metric._single_turn_ascore(sample, None)
