from unittest.mock import MagicMock, patch

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore
from ragas.metrics.base import MetricType


def test_bleu_score_init():
    """Test BleuScore initialization with sacrebleu import."""
    metric = BleuScore()
    assert hasattr(metric, "corpus_bleu")
    assert metric.name == "bleu_score"
    assert metric._required_columns == {
        MetricType.SINGLE_TURN: {"reference", "response"}
    }


def test_bleu_score_init_import_error():
    """Test BleuScore initialization raises ImportError if sacrebleu is missing."""
    with patch("builtins.__import__", side_effect=ImportError):
        with pytest.raises(ImportError, match="sacrebleu is required"):
            BleuScore()


@pytest.mark.asyncio
async def test_bleu_score_single_turn_ascore():
    """Test single turn async score calculation."""
    metric = BleuScore()

    mock_result = MagicMock()
    mock_result.score = 85.0

    with patch.object(metric, "corpus_bleu", return_value=mock_result) as mock_bleu:
        sample = SingleTurnSample(
            reference="The cat sat on the mat.",
            response="The cat sat on the mat.",
        )
        score = await metric._single_turn_ascore(sample, None)

        assert isinstance(score, float)
        assert score == 0.85
        mock_bleu.assert_called_once()


@pytest.mark.asyncio
async def test_bleu_score_ascore():
    """Test async score calculation from dictionary row."""
    metric = BleuScore()

    mock_result = MagicMock()
    mock_result.score = 60.0

    with patch.object(metric, "corpus_bleu", return_value=mock_result):
        row = {"reference": "Hello world.", "response": "Hello there."}
        score = await metric._ascore(row, None)

        assert isinstance(score, float)
        assert score == 0.60


@pytest.mark.asyncio
async def test_bleu_score_assertion_error_on_none_reference():
    """Test that None reference raises AssertionError."""
    metric = BleuScore()
    sample = SingleTurnSample(reference=None, response="Hello there")
    with pytest.raises(AssertionError, match="expects a valid reference string"):
        await metric._single_turn_ascore(sample, None)


@pytest.mark.asyncio
async def test_bleu_score_assertion_error_on_none_response():
    """Test that None response raises AssertionError."""
    metric = BleuScore()
    sample = SingleTurnSample(reference="Hello world", response=None)
    with pytest.raises(AssertionError, match="expects a valid response string"):
        await metric._single_turn_ascore(sample, None)
