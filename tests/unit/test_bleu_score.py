from unittest.mock import patch

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore
from ragas.metrics.base import MetricType


@pytest.fixture
def mock_sacrebleu():
    """Mock sacrebleu corpus_bleu function."""
    with patch("sacrebleu.corpus_bleu") as mock:
        yield mock


def test_bleu_score_init_sacrebleu_import():
    """Test BleuScore initialization with sacrebleu import."""
    metric = BleuScore()
    assert hasattr(metric, "corpus_bleu")
    assert metric.name == "bleu_score"
    assert metric._required_columns == {
        MetricType.SINGLE_TURN: {"reference", "response"}
    }


def test_bleu_score_init_sacrebleu_import_error():
    """Test BleuScore initialization raises ImportError if sacrebleu is missing."""
    with patch("builtins.__import__", side_effect=ImportError):
        with pytest.raises(ImportError, match="sacrebleu is required"):
            BleuScore()


@pytest.mark.asyncio
async def test_bleu_score_single_turn_ascore(mock_sacrebleu):
    """Test single turn async score passes the whole text as a single segment.

    Regression test for #2888: the reference must be a single stream of one
    segment (``[[reference]]``) rather than N streams of one segment each, so
    that sacrebleu evaluates the full text instead of only the first sentence.
    """
    metric = BleuScore()

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
async def test_bleu_score_ascore(mock_sacrebleu):
    """Test async score calculation from dictionary row."""
    metric = BleuScore()

    mock_sacrebleu.return_value.score = 75.0

    row = {"reference": "Hello world", "response": "Hello there"}
    score = await metric._ascore(row, None)

    assert isinstance(score, float)
    assert score == 0.75
    mock_sacrebleu.assert_called_once_with(
        ["Hello there"], [["Hello world"]], **metric.kwargs
    )


@pytest.mark.asyncio
async def test_bleu_score_evaluates_all_sentences():
    """Regression test for #2888: multi-sentence text must be fully evaluated.

    With the old sentence-splitting logic, sacrebleu's internal
    ``zip(hypotheses, *references)`` truncated to the first sentence, so a
    response whose first sentence matched the reference scored a perfect 1.0
    regardless of the remaining sentences. The whole text must be scored.
    """
    metric = BleuScore()

    reference = "The cat sat on the mat. It was a sunny day."
    # First sentence matches exactly; the second differs entirely.
    response = "The cat sat on the mat. Dogs barked."

    sample = SingleTurnSample(reference=reference, response=response)
    score = await metric._single_turn_ascore(sample, None)

    # The buggy implementation returned 1.0 (only the first sentence compared).
    assert score < 1.0
