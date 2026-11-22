from unittest.mock import patch

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore
from ragas.metrics.base import MetricType
from ragas.metrics.collections import BleuScore as CollectionsBleuScore


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
    """Test single turn async score calculation."""
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
async def test_bleu_score_scores_full_text_not_first_sentence(mock_sacrebleu):
    """Regression test: multi-sentence texts must be scored as one segment.

    Previously the reference was split into one reference stream per
    sentence, so sacrebleu's zip() truncated the corpus to the first
    sentence and the rest of the text was silently ignored.
    """
    metric = BleuScore()

    mock_sacrebleu.return_value.score = 40

    sample = SingleTurnSample(
        reference="The cat sat on the mat. It was a sunny day.",
        response="The cat sat on the mat. Dogs barked.",
    )
    score = await metric._single_turn_ascore(sample, None)

    assert score == 0.40
    mock_sacrebleu.assert_called_once_with(
        ["The cat sat on the mat. Dogs barked."],
        [["The cat sat on the mat. It was a sunny day."]],
        **metric.kwargs,
    )


@pytest.mark.asyncio
async def test_bleu_score_collections_scores_full_text():
    """Regression test for the v2 collections BleuScore: full text as a
    single segment, verified against real sacrebleu."""
    metric = CollectionsBleuScore()

    result = await metric.ascore(
        reference="The cat sat on the mat. It was a sunny day.",
        response="The cat sat on the mat. Dogs barked.",
    )

    # If only the first sentence were evaluated the score would be 1.0.
    # The correct full-text BLEU is substantially lower.
    assert result.value < 0.9
