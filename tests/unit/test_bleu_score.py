"""Tests for BleuScore (legacy and collections implementations).

Regression tests for multi-sentence inputs: every sentence of the response and
reference must contribute to the score, not just the first one.
"""

from unittest.mock import patch

import pytest

try:
    from sacrebleu import corpus_bleu
except ImportError:
    pytest.skip("sacrebleu not available", allow_module_level=True)

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore
from ragas.metrics.collections import BleuScore as BleuScoreCollections

REFERENCE = "The capital of France is Paris. It is known for the Eiffel Tower."
RIGHT_THEN_JUNK = "The capital of France is Paris. Bananas are a popular fruit."
JUNK_THEN_RIGHT = "Bananas are a popular fruit. The capital of France is Paris."


async def legacy_score(reference: str, response: str, **kwargs) -> float:
    metric = BleuScore(kwargs=kwargs)
    sample = SingleTurnSample(reference=reference, response=response)
    return await metric._single_turn_ascore(sample, None)


async def collections_score(reference: str, response: str, **kwargs) -> float:
    metric = BleuScoreCollections(kwargs=kwargs)
    result = await metric.ascore(reference=reference, response=response)
    return result.value


def one_segment_bleu(reference: str, response: str, **kwargs) -> float:
    return corpus_bleu([response], [[reference]], **kwargs).score / 100


SCORERS = [
    pytest.param(legacy_score, id="legacy"),
    pytest.param(collections_score, id="collections"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("score", SCORERS)
async def test_identical_multi_sentence_text_scores_one(score):
    assert await score(REFERENCE, REFERENCE) == pytest.approx(1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("score", SCORERS)
async def test_later_sentences_are_scored(score):
    """A wrong second sentence must lower the score.

    Before the fix only the first sentence was graded, so this scored 1.0.
    """
    assert await score(REFERENCE, RIGHT_THEN_JUNK) < 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize("score", SCORERS)
async def test_sentence_order_does_not_drop_content(score):
    """Moving the correct sentence second must not wipe out its credit.

    Before the fix this scored about 0.04 while RIGHT_THEN_JUNK scored 1.0.
    """
    right_first = await score(REFERENCE, RIGHT_THEN_JUNK)
    right_second = await score(REFERENCE, JUNK_THEN_RIGHT)
    assert right_second > 0.2
    assert abs(right_first - right_second) < 0.15


@pytest.mark.asyncio
@pytest.mark.parametrize("score", SCORERS)
async def test_different_sentence_counts(score):
    """Unequal sentence counts are scored, not truncated or raised on."""
    response = "The capital of France is Paris."
    assert await score(REFERENCE, response) == pytest.approx(
        one_segment_bleu(REFERENCE, response)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("score", SCORERS)
@pytest.mark.parametrize(
    "response", [RIGHT_THEN_JUNK, JUNK_THEN_RIGHT, "Paris. Eiffel Tower."]
)
async def test_matches_whole_text_bleu(score, response):
    assert await score(REFERENCE, response) == pytest.approx(
        one_segment_bleu(REFERENCE, response)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("score", SCORERS)
async def test_kwargs_are_passed_to_sacrebleu(score):
    kwargs = {"smooth_method": "floor", "smooth_value": 0.1}
    assert await score(REFERENCE, RIGHT_THEN_JUNK, **kwargs) == pytest.approx(
        one_segment_bleu(REFERENCE, RIGHT_THEN_JUNK, **kwargs)
    )


@pytest.mark.asyncio
async def test_legacy_calls_sacrebleu_with_one_segment():
    metric = BleuScore()
    sample = SingleTurnSample(reference=REFERENCE, response=JUNK_THEN_RIGHT)
    with patch.object(metric, "corpus_bleu") as mock_bleu:
        mock_bleu.return_value.score = 50.0
        score = await metric._single_turn_ascore(sample, None)

    assert score == 0.5
    mock_bleu.assert_called_once_with([JUNK_THEN_RIGHT], [[REFERENCE]])


@pytest.mark.asyncio
async def test_collections_calls_sacrebleu_with_one_segment():
    metric = BleuScoreCollections()
    with patch("sacrebleu.corpus_bleu") as mock_bleu:
        mock_bleu.return_value.score = 50.0
        result = await metric.ascore(reference=REFERENCE, response=JUNK_THEN_RIGHT)

    assert result.value == 0.5
    mock_bleu.assert_called_once_with([JUNK_THEN_RIGHT], [[REFERENCE]])
