import pytest

pytest.importorskip("rapidfuzz")

from ragas.metrics.collections import DistanceMeasure, NonLLMStringSimilarity


def test_invalid_distance_measure_lists_the_valid_options():
    with pytest.raises(ValueError, match="Valid options are"):
        NonLLMStringSimilarity(distance_measure="euclidean")


def test_non_string_distance_measure_is_rejected():
    with pytest.raises(ValueError, match="must be a DistanceMeasure"):
        NonLLMStringSimilarity(distance_measure=3)


def test_distance_measure_accepts_the_enum_value():
    metric = NonLLMStringSimilarity(distance_measure="levenshtein")

    assert metric.distance_measure is DistanceMeasure.LEVENSHTEIN


@pytest.mark.asyncio
async def test_scoring_still_works():
    metric = NonLLMStringSimilarity()

    result = await metric.ascore(reference="kitten", response="sitting")

    assert 0.0 <= result.value <= 1.0
