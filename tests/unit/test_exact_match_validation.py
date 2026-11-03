import pytest

from ragas.metrics.collections import ExactMatch


@pytest.mark.asyncio
async def test_exact_match_rejects_missing_reference():
    metric = ExactMatch()

    with pytest.raises(AssertionError, match="valid reference string"):
        await metric.ascore(reference=None, response="Hello World")


@pytest.mark.asyncio
async def test_exact_match_rejects_missing_response():
    metric = ExactMatch()

    with pytest.raises(AssertionError, match="valid response string"):
        await metric.ascore(reference="Hello World", response=None)


@pytest.mark.asyncio
async def test_exact_match_still_scores_strings():
    metric = ExactMatch()

    assert (await metric.ascore(reference="a", response="a")).value == 1.0
    assert (await metric.ascore(reference="a", response="b")).value == 0.0
