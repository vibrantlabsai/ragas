"""Offline tests for the collections SemanticSimilarity metric."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from ragas.embeddings.openai_provider import OpenAIEmbeddings
from ragas.metrics.collections import SemanticSimilarity


@pytest.fixture(params=[False, True], ids=["sync-client", "async-client"])
def embeddings(request):
    """Use the real provider with SDK calls that check their execution context."""
    thread_id = threading.get_ident()
    calls = []
    vectors = {"reference": [1.0, 0.0], "response": [0.6, 0.8]}

    def response(input, model):
        assert model == "test-embedding"
        calls.append(input)
        return SimpleNamespace(data=[SimpleNamespace(embedding=vectors[input])])

    if request.param:

        async def create(input, model):
            # A native async client must stay on its caller's thread/event loop.
            assert threading.get_ident() == thread_id
            await asyncio.sleep(0)
            return response(input, model)

    else:

        def create(input, model):
            # A blocking SDK call must not run on the event-loop thread.
            assert threading.get_ident() != thread_id
            return response(input, model)

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    return OpenAIEmbeddings(client=client, model="test-embedding"), calls


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold, expected", [(None, 0.6), (0.5, 1.0), (0.9, 0.0)])
async def test_ascore_uses_nonblocking_embeddings(embeddings, threshold, expected):
    provider, calls = embeddings
    metric = SemanticSimilarity(embeddings=provider, threshold=threshold)

    result = await metric.ascore(reference="reference", response="response")

    assert result.value == pytest.approx(expected)
    assert calls == ["reference", "response"]


@pytest.mark.asyncio
async def test_abatch_score_preserves_input_order(embeddings):
    provider, calls = embeddings
    metric = SemanticSimilarity(embeddings=provider)

    results = await metric.abatch_score(
        [
            {"reference": "reference", "response": "response"},
            {"reference": "reference", "response": "reference"},
        ]
    )

    assert [result.value for result in results] == pytest.approx([0.6, 1.0])
    assert len(calls) == 4


def test_score_preserves_sync_entry_point(embeddings):
    provider, calls = embeddings
    metric = SemanticSimilarity(embeddings=provider)

    result = metric.score(reference="reference", response="response")

    assert result.value == pytest.approx(0.6)
    assert calls == ["reference", "response"]
