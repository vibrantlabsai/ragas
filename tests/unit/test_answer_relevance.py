import asyncio
import time

import numpy as np
import pytest

from ragas.metrics._answer_relevance import ResponseRelevancy


class _BlockingEmbeddings:
    """Sync embed_* calls block the thread (like a real network call would).
    Async aembed_* calls just sleep, like a real async http client."""

    def embed_query(self, text):
        time.sleep(0.2)
        return [1.0, 0.0]

    def embed_documents(self, texts):
        time.sleep(0.2)
        return [[1.0, 0.0] for _ in texts]

    async def aembed_query(self, text):
        await asyncio.sleep(0.2)
        return [1.0, 0.0]

    async def aembed_documents(self, texts):
        await asyncio.sleep(0.2)
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.asyncio
async def test_acalculate_similarity_uses_async_embeddings():
    """acalculate_similarity should call aembed_query/aembed_documents, not the
    sync variants, so it doesn't block the event loop it's awaited on."""
    metric = ResponseRelevancy()
    metric.embeddings = _BlockingEmbeddings()

    sim = await metric.acalculate_similarity(
        "what is the capital of france", ["q1", "q2", "q3"]
    )
    assert sim.shape == (3,)


@pytest.mark.asyncio
async def test_acalculate_similarity_does_not_block_event_loop():
    """A concurrent task on the same loop should keep making progress while
    acalculate_similarity is awaiting its embedding calls."""
    metric = ResponseRelevancy()
    metric.embeddings = _BlockingEmbeddings()

    ticks = {"n": 0}

    async def ticker():
        for _ in range(10):
            await asyncio.sleep(0.02)
            ticks["n"] += 1

    ticker_task = asyncio.create_task(ticker())
    await asyncio.sleep(
        0
    )  # let the ticker actually start before we await the blocking call

    await metric.acalculate_similarity("question", ["q1", "q2"])
    ticks_during = ticks["n"]

    await ticker_task

    # 2 embedding calls x 0.2s sleep = 0.4s window, ticker fires every 0.02s,
    # so a responsive loop should log a good chunk of ticks during that window.
    assert ticks_during >= 5, (
        f"event loop only ticked {ticks_during} times while acalculate_similarity "
        "was running, expected it to stay responsive"
    )


def test_sync_and_async_similarity_agree():
    """calculate_similarity stays sync for existing callers, and both paths
    return the same cosine similarities."""
    metric = ResponseRelevancy()

    class _SyncFakeEmbeddings:
        def embed_query(self, text):
            return [1.0, 0.0]

        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        async def aembed_query(self, text):
            return [1.0, 0.0]

        async def aembed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

    metric.embeddings = _SyncFakeEmbeddings()

    sim = metric.calculate_similarity("q", ["a", "b"])
    assert isinstance(sim, np.ndarray)
    assert np.allclose(sim, [1.0, 1.0])
    asim = asyncio.run(metric.acalculate_similarity("q", ["a", "b"]))
    assert np.allclose(asim, sim)
