from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from ragas.metrics._answer_relevance import AnswerRelevancy


@pytest.mark.asyncio
async def test_answer_relevancy_uses_async_embeddings():
    metric = AnswerRelevancy()

    mock_embeddings = MagicMock()
    mock_embeddings.aembed_query = AsyncMock(return_value=[1.0, 0.0])
    mock_embeddings.aembed_documents = AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])

    metric.embeddings = mock_embeddings

    similarity = await metric.acalculate_similarity(
        "What is AI?", ["What is AI?", "What is bread?"]
    )

    # Verify that async methods were awaited
    mock_embeddings.aembed_query.assert_awaited_once_with("What is AI?")
    mock_embeddings.aembed_documents.assert_awaited_once_with(
        ["What is AI?", "What is bread?"]
    )

    assert len(similarity) == 2
    assert np.isclose(similarity[0], 1.0)
    assert np.isclose(similarity[1], 0.0)


@pytest.mark.asyncio
async def test_answer_relevancy_fallback_to_sync_embeddings():
    metric = AnswerRelevancy()

    # Object without aembed_query/aembed_documents attributes
    class SyncOnlyEmbeddings:
        def embed_query(self, text: str):
            return [1.0, 0.0]

        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

    metric.embeddings = SyncOnlyEmbeddings()

    similarity = await metric.acalculate_similarity("query", ["doc1", "doc2"])
    assert len(similarity) == 2
    assert np.isclose(similarity[0], 1.0)
    assert np.isclose(similarity[1], 1.0)


def test_answer_relevancy_sync_calculate_similarity_preserved():
    metric = AnswerRelevancy()

    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [1.0, 0.0]
    mock_embeddings.embed_documents.return_value = [[1.0, 0.0]]

    metric.embeddings = mock_embeddings

    similarity = metric.calculate_similarity("query", ["doc1"])
    assert len(similarity) == 1
    assert np.isclose(similarity[0], 1.0)
    mock_embeddings.embed_query.assert_called_once_with("query")
    mock_embeddings.embed_documents.assert_called_once_with(["doc1"])
