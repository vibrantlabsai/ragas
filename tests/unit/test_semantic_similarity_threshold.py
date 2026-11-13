"""Offline threshold behavior for both semantic similarity interfaces."""

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.metrics._answer_similarity import (
    AnswerSimilarity,
    SemanticSimilarity as LegacySemanticSimilarity,
)
from ragas.metrics.collections import SemanticSimilarity


class FixedEmbeddings(BaseRagasEmbedding):
    def __init__(self, response_vector):
        super().__init__()
        self.response_vector = response_vector

    def embed_text(self, text, **kwargs):
        return [1.0, 0.0] if text == "reference" else self.response_vector

    async def aembed_text(self, text, **kwargs):
        return self.embed_text(text, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metric_class", [SemanticSimilarity, LegacySemanticSimilarity, AnswerSimilarity]
)
@pytest.mark.parametrize(
    "threshold,response_vector,expected",
    [
        (0.0, [0.6, 0.8], 1.0),
        (0.0, [0.0, 1.0], 1.0),
        (0.0, [-0.6, 0.8], 0.0),
        (None, [0.6, 0.8], 0.6),
        (0.5, [0.6, 0.8], 1.0),
        (0.7, [0.6, 0.8], 0.0),
    ],
)
async def test_semantic_similarity_threshold(
    metric_class, threshold, response_vector, expected
):
    metric = metric_class(
        embeddings=FixedEmbeddings(response_vector), threshold=threshold
    )
    if metric_class is SemanticSimilarity:
        result = await metric.ascore(reference="reference", response="response")
        score = result.value
    else:
        score = await metric.single_turn_ascore(
            SingleTurnSample(reference="reference", response="response")
        )
    assert score == pytest.approx(expected)
