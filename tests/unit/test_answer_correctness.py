"""Unit tests for AnswerCorrectness factuality delegation.

Regression tests for #1958: AnswerCorrectness used to compute factuality
through its own ``StatementGenerator + CorrectnessClassifier`` pipeline,
producing scores that disagreed with the standalone ``FactualCorrectness``
metric on the same input. AnswerCorrectness now delegates factuality to
``FactualCorrectness`` so the two metrics stay in sync.
"""

from __future__ import annotations

import typing as t
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics._factual_correctness import FactualCorrectness


def _build_metric(
    factuality_value: float,
    weights: t.Optional[list[float]] = None,
    similarity_value: t.Optional[float] = None,
) -> AnswerCorrectness:
    metric = AnswerCorrectness(weights=weights or [0.75, 0.25])

    fake_factual = MagicMock(spec=FactualCorrectness)
    fake_factual._single_turn_ascore = AsyncMock(return_value=factuality_value)
    fake_factual.llm = object()
    metric.factual_correctness = fake_factual
    metric.llm = object()

    if metric.weights[1] != 0:
        fake_similarity = MagicMock()
        fake_similarity.single_turn_ascore = AsyncMock(
            return_value=similarity_value if similarity_value is not None else 0.0
        )
        metric.answer_similarity = fake_similarity

    return metric


def _row() -> dict:
    return {
        "user_input": "What powers the sun?",
        "response": "The sun is powered by nuclear fusion.",
        "reference": "The sun runs on nuclear fusion of hydrogen atoms.",
    }


@pytest.mark.asyncio
async def test_factuality_delegated_to_factual_correctness():
    """AnswerCorrectness should call FactualCorrectness for the factuality term."""

    metric = _build_metric(factuality_value=0.8, similarity_value=0.6)

    score = await metric._ascore(_row(), callbacks=[])

    assert metric.factual_correctness._single_turn_ascore.await_count == 1
    expected = float(np.average([0.8, 0.6], weights=[0.75, 0.25]))
    assert score == pytest.approx(expected)


@pytest.mark.asyncio
async def test_zero_factuality_weight_skips_factual_correctness():
    """When weights[0] == 0 the FactualCorrectness call is short-circuited."""

    metric = _build_metric(
        factuality_value=0.8, weights=[0.0, 1.0], similarity_value=0.42
    )

    score = await metric._ascore(_row(), callbacks=[])

    metric.factual_correctness._single_turn_ascore.assert_not_called()
    assert score == pytest.approx(0.42)


@pytest.mark.asyncio
async def test_zero_similarity_weight_skips_answer_similarity():
    """When weights[1] == 0 AnswerSimilarity is not consulted."""

    metric = _build_metric(factuality_value=0.7, weights=[1.0, 0.0])

    score = await metric._ascore(_row(), callbacks=[])

    assert metric.answer_similarity is None
    assert score == pytest.approx(0.7)


def test_init_lazy_constructs_factual_correctness():
    """``init`` should build a FactualCorrectness instance sharing the LLM/beta."""

    from ragas.run_config import RunConfig

    metric = AnswerCorrectness(weights=[0.75, 0.25], beta=2.0)
    metric.llm = MagicMock()
    metric.embeddings = MagicMock()

    metric.init(RunConfig())

    assert metric.factual_correctness is not None
    assert metric.factual_correctness.llm is metric.llm
    assert metric.factual_correctness.beta == 2.0


def test_init_does_not_construct_factual_correctness_when_weight_zero():
    """If factuality weight is 0 we shouldn't build a FactualCorrectness."""

    from ragas.run_config import RunConfig

    metric = AnswerCorrectness(weights=[0.0, 1.0])
    metric.llm = MagicMock()
    metric.embeddings = MagicMock()

    metric.init(RunConfig())

    assert metric.factual_correctness is None


def test_passes_through_user_supplied_factual_correctness_instance():
    """A user-provided FactualCorrectness instance should not be replaced."""

    from ragas.run_config import RunConfig

    user_factual = MagicMock(spec=FactualCorrectness)
    user_factual.llm = None

    metric = AnswerCorrectness(factual_correctness=user_factual)
    metric.llm = MagicMock()
    metric.embeddings = MagicMock()

    metric.init(RunConfig())

    assert metric.factual_correctness is user_factual
    # llm gets back-filled when not preset
    assert metric.factual_correctness.llm is metric.llm


@pytest.mark.asyncio
async def test_score_invokes_factual_correctness_with_sample_payload():
    """The sample reaching FactualCorrectness should carry response + reference."""

    metric = _build_metric(factuality_value=0.5, similarity_value=0.5)

    await metric._ascore(_row(), callbacks=[])

    call_args = metric.factual_correctness._single_turn_ascore.await_args
    sent_sample = call_args.args[0]
    assert isinstance(sent_sample, SingleTurnSample)
    assert sent_sample.response == _row()["response"]
    assert sent_sample.reference == _row()["reference"]
