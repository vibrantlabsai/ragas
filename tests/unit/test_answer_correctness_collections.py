"""Regression tests for empty statement sets in collections AnswerCorrectness."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections.answer_correctness import AnswerCorrectness
from ragas.metrics.collections.answer_correctness.util import (
    ClassificationWithReason,
    StatementGeneratorOutput,
)

CLAIM = "Paris is the capital of France."
STATEMENT = {"statement": CLAIM, "reason": "Synthetic classification for this test."}
WRONG_CLAIM = "Lyon is the capital of France."


def make_metric(
    response_statements, reference_statements, classification=None, weights=None
):
    """Mock model I/O only, validating each response with its requested schema."""
    payloads = [
        {"statements": response_statements},
        {"statements": reference_statements},
    ]
    if classification is not None:
        payloads.append(classification)
    pending = iter(payloads)

    async def generate(prompt, response_model):
        return response_model.model_validate_json(json.dumps(next(pending)))

    llm = MagicMock(spec=InstructorBaseRagasLLM)
    llm.agenerate = AsyncMock(side_effect=generate)
    embeddings = None
    if weights is not None:
        embeddings = MagicMock(spec=BaseRagasEmbedding)
        embeddings.aembed_text = AsyncMock(side_effect=[[1.0, 0.0], [0.0, 1.0]])

    metric = AnswerCorrectness(
        llm=llm,
        embeddings=embeddings,
        weights=weights if weights is not None else [1.0, 0.0],
    )
    return metric, llm


@pytest.mark.parametrize(
    "response_statements, reference_statements, classification",
    [
        ([], [CLAIM], {"TP": [], "FP": [], "FN": [STATEMENT]}),
        ([CLAIM], [], {"TP": [], "FP": [STATEMENT], "FN": []}),
    ],
)
@pytest.mark.parametrize("weights", [None, [0.75, 0.25]])
@pytest.mark.asyncio
async def test_one_empty_statement_set_is_classified(
    response_statements, reference_statements, classification, weights
):
    metric, llm = make_metric(
        response_statements, reference_statements, classification, weights
    )

    result = await metric.ascore(
        user_input="What is the capital of France?",
        response=CLAIM if response_statements else "",
        reference=CLAIM if reference_statements else "",
    )

    assert result.value == 0.0
    assert [call.args[1] for call in llm.agenerate.await_args_list] == [
        StatementGeneratorOutput,
        StatementGeneratorOutput,
        ClassificationWithReason,
    ]


@pytest.mark.asyncio
async def test_both_empty_statement_sets_preserve_perfect_match():
    metric, llm = make_metric([], [])

    result = await metric.ascore(
        user_input="What is the capital of France?", response="", reference=""
    )

    assert result.value == 1.0
    assert llm.agenerate.await_count == 2


@pytest.mark.parametrize(
    "response_claim, classification, expected",
    [
        (CLAIM, {"TP": [STATEMENT], "FP": [], "FN": []}, 1.0),
        (
            WRONG_CLAIM,
            {
                "TP": [],
                "FP": [{"statement": WRONG_CLAIM, "reason": "Unsupported claim."}],
                "FN": [STATEMENT],
            },
            0.0,
        ),
    ],
)
@pytest.mark.asyncio
async def test_nonempty_statement_sets_still_use_classifier(
    response_claim, classification, expected
):
    metric, llm = make_metric([response_claim], [CLAIM], classification)

    result = await metric.ascore(
        user_input="What is the capital of France?",
        response=response_claim,
        reference=CLAIM,
    )

    assert result.value == expected
    assert llm.agenerate.await_count == 3
