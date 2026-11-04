"""Regression tests for binary verdicts in the collections Faithfulness metric."""

import json
import math
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import ValidationError
from tenacity import AsyncRetrying, stop_after_attempt

from ragas.llms.base import InstructorBaseRagasLLM, llm_factory
from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
    StatementGeneratorOutput,
)

try:
    from instructor.core import InstructorRetryException
except ImportError:  # Older Instructor versions expose exceptions here.
    from instructor.exceptions import InstructorRetryException


def statement_with_verdict(verdict):
    return {
        "statement": "The device has a pressure of 42 kPa.",
        "reason": "A synthetic judge response.",
        "verdict": verdict,
    }


@pytest.mark.parametrize("verdict", [-1, 2, "-1", "2", 0.5, None, "yes"])
def test_statement_rejects_invalid_verdict(verdict):
    with pytest.raises(ValidationError):
        StatementFaithfulnessAnswer.model_validate(statement_with_verdict(verdict))


@pytest.mark.parametrize("verdict", [-1, 2, "-1", "2", 0.5, None, "yes"])
def test_nli_json_rejects_invalid_verdict(verdict):
    payload = {"statements": [statement_with_verdict(verdict)]}
    with pytest.raises(ValidationError) as exc_info:
        NLIStatementOutput.model_validate_json(json.dumps(payload))
    assert exc_info.value.errors()[0]["loc"] == ("statements", 0, "verdict")


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [(0, 0), (1, 1), ("0", 0), ("1", 1), (False, 0), (True, 1), (0.0, 0), (1.0, 1)],
)
def test_binary_verdict_preserves_integer_coercion(verdict, expected):
    payload = {"statements": [statement_with_verdict(verdict)]}
    result = NLIStatementOutput.model_validate_json(json.dumps(payload))
    assert result.statements[0].verdict == expected
    assert type(result.statements[0].verdict) is int


def test_nli_schema_exposes_binary_integer_bounds():
    schema = NLIStatementOutput.model_json_schema()
    verdict_schema = schema["$defs"]["StatementFaithfulnessAnswer"]["properties"][
        "verdict"
    ]
    assert verdict_schema["type"] == "integer"
    assert verdict_schema["minimum"] == 0
    assert verdict_schema["maximum"] == 1


def metric_with_judge_verdicts(verdicts):
    """Replace model I/O only; validate JSON with the metric's response models."""
    statements = [statement_with_verdict(verdict) for verdict in verdicts]
    payloads = iter(
        [
            {"statements": [statement["statement"] for statement in statements]},
            {"statements": statements},
        ]
    )

    async def generate(prompt, response_model):
        return response_model.model_validate_json(json.dumps(next(payloads)))

    llm = MagicMock(spec=InstructorBaseRagasLLM)
    llm.agenerate = AsyncMock(side_effect=generate)
    return Faithfulness(llm=llm), llm


SAMPLE = {
    "user_input": "What is the device's pressure?",
    "response": "The device has a pressure of 42 kPa.",
    "retrieved_contexts": ["The device has a pressure of 42 kPa."],
}


@pytest.mark.parametrize("verdict", [-1, 2])
@pytest.mark.asyncio
async def test_ascore_rejects_invalid_judge_verdict(verdict):
    metric, llm = metric_with_judge_verdicts([verdict])
    with pytest.raises(ValidationError):
        await metric.ascore(**SAMPLE)
    assert llm.agenerate.await_count == 2
    assert llm.agenerate.await_args_list[0].args[1] is StatementGeneratorOutput
    assert llm.agenerate.await_args_list[1].args[1] is NLIStatementOutput


@pytest.mark.parametrize("verdict", [-1, 2])
def test_score_rejects_invalid_judge_verdict(verdict):
    metric, _ = metric_with_judge_verdicts([verdict])
    with pytest.raises(ValidationError):
        metric.score(**SAMPLE)


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [([0], 0.0), ([1], 1.0), ([0, 1], 0.5), ([1, 0, 1], 2 / 3)],
)
@pytest.mark.asyncio
async def test_ascore_preserves_valid_scores(verdicts, expected):
    metric, _ = metric_with_judge_verdicts(verdicts)
    result = await metric.ascore(**SAMPLE)
    assert result.value == pytest.approx(expected)


def test_empty_verdicts_preserve_nan_score():
    metric, _ = metric_with_judge_verdicts([])
    assert math.isnan(metric._compute_score(NLIStatementOutput(statements=[])))


@pytest.mark.parametrize("final_verdict", [0, 2])
@pytest.mark.asyncio
async def test_instructor_retries_invalid_verdict_before_scoring(
    final_verdict, monkeypatch
):
    """Use real Instructor parsing/retries, replacing only the API response."""
    monkeypatch.setattr("ragas.llms.base.track", lambda event: None)
    payloads = [
        {"statements": [statement_with_verdict(2)["statement"]]},
        {"statements": [statement_with_verdict(2)]},
        {"statements": [statement_with_verdict(final_verdict)]},
    ]
    completions = [
        ChatCompletion(
            id="test-completion",
            created=0,
            model="test-model",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(payload)},
                }
            ],
        )
        for payload in payloads
    ]
    create = AsyncMock(side_effect=completions)
    async with AsyncOpenAI(api_key="unused-test-key") as client:
        monkeypatch.setattr(client.chat.completions, "create", create)
        llm = llm_factory(
            "test-model",
            client=client,
            adapter="instructor",
            max_retries=AsyncRetrying(stop=stop_after_attempt(2), reraise=True),
        )
        metric = Faithfulness(llm=llm)
        if final_verdict == 2:
            with pytest.raises(InstructorRetryException):
                await metric.ascore(**SAMPLE)
        else:
            result = await metric.ascore(**SAMPLE)
            assert result.value == 0.0
    assert create.await_count == 3
