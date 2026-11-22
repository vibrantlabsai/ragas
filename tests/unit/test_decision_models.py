"""Contracts for bounded judgments and the opt-in LLM adapter."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from ragas.decision_models import (
    BaseDecisionModel,
    BinaryDecisionResult,
    LLMDecisionModel,
)
from ragas.llms.base import InstructorBaseRagasLLM


@pytest.mark.parametrize(
    "probability, threshold, expected",
    [
        (0.9, 0.5, True),
        (0.1, 0.5, False),
        (0.4999, 0.5, False),
        (0.5, 0.5, True),
        (0.5001, 0.5, True),
        (0.72, 0.8, False),
        (0.0, 0.0, True),
        (1.0, 1.0, True),
    ],
)
def test_explicit_threshold_preserves_probability(probability, threshold, expected):
    result = BinaryDecisionResult.from_probability(probability, threshold=threshold)
    assert result.value is expected
    assert result.probability == probability
    assert BinaryDecisionResult.model_validate_json(result.model_dump_json()) == result


def test_no_implicit_threshold():
    with pytest.raises(TypeError, match="threshold"):
        BinaryDecisionResult.from_probability(0.9)  # type: ignore


@pytest.mark.parametrize(
    "threshold", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"]
)
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        BinaryDecisionResult.from_probability(0.9, threshold=threshold)


@pytest.mark.parametrize(
    "probability", [-0.1, 1.1, float("nan"), float("inf"), True, "0.9"]
)
def test_invalid_probability(probability):
    with pytest.raises(ValidationError):
        BinaryDecisionResult(value=True, probability=probability)
    with pytest.raises(ValidationError):
        BinaryDecisionResult.from_probability(probability, threshold=0.5)


@pytest.mark.parametrize("value", [0, 1, "true", "false", None, 0.5])
def test_verdict_is_strictly_boolean(value):
    with pytest.raises(ValidationError):
        BinaryDecisionResult(value=value)


def test_result_requires_value_and_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        BinaryDecisionResult.model_validate({"probability": 0.9})
    with pytest.raises(ValidationError):
        BinaryDecisionResult.model_validate({"value": True, "confidence": 0.9})


def test_result_is_immutable_and_probability_is_optional():
    result = BinaryDecisionResult(value=False)
    assert result.probability is None
    with pytest.raises(ValidationError):
        result.value = True  # type: ignore


def test_base_model_requires_binary_implementation():
    with pytest.raises(TypeError):
        BaseDecisionModel()  # type: ignore


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, False])
async def test_llm_adapter_uses_structured_generation_without_confidence(value):
    llm = MagicMock(spec=InstructorBaseRagasLLM)

    async def generate(prompt, response_model):
        assert '"value"' in prompt
        assert json.dumps({"context": 'A "quoted" fact\nwith a newline'}) in prompt
        assert set(response_model.model_fields) == {"value"}
        return response_model(value=value)

    llm.agenerate = AsyncMock(side_effect=generate)
    result = await LLMDecisionModel(llm).binary(
        instruction="Is the fact supported?",
        data={"context": 'A "quoted" fact\nwith a newline'},
    )
    assert result == BinaryDecisionResult(value=value)
    assert llm.agenerate.await_args.args[0].startswith("Is the fact supported?")
    llm.generate.assert_not_called()


def test_llm_adapter_rejects_unsupported_llm():
    with pytest.raises(TypeError, match="InstructorBaseRagasLLM"):
        LLMDecisionModel(object())  # type: ignore


@pytest.mark.asyncio
async def test_llm_adapter_propagates_provider_error():
    llm = MagicMock(spec=InstructorBaseRagasLLM)
    llm.agenerate = AsyncMock(side_effect=RuntimeError("provider unavailable"))
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await LLMDecisionModel(llm).binary(instruction="Supported?", data={})
