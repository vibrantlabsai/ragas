"""Context Precision keeps its aggregation across LLM and decision paths."""

from unittest.mock import AsyncMock, MagicMock, call

import pytest
from pydantic import ValidationError

from ragas.decision_models import BaseDecisionModel, BinaryDecisionResult
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import (
    ContextPrecision,
    ContextPrecisionWithoutReference,
    ContextPrecisionWithReference,
    ContextUtilization,
)
from ragas.metrics.collections.context_precision.util import (
    ContextPrecisionInput,
    ContextPrecisionOutput,
)

SAMPLE = {
    "user_input": "What is the capital of France?",
    "reference": "Paris is the capital of France.",
    "retrieved_contexts": [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "France's capital city is Paris.",
    ],
}


class FakeDecisionModel(BaseDecisionModel):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.calls = []

    async def binary(self, *, instruction, data):
        self.calls.append((instruction, data))
        probability = 0.1 if "Berlin" in data["context"] else 0.9
        return BinaryDecisionResult.from_probability(
            probability, threshold=self.threshold
        )


@pytest.fixture
def llm():
    model = MagicMock(spec=InstructorBaseRagasLLM)
    model.agenerate = AsyncMock(
        side_effect=[
            ContextPrecisionOutput(reason="Deterministic judgment", verdict=value)
            for value in [1, 0, 1]
        ]
    )
    return model


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metric_class, answer_field",
    [
        (ContextPrecision, "reference"),
        (ContextPrecisionWithReference, "reference"),
        (ContextPrecisionWithoutReference, "response"),
        (ContextUtilization, "response"),
    ],
)
async def test_existing_llm_prompt_schema_and_score_unchanged(
    llm, metric_class, answer_field
):
    metric = metric_class(llm)
    # User-customized prompts must still reach the LLM unchanged.
    metric.prompt.instruction = "Custom usefulness instruction"
    sample = {**SAMPLE}
    sample[answer_field] = sample.pop("reference")
    result = await metric.ascore(**sample)

    assert result.value == pytest.approx(5 / 6)
    assert result.traces is None
    assert result.reason is None
    assert llm.agenerate.await_args_list == [
        call(
            metric.prompt.to_string(
                ContextPrecisionInput(
                    question=SAMPLE["user_input"],
                    context=context,
                    answer=SAMPLE["reference"],
                )
            ),
            ContextPrecisionOutput,
        )
        for context in SAMPLE["retrieved_contexts"]
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metric_class", [ContextPrecision, ContextPrecisionWithReference]
)
async def test_decision_path_matches_llm_and_preserves_ordered_results(
    llm, metric_class
):
    model = FakeDecisionModel()
    metric = metric_class(decision_model=model)
    result = await metric.ascore(**SAMPLE)
    legacy_path = await metric_class(llm=llm).ascore(**SAMPLE)

    assert result.value == legacy_path.value
    assert result.value == pytest.approx(5 / 6)
    assert result.traces == {
        "output": {
            "decisions": [
                {"value": True, "probability": 0.9},
                {"value": False, "probability": 0.1},
                {"value": True, "probability": 0.9},
            ]
        }
    }
    assert [data for _, data in model.calls] == [
        {
            "question": SAMPLE["user_input"],
            "context": context,
            "answer": SAMPLE["reference"],
        }
        for context in SAMPLE["retrieved_contexts"]
    ]
    assert all("useful" in instruction for instruction, _ in model.calls)


@pytest.mark.asyncio
async def test_explicit_decision_model_takes_precedence(llm):
    metric = ContextPrecision(llm=llm, decision_model=FakeDecisionModel())
    assert (await metric.ascore(**SAMPLE)).value == pytest.approx(5 / 6)
    llm.agenerate.assert_not_called()


def test_sync_score_and_batch_score_use_async_decision_path():
    metric = ContextPrecision(decision_model=FakeDecisionModel())
    assert metric.score(**SAMPLE).value == pytest.approx(5 / 6)
    assert [r.value for r in metric.batch_score([SAMPLE, SAMPLE])] == pytest.approx(
        [5 / 6, 5 / 6]
    )


@pytest.mark.asyncio
async def test_async_batch_keeps_traces_local_to_each_result():
    metric = ContextPrecision(decision_model=FakeDecisionModel())
    irrelevant = {**SAMPLE, "retrieved_contexts": ["Berlin is in Germany."]}
    results = await metric.abatch_score([SAMPLE, irrelevant])
    assert [r.value for r in results] == pytest.approx([5 / 6, 0])
    assert results[0].traces is not None
    assert results[1].traces is not None
    assert len(results[0].traces["output"]["decisions"]) == 3
    assert results[1].traces["output"]["decisions"] == [
        {"value": False, "probability": 0.1}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("probability", [None, 0.1, 0.5, 0.9])
async def test_metric_uses_selected_value_without_rethresholding(probability):
    model = MagicMock(spec=BaseDecisionModel)
    model.binary = AsyncMock(
        return_value=BinaryDecisionResult(value=False, probability=probability)
    )
    result = await ContextPrecision(decision_model=model).ascore(**SAMPLE)
    assert result.value == 0.0
    assert result.traces is not None
    assert result.traces["output"]["decisions"][0]["probability"] == probability


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [True, 1, 0.9, {"value": True}, None])
async def test_unsupported_decision_output_rejected(output):
    model = MagicMock(spec=BaseDecisionModel)
    model.binary = AsyncMock(return_value=output)
    with pytest.raises(TypeError, match="must return BinaryDecisionResult"):
        await ContextPrecision(decision_model=model).ascore(**SAMPLE)


@pytest.mark.asyncio
async def test_unvalidated_decision_result_rejected():
    model = MagicMock(spec=BaseDecisionModel)
    model.binary = AsyncMock(
        return_value=BinaryDecisionResult.model_construct(value="false", probability=2)
    )
    with pytest.raises(ValidationError):
        await ContextPrecision(decision_model=model).ascore(**SAMPLE)


@pytest.mark.asyncio
async def test_unsupported_task_propagates_without_llm_fallback(llm):
    model = MagicMock(spec=BaseDecisionModel)
    model.binary = AsyncMock(side_effect=NotImplementedError("unsupported task"))
    metric = ContextPrecision(llm=llm, decision_model=model)
    with pytest.raises(NotImplementedError, match="unsupported task"):
        await metric.ascore(**SAMPLE)
    llm.agenerate.assert_not_called()


@pytest.mark.parametrize(
    "metric_class", [ContextPrecision, ContextPrecisionWithReference]
)
def test_existing_llm_validation_is_preserved(metric_class):
    with pytest.raises(ValueError, match="only support modern InstructorLLM"):
        metric_class(llm=None)
    with pytest.raises(ValueError, match="only support modern InstructorLLM"):
        metric_class(llm=object())
    with pytest.raises(TypeError, match="decision_model must be"):
        metric_class(decision_model=object())


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["user_input", "reference", "retrieved_contexts"])
async def test_decision_path_preserves_input_validation(field):
    model = FakeDecisionModel()
    sample = {**SAMPLE, field: [] if field == "retrieved_contexts" else ""}
    with pytest.raises(ValueError, match=f"{field} cannot be empty"):
        await ContextPrecision(decision_model=model).ascore(**sample)
    assert model.calls == []
