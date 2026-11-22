"""Faithfulness separates statement generation from optional binary judgments."""

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from pydantic import ValidationError

from ragas.decision_models import BaseDecisionModel, BinaryDecisionResult
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import Faithfulness
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementInput,
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
    StatementGeneratorInput,
    StatementGeneratorOutput,
)

STATEMENTS = [
    "Paris is the capital of France.",
    "Berlin is the capital of France.",
    "Paris is in Europe.",
]
SAMPLE = {
    "user_input": "What is the capital of France and where is it?",
    "response": " ".join(STATEMENTS),
    "retrieved_contexts": [STATEMENTS[0], STATEMENTS[2]],
}


class FakeDecisionModel(BaseDecisionModel):
    async def binary(self, *, instruction, data):
        # Yield so batch tests exercise overlapping metric calls.
        await asyncio.sleep(0)
        probabilities = dict(zip(STATEMENTS, [0.9, 0.1, 0.5]))
        return BinaryDecisionResult.from_probability(
            probabilities[data["statement"]], threshold=0.5
        )


@pytest.fixture
def generator():
    llm = MagicMock(spec=InstructorBaseRagasLLM)
    llm.agenerate = AsyncMock(
        return_value=StatementGeneratorOutput(statements=STATEMENTS)
    )
    return llm


@pytest.mark.asyncio
async def test_default_llm_prompts_schemas_and_aggregation_unchanged(generator):
    generator.agenerate.side_effect = [
        StatementGeneratorOutput(statements=STATEMENTS),
        NLIStatementOutput(
            statements=[
                StatementFaithfulnessAnswer(
                    statement=statement, reason="Controlled judgment", verdict=value
                )
                for statement, value in zip(STATEMENTS, [1, 0, 1])
            ]
        ),
    ]
    metric = Faithfulness(generator, "custom_faithfulness")
    metric.statement_generator_prompt.instruction = "Custom statement instruction"
    metric.nli_statement_prompt.instruction = "Custom entailment instruction"
    result = await metric.ascore(**SAMPLE)

    assert metric.name == "custom_faithfulness"
    assert result.value == pytest.approx(2 / 3)
    assert result.traces is None
    assert result.reason is None
    assert generator.agenerate.await_args_list == [
        call(
            metric.statement_generator_prompt.to_string(
                StatementGeneratorInput(
                    question=SAMPLE["user_input"], answer=SAMPLE["response"]
                )
            ),
            StatementGeneratorOutput,
        ),
        call(
            metric.nli_statement_prompt.to_string(
                NLIStatementInput(
                    context="\n".join(SAMPLE["retrieved_contexts"]),
                    statements=STATEMENTS,
                )
            ),
            NLIStatementOutput,
        ),
    ]


@pytest.mark.asyncio
async def test_decision_path_keeps_generation_and_preserves_statement_results(
    generator,
):
    model = FakeDecisionModel()
    model.binary = AsyncMock(wraps=model.binary)
    metric = Faithfulness(llm=generator, decision_model=model)
    result = await metric.ascore(**SAMPLE)

    assert result.value == pytest.approx(2 / 3)
    assert result.traces == {
        "output": {
            "decisions": [
                {"statement": STATEMENTS[0], "value": True, "probability": 0.9},
                {"statement": STATEMENTS[1], "value": False, "probability": 0.1},
                {"statement": STATEMENTS[2], "value": True, "probability": 0.5},
            ]
        }
    }
    generator.agenerate.assert_awaited_once_with(
        metric.statement_generator_prompt.to_string(
            StatementGeneratorInput(
                question=SAMPLE["user_input"], answer=SAMPLE["response"]
            )
        ),
        StatementGeneratorOutput,
    )
    assert [c.kwargs["data"] for c in model.binary.await_args_list] == [
        {"context": "\n".join(SAMPLE["retrieved_contexts"]), "statement": statement}
        for statement in STATEMENTS
    ]
    assert all(
        "directly inferred" in c.kwargs["instruction"]
        for c in model.binary.await_args_list
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("probability", [None, 0.1, 0.5, 0.9])
async def test_selected_verdict_controls_score_without_rethresholding(
    generator, value, probability
):
    model = MagicMock(spec=BaseDecisionModel)
    model.binary = AsyncMock(
        return_value=BinaryDecisionResult(value=value, probability=probability)
    )
    result = await Faithfulness(generator, decision_model=model).ascore(**SAMPLE)
    assert result.value == float(value)
    assert result.traces is not None
    assert result.traces["output"]["decisions"][0]["probability"] == probability


def test_sync_score_uses_async_generation_and_decisions(generator):
    metric = Faithfulness(generator, decision_model=FakeDecisionModel())
    assert metric.score(**SAMPLE).value == pytest.approx(2 / 3)
    generator.agenerate.assert_awaited_once()
    generator.generate.assert_not_called()


@pytest.mark.asyncio
async def test_async_batch_keeps_decisions_local_to_each_result(generator):
    generator.agenerate.side_effect = [
        StatementGeneratorOutput(statements=STATEMENTS),
        StatementGeneratorOutput(statements=[STATEMENTS[1]]),
    ]
    metric = Faithfulness(generator, decision_model=FakeDecisionModel())
    unsupported = {**SAMPLE, "response": STATEMENTS[1]}
    results = await metric.abatch_score([SAMPLE, unsupported])

    assert [r.value for r in results] == pytest.approx([2 / 3, 0])
    assert results[0].traces is not None
    assert results[1].traces is not None
    assert len(results[0].traces["output"]["decisions"]) == 3
    assert results[1].traces["output"]["decisions"] == [
        {"statement": STATEMENTS[1], "value": False, "probability": 0.1}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_decision_model", [False, True])
async def test_empty_statements_keep_nan_and_skip_judgments(
    generator, use_decision_model
):
    generator.agenerate.return_value = StatementGeneratorOutput(statements=[])
    model = MagicMock(spec=BaseDecisionModel)
    metric = Faithfulness(
        generator, decision_model=model if use_decision_model else None
    )
    result = await metric.ascore(**SAMPLE)

    assert math.isnan(result.value)
    assert result.traces is None
    generator.agenerate.assert_awaited_once()
    model.binary.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["user_input", "response", "retrieved_contexts"])
@pytest.mark.parametrize("use_decision_model", [False, True])
async def test_input_validation_precedes_model_calls(
    generator, field, use_decision_model
):
    model = MagicMock(spec=BaseDecisionModel)
    metric = Faithfulness(
        generator, decision_model=model if use_decision_model else None
    )
    sample = {**SAMPLE, field: [] if field == "retrieved_contexts" else ""}
    with pytest.raises(ValueError, match=f"{field} is missing"):
        await metric.ascore(**sample)
    generator.agenerate.assert_not_called()
    model.binary.assert_not_called()


@pytest.mark.parametrize("llm", [None, object()])
def test_decision_model_does_not_remove_generator_requirement(llm):
    with pytest.raises(ValueError, match="only support modern InstructorLLM"):
        Faithfulness(llm=llm, decision_model=FakeDecisionModel())


def test_invalid_decision_model_rejected(generator):
    with pytest.raises(TypeError, match="decision_model must be a BaseDecisionModel"):
        Faithfulness(generator, decision_model=object())  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [True, 1, 0.9, {"value": True}, None])
async def test_unsupported_decision_output_rejected(generator, output):
    model = MagicMock(spec=BaseDecisionModel)
    model.binary = AsyncMock(return_value=output)
    with pytest.raises(TypeError, match="must return BinaryDecisionResult"):
        await Faithfulness(generator, decision_model=model).ascore(**SAMPLE)
    generator.agenerate.assert_awaited_once()


@pytest.mark.asyncio
async def test_unvalidated_decision_result_rejected(generator):
    model = MagicMock(spec=BaseDecisionModel)
    model.binary = AsyncMock(
        return_value=BinaryDecisionResult.model_construct(value="false", probability=2)
    )
    with pytest.raises(ValidationError):
        await Faithfulness(generator, decision_model=model).ascore(**SAMPLE)


@pytest.mark.asyncio
async def test_unsupported_task_does_not_fall_back_to_llm(generator):
    model = MagicMock(spec=BaseDecisionModel)
    model.binary = AsyncMock(side_effect=NotImplementedError("unsupported task"))
    with pytest.raises(NotImplementedError, match="unsupported task"):
        await Faithfulness(generator, decision_model=model).ascore(**SAMPLE)
    generator.agenerate.assert_awaited_once()


@pytest.mark.asyncio
async def test_generator_failure_prevents_decisions(generator):
    generator.agenerate.side_effect = RuntimeError("generation failed")
    model = MagicMock(spec=BaseDecisionModel)
    with pytest.raises(RuntimeError, match="generation failed"):
        await Faithfulness(generator, decision_model=model).ascore(**SAMPLE)
    model.binary.assert_not_called()
