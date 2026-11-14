import typing as t

import numpy as np
import pytest

from ragas.dataset_schema import (
    EvaluationDataset,
    EvaluationResult,
    HumanMessage,
    MultiTurnSample,
    PromptAnnotation,
    SampleAnnotation,
    SingleMetricAnnotation,
    SingleTurnSample,
)

samples = [
    SingleTurnSample(user_input="What is X", response="Y"),
    MultiTurnSample(
        user_input=[HumanMessage(content="What is X")],
        reference="Y",
    ),
]


def create_sample_annotation(metric_output):
    return SampleAnnotation(
        metric_input={
            "response": "",
            "reference": "",
            "user_input": "",
        },
        metric_output=metric_output,
        prompts={
            "single_turn_aspect_critic_prompt": PromptAnnotation(
                prompt_input={
                    "response": "",
                    "reference": "",
                    "user_input": "",
                },
                prompt_output={"reason": "", "verdict": 1},
                edited_output=None,
            )
        },
        is_accepted=True,
        target=None,
    )


def create_evaluation_result(
    scores: t.Optional[t.List[t.Dict[str, t.Any]]] = None,
) -> EvaluationResult:
    class EmptyTrace:
        parent_run_id = None
        children: t.List[str] = []

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(user_input="What is X", response="Y"),
            SingleTurnSample(user_input="What is A", response="B"),
        ]
    )
    return EvaluationResult(
        scores=scores
        or [
            {"faithfulness": 0.8, "answer_relevancy": 0.7, "error_rate": 0.1},
            {"faithfulness": 0.9, "answer_relevancy": 0.9, "error_rate": 0.3},
        ],
        dataset=dataset,
        ragas_traces={"root": EmptyTrace()},
    )


def test_evaluation_result_summary_returns_aggregate_scores():
    result = create_evaluation_result()

    assert result.summary() == pytest.approx(
        {
            "faithfulness": 0.85,
            "answer_relevancy": 0.8,
            "error_rate": 0.2,
        }
    )


def test_evaluation_result_check_thresholds_passes_all_rules():
    result = create_evaluation_result()

    report = result.check_thresholds(
        min_scores={"faithfulness": 0.8},
        max_scores={"error_rate": 0.25},
        ranges={"answer_relevancy": (0.75, 0.85)},
        baseline_scores={"faithfulness": 0.84},
        relative_thresholds={"faithfulness": 0.05},
    )

    assert report["passed"] is True
    assert len(report["checks"]) == 4


def test_evaluation_result_check_thresholds_reports_failures_without_raising():
    result = create_evaluation_result()

    report = result.check_thresholds(
        min_scores={"faithfulness": 0.9, "missing_metric": 0.5},
        max_scores={"error_rate": 0.1},
        ranges={"answer_relevancy": (0.1, 0.5)},
    )

    assert report["passed"] is False
    assert [check["passed"] for check in report["checks"]] == [
        False,
        False,
        False,
        False,
    ]
    assert report["checks"][1]["reason"] == "missing metric"


def test_evaluation_result_check_thresholds_reports_nan_scores():
    result = create_evaluation_result(
        scores=[
            {"faithfulness": np.nan},
            {"faithfulness": np.nan},
        ]
    )

    report = result.check_thresholds(min_scores={"faithfulness": 0.5})

    assert report["passed"] is False
    assert report["checks"][0]["reason"] == "score is missing, NaN, or non-numeric"


def test_evaluation_result_check_thresholds_reports_non_numeric_scores():
    result = create_evaluation_result()
    result._repr_dict["faithfulness"] = "bad-score"

    report = result.check_thresholds(min_scores={"faithfulness": 0.5})

    assert report["passed"] is False
    assert report["checks"][0]["reason"] == "score is missing, NaN, or non-numeric"


def test_evaluation_result_check_thresholds_validates_configuration():
    result = create_evaluation_result()

    with pytest.raises(ValueError, match="min_scores"):
        result.check_thresholds(min_scores={"faithfulness": np.nan})

    with pytest.raises(ValueError, match="ranges"):
        result.check_thresholds(ranges={"faithfulness": (0.9, 0.1)})

    with pytest.raises(ValueError, match="baseline_scores"):
        result.check_thresholds(relative_thresholds={"faithfulness": 0.1})


def test_evaluation_result_assert_thresholds_raises_on_failure():
    result = create_evaluation_result()

    with pytest.raises(AssertionError, match="faithfulness"):
        result.assert_thresholds(min_scores={"faithfulness": 0.9})


def test_loader_sample():
    annotated_samples = [create_sample_annotation(1) for _ in range(10)] + [
        create_sample_annotation(0) for _ in range(10)
    ]
    test_dataset = SingleMetricAnnotation(name="metric", samples=annotated_samples)
    sample = test_dataset.sample(2)
    assert len(sample) == 2

    sample = test_dataset.sample(2, stratify_key="metric_output")
    assert len(sample) == 2
    assert sum(item["metric_output"] for item in sample) == 1


def test_loader_batch():
    annotated_samples = [create_sample_annotation(1) for _ in range(10)] + [
        create_sample_annotation(0) for _ in range(10)
    ]
    dataset = SingleMetricAnnotation(name="metric", samples=annotated_samples)
    batches = dataset.batch(batch_size=2)
    assert all([len(item) == 2 for item in batches])

    batches = dataset.stratified_batches(batch_size=2, stratify_key="metric_output")
    assert all(sum([item["metric_output"] for item in batch]) == 1 for batch in batches)


@pytest.mark.parametrize("eval_sample", samples)
def test_evaluation_dataset(eval_sample):
    dataset = EvaluationDataset(samples=[eval_sample, eval_sample])

    hf_dataset = dataset.to_hf_dataset()

    assert dataset.get_sample_type() is type(eval_sample)
    assert len(hf_dataset) == 2
    assert len(dataset) == 2
    assert dataset[0] == eval_sample

    dataset_from_hf = EvaluationDataset.from_hf_dataset(hf_dataset)
    assert dataset_from_hf == dataset


@pytest.mark.parametrize("eval_sample", samples)
def test_evaluation_dataset_save_load_csv(tmpdir, eval_sample):
    dataset = EvaluationDataset(samples=[eval_sample, eval_sample])

    # save and load to csv
    csv_path = tmpdir / "csvfile.csv"
    dataset.to_csv(csv_path)


@pytest.mark.parametrize("eval_sample", samples)
def test_evaluation_dataset_save_load_jsonl(tmpdir, eval_sample):
    dataset = EvaluationDataset(samples=[eval_sample, eval_sample])

    # save and load to jsonl
    jsonl_path = tmpdir / "jsonlfile.jsonl"
    dataset.to_jsonl(jsonl_path)
    loaded_dataset = EvaluationDataset.from_jsonl(jsonl_path)
    assert loaded_dataset == dataset


@pytest.mark.parametrize("eval_sample", samples)
def test_evaluation_dataset_load_from_hf(eval_sample):
    dataset = EvaluationDataset(samples=[eval_sample, eval_sample])

    # convert to and load from hf dataset
    hf_dataset = dataset.to_hf_dataset()
    loaded_dataset = EvaluationDataset.from_hf_dataset(hf_dataset)
    assert loaded_dataset == dataset


def test_single_turn_sample_metadata_roundtrip_hf_and_jsonl(tmpdir):
    sample = SingleTurnSample(
        user_input="Q",
        response="A",
        reference_contexts=["ctx"],
        persona_name="Researcher",
        query_style="FORMAL",
        query_length="SHORT",
    )
    dataset = EvaluationDataset(samples=[sample])

    # HF round-trip
    hf = dataset.to_hf_dataset()
    loaded_hf = EvaluationDataset.from_hf_dataset(hf)
    assert loaded_hf.samples[0].persona_name == "Researcher"
    assert loaded_hf.samples[0].query_style == "FORMAL"
    assert loaded_hf.samples[0].query_length == "SHORT"

    # JSONL round-trip
    jsonl_path = tmpdir / "ds.jsonl"
    dataset.to_jsonl(jsonl_path)
    loaded_jsonl = EvaluationDataset.from_jsonl(jsonl_path)
    assert loaded_jsonl.samples[0].persona_name == "Researcher"
    assert loaded_jsonl.samples[0].query_style == "FORMAL"
    assert loaded_jsonl.samples[0].query_length == "SHORT"


@pytest.mark.parametrize("eval_sample", samples)
def test_single_type_evaluation_dataset(eval_sample):
    single_turn_sample = SingleTurnSample(user_input="What is X", response="Y")
    multi_turn_sample = MultiTurnSample(
        user_input=[{"content": "What is X"}],
        response="Y",  # type: ignore (this type error is what we want to test)
    )

    with pytest.raises(ValueError) as exc_info:
        EvaluationDataset(samples=[single_turn_sample, multi_turn_sample])

    error_message = str(exc_info.value)

    assert (
        "Sample at index 1 is of type <class 'ragas.dataset_schema.MultiTurnSample'>"
        in error_message
    )
    assert "expected <class 'ragas.dataset_schema.SingleTurnSample'>" in error_message


def test_base_eval_sample():
    from ragas.dataset_schema import BaseSample

    class FakeSample(BaseSample):
        user_input: str
        response: str
        reference: t.Optional[str] = None

    fake_sample = FakeSample(user_input="What is X", response="Y")
    assert fake_sample.to_dict() == {"user_input": "What is X", "response": "Y"}
    assert fake_sample.get_features() == ["user_input", "response"]


def test_evaluation_dataset_iter():
    single_turn_sample = SingleTurnSample(user_input="What is X", response="Y")

    dataset = EvaluationDataset(samples=[single_turn_sample, single_turn_sample])

    for sample in dataset:
        assert sample == single_turn_sample


def test_evaluation_dataset_type():
    single_turn_sample = SingleTurnSample(user_input="What is X", response="Y")
    multi_turn_sample = MultiTurnSample(
        user_input=[{"content": "What is X"}],
        response="Y",  # type: ignore (this type error is what we want to test)
    )

    dataset = EvaluationDataset(samples=[single_turn_sample])
    assert dataset.get_sample_type() == SingleTurnSample

    dataset = EvaluationDataset(samples=[multi_turn_sample])
    assert dataset.get_sample_type() == MultiTurnSample


def test_multiturn_sample_validate_user_input_invalid_type():
    """Test that MultiTurnSample validation correctly rejects invalid message types."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MultiTurnSample(
            user_input=[
                HumanMessage(content="Hello"),
                "invalid_string",  # This should be rejected by Pydantic
            ]
        )


def test_multiturn_sample_validate_user_input_valid_types():
    """Test that MultiTurnSample validation accepts valid message types."""
    from ragas.messages import AIMessage

    sample = MultiTurnSample(
        user_input=[
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]
    )
    assert len(sample.user_input) == 2
    assert isinstance(sample.user_input[0], HumanMessage)
    assert isinstance(sample.user_input[1], AIMessage)
