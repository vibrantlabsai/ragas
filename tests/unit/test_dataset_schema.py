import subprocess
import sys
import textwrap
import typing as t

import pytest

from ragas.dataset_schema import (
    EvaluationDataset,
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


def test_stratified_batches_without_drop_last_batch_uses_all_samples():
    annotated_samples = [create_sample_annotation(1) for _ in range(11)] + [
        create_sample_annotation(0) for _ in range(9)
    ]
    dataset = SingleMetricAnnotation(name="metric", samples=annotated_samples)

    batches = dataset.stratified_batches(
        batch_size=2, stratify_key="metric_output", drop_last_batch=False
    )

    batched_samples = [sample for batch in batches for sample in batch]
    assert len(batched_samples) == len(annotated_samples)
    assert len({id(sample) for sample in batched_samples}) == len(annotated_samples)
    assert len(batches) == 10
    assert all(len(batch) <= 2 for batch in batches)


def test_stratified_batches_with_drop_last_batch_terminates():
    """Regression test: with drop_last_batch=True, stratified_batches used to
    loop forever when the remaining samples could not fill another complete
    batch (num_batches was computed with ceil() instead of floor())."""
    script = textwrap.dedent(
        """
        from ragas.dataset_schema import SampleAnnotation, SingleMetricAnnotation

        def make_sample(metric_output):
            return SampleAnnotation(
                metric_input={"user_input": ""},
                metric_output=metric_output,
                prompts={},
                is_accepted=True,
            )

        samples = [make_sample(1) for _ in range(11)]
        samples += [make_sample(0) for _ in range(9)]
        dataset = SingleMetricAnnotation(name="metric", samples=samples)

        # 20 samples with batch_size=2 -> ten complete batches
        batches = dataset.stratified_batches(
            batch_size=2, stratify_key="metric_output", drop_last_batch=True
        )
        assert len(batches) == 10, [len(batch) for batch in batches]
        assert all(len(batch) == 2 for batch in batches)

        # 20 samples with batch_size=3 -> the partial last batch is dropped
        batches = dataset.stratified_batches(
            batch_size=3, stratify_key="metric_output", drop_last_batch=True
        )
        assert len(batches) == 6, [len(batch) for batch in batches]
        assert all(len(batch) == 3 for batch in batches)

        # no complete batch fits -> nothing to return
        batches = dataset.stratified_batches(
            batch_size=25, stratify_key="metric_output", drop_last_batch=True
        )
        assert batches == [], [len(batch) for batch in batches]
        """
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("stratified_batches(drop_last_batch=True) did not terminate")

    assert result.returncode == 0, result.stderr


def test_stratified_batches_with_fewer_samples_than_batch_size():
    annotated_samples = [create_sample_annotation(1) for _ in range(11)] + [
        create_sample_annotation(0) for _ in range(9)
    ]
    dataset = SingleMetricAnnotation(name="metric", samples=annotated_samples)

    batches = dataset.stratified_batches(
        batch_size=25, stratify_key="metric_output", drop_last_batch=False
    )

    assert len(batches) == 1
    assert len(batches[0]) == len(annotated_samples)


def test_stratified_batches_with_replace_fills_batches():
    annotated_samples = [create_sample_annotation(1) for _ in range(11)] + [
        create_sample_annotation(0) for _ in range(9)
    ]
    dataset = SingleMetricAnnotation(name="metric", samples=annotated_samples)

    batches = dataset.stratified_batches(
        batch_size=3, stratify_key="metric_output", replace=True
    )

    assert len(batches) == 7
    assert all(len(batch) == 3 for batch in batches)


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
