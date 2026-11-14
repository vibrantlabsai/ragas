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


# ---------------------------------------------------------------------------
# EvaluationResult coverage exposure (issue #3028)
# ---------------------------------------------------------------------------


def _make_dataset(n: int) -> EvaluationDataset:
    return EvaluationDataset(
        samples=[
            SingleTurnSample(user_input=f"q{i}", response=f"a{i}") for i in range(n)
        ]
    )


def test_evaluation_result_full_coverage_is_silent():
    import warnings

    import numpy as np

    from ragas.dataset_schema import EvaluationResult

    scores = [{"faithfulness": 1.0}, {"faithfulness": 0.0}]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = EvaluationResult(scores=scores, dataset=_make_dataset(2))

    assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == []
    assert result.nan_counts == {"faithfulness": 0}
    assert result.total_counts == {"faithfulness": 2}
    assert result.coverage == {"faithfulness": 1.0}
    assert result.summary() == {
        "faithfulness": {"mean": 0.5, "n_scored": 2, "n_total": 2, "coverage": 1.0}
    }
    assert repr(result) == "{'faithfulness': 0.5000}"
    # sanity: mean must not be moved by the new machinery
    assert result._repr_dict["faithfulness"] == 0.5
    _ = np  # silence unused import lint when not otherwise used


def test_evaluation_result_partial_coverage_warns_and_marks_repr():
    import inspect

    import numpy as np

    from ragas.dataset_schema import EvaluationResult

    scores = [{"faithfulness": 1.0}, {"faithfulness": float("nan")}]
    with pytest.warns(UserWarning, match="faithfulness: 1/2") as caught:
        constructor_line = inspect.currentframe().f_lineno + 1
        result = EvaluationResult(scores=scores, dataset=_make_dataset(2))

    assert len(caught) == 1
    assert caught[0].filename == __file__
    assert caught[0].lineno == constructor_line

    # backward compat: nanmean semantics preserved
    assert result._repr_dict["faithfulness"] == 1.0

    # coverage reported alongside the mean
    assert result.nan_counts == {"faithfulness": 1}
    assert result.total_counts == {"faithfulness": 2}
    assert result.coverage == {"faithfulness": 0.5}

    summary = result.summary()
    assert summary["faithfulness"]["n_scored"] == 1
    assert summary["faithfulness"]["n_total"] == 2
    assert summary["faithfulness"]["coverage"] == 0.5
    assert summary["faithfulness"]["mean"] == 1.0

    # repr surfaces the shrunken denominator
    assert "(1/2)" in repr(result)
    _ = np


def test_evaluation_result_all_nan_reports_zero_coverage():
    from ragas.dataset_schema import EvaluationResult

    scores = [{"faithfulness": float("nan")}, {"faithfulness": float("nan")}]
    with pytest.warns(UserWarning):
        result = EvaluationResult(scores=scores, dataset=_make_dataset(2))

    assert result.coverage == {"faithfulness": 0.0}
    summary = result.summary()
    assert summary["faithfulness"]["n_scored"] == 0
    assert summary["faithfulness"]["n_total"] == 2
    assert summary["faithfulness"]["coverage"] == 0.0
    # mean is nan when everything is nan
    assert summary["faithfulness"]["mean"] != summary["faithfulness"]["mean"]


def test_evaluation_result_mixed_metric_coverage_is_per_metric():
    from ragas.dataset_schema import EvaluationResult

    scores = [
        {"faithfulness": 1.0, "answer_relevancy": 0.8},
        {"faithfulness": float("nan"), "answer_relevancy": 0.6},
        {"faithfulness": 0.5, "answer_relevancy": 0.7},
    ]
    with pytest.warns(UserWarning, match="faithfulness: 2/3"):
        result = EvaluationResult(scores=scores, dataset=_make_dataset(3))

    assert result.coverage == {"faithfulness": 2 / 3, "answer_relevancy": 1.0}
    summary = result.summary()
    assert summary["faithfulness"] == {
        "mean": 0.75,
        "n_scored": 2,
        "n_total": 3,
        "coverage": 2 / 3,
    }
    assert summary["answer_relevancy"] == {
        "mean": pytest.approx(0.7),
        "n_scored": 3,
        "n_total": 3,
        "coverage": 1.0,
    }
    repr_str = repr(result)
    assert "'faithfulness': 0.7500 (2/3)" in repr_str
    assert "'answer_relevancy': 0.7000" in repr_str
    assert "(3/3)" not in repr_str  # full-coverage metrics stay clean


def test_evaluation_result_warn_on_missing_scores_can_be_disabled():
    import warnings

    from ragas.dataset_schema import EvaluationResult

    scores = [{"faithfulness": 1.0}, {"faithfulness": float("nan")}]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = EvaluationResult(
            scores=scores,
            dataset=_make_dataset(2),
            warn_on_missing_scores=False,
        )

    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    # coverage data still exposed even when the warning is silenced
    assert result.coverage == {"faithfulness": 0.5}
