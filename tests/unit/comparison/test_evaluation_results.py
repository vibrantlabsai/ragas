import math

import pytest

from ragas.callbacks import ChainRun
from ragas.comparison import ComparisonPlan, MetricSpec
from ragas.comparison.evaluation import (
    EvaluationKey,
    compare_evaluations,
    evaluation_rows,
    plan_for_evaluation_dataset,
)
from ragas.dataset_schema import EvaluationDataset, EvaluationResult, SingleTurnSample


def result(values, questions):
    return EvaluationResult(
        scores=[{"quality": value} for value in values],
        dataset=EvaluationDataset(
            samples=[SingleTurnSample(user_input=q) for q in questions]
        ),
        ragas_traces={
            "root": ChainRun(
                run_id="root",
                parent_run_id=None,
                name="evaluation",
                inputs={},
                metadata={},
            )
        },
    )


def test_legacy_result_adapter_pairs_explicit_ids_after_reordering():
    before = result([0.2, 0.8], ["question a", "question b"])
    after = result([0.9, 0.3], ["question b", "question a"])
    plan = ComparisonPlan(
        metrics=(MetricSpec("quality"),), case_fields=("user_input",), mode="observed"
    )
    report = compare_evaluations(
        before,
        after,
        baseline_keys=(EvaluationKey("a"), EvaluationKey("b")),
        candidate_keys=(EvaluationKey("b"), EvaluationKey("a")),
        plan=plan,
    )
    assert report.verdict == "pass"
    assert report.comparisons[0].estimate.raw_delta == pytest.approx(0.1)
    assert "question a" not in report.to_json()


def test_legacy_nan_remains_missing_and_manifest_exposes_absent_both():
    baseline = result([1], ["a"])
    candidate = result([math.nan], ["a"])
    dataset = EvaluationDataset(
        samples=[SingleTurnSample(user_input="a"), SingleTurnSample(user_input="b")]
    )
    plan = plan_for_evaluation_dataset(
        dataset,
        (EvaluationKey("a"), EvaluationKey("b")),
        ComparisonPlan(metrics=(MetricSpec("quality"),), mode="observed"),
    )
    report = compare_evaluations(
        baseline,
        candidate,
        baseline_keys=(EvaluationKey("a"),),
        candidate_keys=(EvaluationKey("a"),),
        plan=plan,
    )
    assert report.comparisons[0].coverage.total_cases == 2
    assert report.verdict == "inconclusive"


def test_result_shape_and_key_mismatches_reject():
    plan = ComparisonPlan(metrics=(MetricSpec("quality"),), mode="observed")
    baseline = result([1], ["a"])
    with pytest.raises(ValueError, match="length"):
        evaluation_rows(baseline, (), plan=plan)
    baseline.scores.append({"quality": 1})
    with pytest.raises(ValueError, match="length"):
        evaluation_rows(baseline, (EvaluationKey("a"), EvaluationKey("b")), plan=plan)


def test_result_adapter_does_not_invent_execution_status():
    plan = ComparisonPlan(metrics=(MetricSpec("quality"),), status_field="status")
    with pytest.raises(ValueError, match="status"):
        evaluation_rows(result([1], ["a"]), (EvaluationKey("a"),), plan=plan)


def test_repeated_evaluation_manifest_checks_same_case_identity():
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(user_input="a"),
            SingleTurnSample(user_input="different"),
        ]
    )
    plan = ComparisonPlan(
        metrics=(MetricSpec("quality"),), repeat_id="r", case_fields=("user_input",)
    )
    with pytest.raises(ValueError, match="attribute"):
        plan_for_evaluation_dataset(
            dataset, (EvaluationKey("a", "1"), EvaluationKey("a", "2")), plan
        )


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_native_numpy_metric_scalars_are_exported_as_python_numbers(dtype):
    import numpy as np

    value = getattr(np, dtype)(1)
    plan = ComparisonPlan(metrics=(MetricSpec("quality"),), mode="observed")
    rows = evaluation_rows(result([value], ["a"]), (EvaluationKey("a"),), plan=plan)
    assert type(rows[0]["quality"]) in (int, float)
    assert rows[0]["quality"] == 1


def test_numpy_nan_retains_missing_evidence_and_bool_is_not_a_score():
    import numpy as np

    plan = ComparisonPlan(metrics=(MetricSpec("quality"),), mode="observed")
    rows = evaluation_rows(
        result([np.float32("nan")], ["a"]), (EvaluationKey("a"),), plan=plan
    )
    assert math.isnan(rows[0]["quality"])
    with pytest.raises(ValueError, match="number"):
        evaluation_rows(
            result([np.bool_(True)], ["a"]), (EvaluationKey("a"),), plan=plan
        )


@pytest.mark.asyncio
async def test_actual_aevaluate_results_replay_with_explicit_reordered_keys():
    from ragas import aevaluate
    from ragas.metrics._string import ExactMatch

    before_data = EvaluationDataset(
        samples=[
            SingleTurnSample(user_input="a", response="wrong", reference="correct"),
            SingleTurnSample(user_input="b", response="correct", reference="correct"),
        ]
    )
    after_data = EvaluationDataset(
        samples=[
            SingleTurnSample(user_input="b", response="correct", reference="correct"),
            SingleTurnSample(user_input="a", response="correct", reference="correct"),
        ]
    )
    keys = (EvaluationKey("a"), EvaluationKey("b"))
    plan = plan_for_evaluation_dataset(
        before_data,
        keys,
        ComparisonPlan(
            metrics=(MetricSpec("exact_match", minimum=0, maximum=1),),
            case_fields=("user_input",),
            mode="observed",
        ),
    )
    before = await aevaluate(before_data, metrics=[ExactMatch()], show_progress=False)
    after = await aevaluate(after_data, metrics=[ExactMatch()], show_progress=False)
    report = compare_evaluations(
        before,
        after,
        baseline_keys=keys,
        candidate_keys=tuple(reversed(keys)),
        plan=plan,
    )
    assert report.verdict == "pass"
    assert report.comparisons[0].estimate.raw_delta == 0.5
