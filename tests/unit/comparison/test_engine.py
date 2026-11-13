import pytest

from ragas.comparison import (
    ComparisonLimits,
    ComparisonPlan,
    ExpectedCase,
    MetricSpec,
    SliceSpec,
    compare,
)


def rows(values):
    return [{"case_id": str(i), "score": value} for i, value in enumerate(values)]


def plan(**kwargs):
    return ComparisonPlan(metrics=(MetricSpec("score"),), **kwargs)


def test_observed_boundary_and_lower_is_better():
    p = ComparisonPlan(
        metrics=(MetricSpec("score", direction="lower", margin=0.25),), mode="observed"
    )
    report = compare(rows([1]), rows([1.25]), plan=p)
    assert report.verdict == "pass"
    assert report.exit_code == 0
    assert report.comparisons[0].estimate.oriented_delta == -0.25
    assert compare(rows([1]), rows([1.5]), plan=p).verdict == "regress"


def test_disappeared_failure_cannot_improve_default_gate():
    report = compare(rows([0.1, 0.9]), rows([0.2]), plan=plan(mode="observed"))
    assert report.verdict == "inconclusive"
    assert report.exit_code == 2
    result = report.comparisons[0]
    assert result.coverage.total_cases == 2
    assert result.coverage.eligible_cases == 1
    assert result.conditional_on_eligible
    assert result.estimate.raw_delta == pytest.approx(0.1)
    assert result.reasons == ("insufficient_coverage",)


def test_metric_group_cross_product_has_a_joint_work_ceiling():
    p = ComparisonPlan(
        metrics=(
            MetricSpec("first", column="score"),
            MetricSpec("second", column="score"),
        ),
        slices=(SliceSpec("en", "language", ("en",)),),
        mode="observed",
    )
    data = [{"case_id": "a", "score": 1, "language": "en"}]
    with pytest.raises(ValueError, match="metric-group membership"):
        compare(data, data, plan=p, limits=ComparisonLimits(max_group_memberships=3))


def test_explicit_relaxed_coverage_can_pass_but_reports_conditional_estimand():
    report = compare(
        rows([0, 1]), rows([0.5]), plan=plan(mode="observed", minimum_coverage=0.5)
    )
    assert report.verdict == "pass"
    assert report.comparisons[0].conditional_on_eligible


def test_manifest_exposes_missing_both_versions():
    report = compare(
        rows([1]),
        rows([1]),
        plan=plan(
            mode="observed", expected_cases=(ExpectedCase("0"), ExpectedCase("1"))
        ),
    )
    assert report.cohort_provenance == "expected_manifest"
    assert report.comparisons[0].coverage.excluded_cases == 1
    assert report.verdict == "inconclusive"


def test_repeat_count_does_not_reweight_cases():
    before = [{"case_id": "a", "repeat": str(i), "score": 0} for i in range(9)]
    after = [{"case_id": "a", "repeat": str(i), "score": 1} for i in range(9)]
    before.append({"case_id": "b", "repeat": "0", "score": 1})
    after.append({"case_id": "b", "repeat": "0", "score": 0})
    result = compare(
        before, after, plan=plan(repeat_id="repeat", mode="observed")
    ).comparisons[0]
    assert result.estimate.baseline_mean == 0.5
    assert result.estimate.candidate_mean == 0.5
    assert result.estimate.raw_delta == 0
    assert result.estimate.cases == 2
    assert result.estimate.paired_repeats == 10


def test_bootstrap_detects_clear_regression_and_is_order_invariant():
    before, after = rows([1] * 30), rows([0.2 + i / 100 for i in range(30)])
    p = plan(resamples=1000, confidence=0.9)
    first = compare(before, after, plan=p)
    second = compare(list(reversed(before)), list(reversed(after)), plan=p)
    assert first.to_dict() == second.to_dict()
    assert first.verdict == "regress"
    assert first.exit_code == 1
    assert first.comparisons[0].estimate.interval_upper < 0


def test_empty_and_degenerate_evidence_are_not_passes():
    assert (
        compare([], [], plan=plan(mode="observed", minimum_coverage=0)).verdict
        == "inconclusive"
    )
    report = compare(rows([1] * 30), rows([1] * 30), plan=plan())
    assert report.verdict == "inconclusive"
    assert "degenerate_differences" in report.comparisons[0].reasons


def test_fixed_family_includes_empty_optional_slice():
    before = [{"case_id": str(i), "score": 0, "lang": "en"} for i in range(30)]
    after = [{"case_id": str(i), "score": i / 30, "lang": "en"} for i in range(30)]
    p = plan(
        resamples=1000,
        confidence=0.95,
        slices=(SliceSpec("de", "lang", ("de",), required=False),),
    )
    report = compare(before, after, plan=p)
    assert report.family_size == 2
    assert "insufficient_tail_resamples" in report.comparisons[0].reasons
    assert report.comparisons[1].coverage.total_cases == 0


def test_work_budget_checked_before_any_bootstrap(monkeypatch):
    import ragas.comparison.engine as engine

    monkeypatch.setattr(
        engine,
        "bootstrap_interval",
        lambda *a, **k: pytest.fail("resampled before validating budget"),
    )
    with pytest.raises(ValueError, match="work"):
        compare(
            rows([0] * 30),
            rows([i / 30 for i in range(30)]),
            plan=plan(),
            limits=ComparisonLimits(max_bootstrap_work=2),
        )


def test_detail_truncation_does_not_skip_verdicts():
    report = compare(
        rows([1] * 4), rows([0] * 4), plan=plan(mode="observed", detail_limit=1)
    )
    assert report.verdict == "regress"
    assert len(report.comparisons[0].details) == 1
    assert report.comparisons[0].omitted_details == 3


def test_report_contains_no_unselected_output():
    row = {"case_id": "a", "score": 1, "output": "private customer text"}
    report = compare([row], [row], plan=plan(mode="observed"))
    assert "private customer text" not in report.to_json()
