import pytest

from ragas.comparison import ComparisonPlan, ExpectedCase, MetricSpec, compare


def metric(**kwargs):
    return MetricSpec("score", minimum=0, maximum=1, **kwargs)


def row(case, score, **kwargs):
    return {"case_id": case, "score": score, **kwargs}


def test_attrition_bounds_expose_a_reversal_hidden_by_complete_case_mean():
    plan = ComparisonPlan(
        metrics=(metric(),),
        mode="observed",
        minimum_coverage=0.5,
        expected_cases=(ExpectedCase("a"), ExpectedCase("b")),
    )
    report = compare([row("a", 0.4), row("b", 1)], [row("a", 0.6)], plan=plan)
    bounds = report.comparisons[0].diagnostics.missing_outcomes
    assert report.verdict == "pass"  # explicitly conditional threshold
    assert bounds.lower == pytest.approx(-0.4)
    assert bounds.upper == pytest.approx(0.1)
    assert bounds.conclusion == "could_change_decision"
    assert bounds.unknown_candidate_repeats == 1


def test_failed_present_scores_are_unknown_for_sensitivity_bounds():
    plan = ComparisonPlan(metrics=(metric(),), mode="observed", status_field="state")
    report = compare(
        [row("a", 1, state="success")], [row("a", 1, state="error")], plan=plan
    )
    bounds = report.comparisons[0].diagnostics.missing_outcomes
    assert bounds.lower == -1
    assert bounds.upper == 0
    assert bounds.unknown_candidate_repeats == 1


def test_partial_repeat_bounds_use_all_known_usable_values():
    plan = ComparisonPlan(metrics=(metric(),), mode="observed", repeat_id="r")
    before = [row("a", 0.5, r="1"), row("a", 0.5, r="2")]
    after = [row("a", 0.8, r="1")]
    result = compare(before, after, plan=plan).comparisons[0]
    assert result.coverage.eligible_cases == 0
    assert result.diagnostics.missing_outcomes.lower == pytest.approx(-0.1)
    assert result.diagnostics.missing_outcomes.upper == pytest.approx(0.4)


def test_lower_is_better_bounds_are_oriented_and_unbounded_metrics_decline():
    plan = ComparisonPlan(metrics=(metric(direction="lower"),), mode="observed")
    bounds = (
        compare([row("a", 0.75)], [row("a", None)], plan=plan)
        .comparisons[0]
        .diagnostics.missing_outcomes
    )
    assert bounds.lower == -0.25
    assert bounds.upper == 0.75
    plain = ComparisonPlan(metrics=(MetricSpec("score"),), mode="observed")
    bounds = (
        compare([row("a", 1)], [row("a", None)], plan=plain)
        .comparisons[0]
        .diagnostics.missing_outcomes
    )
    assert bounds.lower is None
    assert bounds.reason == "metric_range_not_declared"


def test_repeat_direction_flips_are_not_hidden_by_a_positive_mean():
    plan = ComparisonPlan(metrics=(metric(),), repeat_id="r", mode="observed")
    before = [row("a", 0.5, r="1"), row("a", 0.5, r="2")]
    after = [row("a", 0.2, r="1"), row("a", 1, r="2")]
    diagnostic = compare(before, after, plan=plan).comparisons[0].diagnostics
    assert diagnostic.repeats.repeated_cases == 1
    assert diagnostic.repeats.direction_flip_cases == 1
    assert diagnostic.repeats.candidate_variable_cases == 1
    assert diagnostic.changes.improved_cases == 1
    assert diagnostic.changes.mean_absolute_change == pytest.approx(0.1)


def test_diagnostic_distributions_weight_cases_not_repeats():
    plan = ComparisonPlan(metrics=(metric(),), repeat_id="r", mode="observed")
    before = [row("a", 0, r=str(i)) for i in range(9)] + [row("b", 1, r="0")]
    after = [row("a", 1, r=str(i)) for i in range(9)] + [row("b", 0, r="0")]
    result = compare(before, after, plan=plan).comparisons[0].diagnostics
    assert result.changes.improved_cases == result.changes.regressed_cases == 1
    assert result.changes.median == 0
    assert result.baseline.median == result.candidate.median == 0.5


def test_complete_evidence_bounds_collapse_to_observed_mean():
    plan = ComparisonPlan(metrics=(metric(),), mode="observed")
    result = compare([row("a", 0.5)], [row("a", 0)], plan=plan).comparisons[0]
    bounds = result.diagnostics.missing_outcomes
    assert bounds.lower == bounds.upper == -0.5
    assert bounds.conclusion == "regression_for_all_completions"


def test_empty_diagnostics_have_no_fake_zeros_or_passing_bounds():
    result = compare(
        [], [], plan=ComparisonPlan(metrics=(metric(),), mode="observed")
    ).comparisons[0]
    assert result.diagnostics.baseline.mean is None
    assert result.diagnostics.missing_outcomes.conclusion == "not_available"


def test_large_scores_do_not_erase_observed_degradation_in_completion_bounds():
    plan = ComparisonPlan(
        metrics=(MetricSpec("score", minimum=0, maximum=1e16 + 2),),
        repeat_id="r",
        mode="observed",
    )
    before = [row("a", 1e16 + 2, r="1"), row("a", 1e16, r="2")]
    after = [row("a", 1e16, r="1"), row("a", 1e16, r="2")]
    result = compare(before, after, plan=plan).comparisons[0]
    assert result.verdict == "regress"
    bounds = result.diagnostics.missing_outcomes
    assert bounds.lower == bounds.upper == result.estimate.oriented_delta == -1
    assert bounds.conclusion == "regression_for_all_completions"
