"""Descriptive changes, repeat stability and bounded missing-outcome sensitivity.

Diagnostics explain what a mean and its interval can hide. They do not replace
the predeclared gate, introduce another hypothesis test, or impute missing data.
Sensitivity bounds concern the finite declared cohort under user-declared score
ranges; they are identification bounds, not statistical confidence intervals.
"""

import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence

from .alignment import CasePair
from .config import MetricSpec
from .statistics import finite_mean, quantile


@dataclass(frozen=True)
class Distribution:
    """Equal-case distribution on the eligible paired cohort."""

    cases: int
    mean: Optional[float]
    minimum: Optional[float]
    p05: Optional[float]
    median: Optional[float]
    p95: Optional[float]
    maximum: Optional[float]


@dataclass(frozen=True)
class ChangeDistribution:
    """Oriented per-case changes, with ties defined by the declared margin."""

    improved_cases: int
    regressed_cases: int
    within_margin_cases: int
    minimum: Optional[float]
    p05: Optional[float]
    median: Optional[float]
    p95: Optional[float]
    maximum: Optional[float]
    mean_absolute_change: Optional[float]


@dataclass(frozen=True)
class RepeatStability:
    """Within-case descriptive variability; repeats are not independent cases."""

    repeated_cases: int
    baseline_variable_cases: int
    candidate_variable_cases: int
    direction_flip_cases: int
    baseline_rms_standard_deviation: Optional[float]
    candidate_rms_standard_deviation: Optional[float]
    difference_rms_standard_deviation: Optional[float]
    maximum_baseline_range: Optional[float]
    maximum_candidate_range: Optional[float]
    maximum_difference_range: Optional[float]


@dataclass(frozen=True)
class MissingOutcomeBounds:
    """Possible full-cohort mean changes under the declared metric range.

    All unusable outcomes, including scores attached to unsuccessful statuses,
    are treated as unknown. Bounds use any remaining usable values in partial
    cases, so they can be informative even when no case is fully paired.
    """

    lower: Optional[float]
    upper: Optional[float]
    known_baseline_repeats: int
    known_candidate_repeats: int
    unknown_baseline_repeats: int
    unknown_candidate_repeats: int
    total_cases: int
    score_minimum: Optional[float]
    score_maximum: Optional[float]
    conclusion: str
    reason: Optional[str]


@dataclass(frozen=True)
class ComparisonDiagnostics:
    """Related descriptive evidence sharing the comparison's exact cohort."""

    baseline: Distribution
    candidate: Distribution
    changes: ChangeDistribution
    repeats: RepeatStability
    missing_outcomes: MissingOutcomeBounds


def _distribution(values: Sequence[float]) -> Distribution:
    if not values:
        return Distribution(0, None, None, None, None, None, None)
    ordered = tuple(sorted(values))
    return Distribution(
        len(ordered),
        finite_mean(ordered),
        ordered[0],
        quantile(ordered, 0.05),
        quantile(ordered, 0.5),
        quantile(ordered, 0.95),
        ordered[-1],
    )


def _changes(values: Sequence[float], margin: float) -> ChangeDistribution:
    if not values:
        return ChangeDistribution(0, 0, 0, None, None, None, None, None, None)
    ordered = tuple(sorted(values))
    improved = sum(value > margin for value in values)
    regressed = sum(value < -margin for value in values)
    return ChangeDistribution(
        improved_cases=improved,
        regressed_cases=regressed,
        within_margin_cases=len(values) - improved - regressed,
        minimum=ordered[0],
        p05=quantile(ordered, 0.05),
        median=quantile(ordered, 0.5),
        p95=quantile(ordered, 0.95),
        maximum=ordered[-1],
        mean_absolute_change=finite_mean(tuple(abs(value) for value in values)),
    )


def _variance(values: Sequence[float]) -> float:
    mean = finite_mean(values)
    # Scores are bounded to 1e150 by input validation, keeping squared
    # differences finite. Divide each summand before accumulating.
    return math.fsum(((value - mean) ** 2) / (len(values) - 1) for value in values)


def _stability(pairs: Sequence[CasePair], metric: MetricSpec) -> RepeatStability:
    repeated = [pair for pair in pairs if pair.eligible and pair.expected_repeats > 1]
    if not repeated:
        return RepeatStability(0, 0, 0, 0, None, None, None, None, None, None)
    baseline_variances = []
    candidate_variances = []
    difference_variances = []
    baseline_ranges = []
    candidate_ranges = []
    difference_ranges = []
    flips = 0
    for pair in repeated:
        differences = tuple(
            (candidate - baseline) * metric.orientation
            for baseline, candidate in zip(pair.baseline, pair.candidate)
        )
        baseline_variances.append(_variance(pair.baseline))
        candidate_variances.append(_variance(pair.candidate))
        difference_variances.append(_variance(differences))
        baseline_ranges.append(max(pair.baseline) - min(pair.baseline))
        candidate_ranges.append(max(pair.candidate) - min(pair.candidate))
        difference_ranges.append(max(differences) - min(differences))
        flips += min(differences) < -metric.margin and max(differences) > metric.margin
    return RepeatStability(
        repeated_cases=len(repeated),
        baseline_variable_cases=sum(value > 0 for value in baseline_ranges),
        candidate_variable_cases=sum(value > 0 for value in candidate_ranges),
        direction_flip_cases=flips,
        baseline_rms_standard_deviation=math.sqrt(finite_mean(baseline_variances)),
        candidate_rms_standard_deviation=math.sqrt(finite_mean(candidate_variances)),
        difference_rms_standard_deviation=math.sqrt(finite_mean(difference_variances)),
        maximum_baseline_range=max(baseline_ranges),
        maximum_candidate_range=max(candidate_ranges),
        maximum_difference_range=max(difference_ranges),
    )


def _completion_delta(
    pair: CasePair, baseline_fill: float, candidate_fill: float
) -> float:
    """Sum signed observations before division, preserving small differences."""
    count = pair.expected_repeats
    unknown_baseline = count - len(pair.baseline)
    unknown_candidate = count - len(pair.candidate)
    if count <= 0 or min(unknown_baseline, unknown_candidate) < 0:
        raise ValueError("invalid expected repeat count for sensitivity analysis")
    if pair.eligible:
        return finite_mean(
            tuple(
                after - before for before, after in zip(pair.baseline, pair.candidate)
            )
        )
    # Input magnitudes and repeat counts are bounded, so summing before division
    # cannot overflow. Avoid independently rounding two large completed means.
    terms = list(pair.candidate) + [-value for value in pair.baseline]
    terms.extend(candidate_fill for _ in range(unknown_candidate))
    terms.extend(-baseline_fill for _ in range(unknown_baseline))
    result = math.fsum(terms) / count
    if not math.isfinite(result):
        raise ValueError("missing-outcome sensitivity exceeded finite arithmetic")
    return result


def missing_outcome_bounds(
    pairs: Sequence[CasePair], metric: MetricSpec
) -> MissingOutcomeBounds:
    """Bound all completions without claiming that any fill value is observed."""
    expected = sum(pair.expected_repeats for pair in pairs)
    baseline_known = sum(len(pair.baseline) for pair in pairs)
    candidate_known = sum(len(pair.candidate) for pair in pairs)
    base = MissingOutcomeBounds(
        lower=None,
        upper=None,
        known_baseline_repeats=baseline_known,
        known_candidate_repeats=candidate_known,
        unknown_baseline_repeats=expected - baseline_known,
        unknown_candidate_repeats=expected - candidate_known,
        total_cases=len(pairs),
        score_minimum=metric.minimum,
        score_maximum=metric.maximum,
        conclusion="not_available",
        reason=None,
    )
    if not pairs:
        return replace(base, reason="empty_group")
    if metric.minimum is None or metric.maximum is None:
        return replace(base, reason="metric_range_not_declared")
    lower_changes = []
    upper_changes = []
    for pair in pairs:
        low = _completion_delta(pair, metric.maximum, metric.minimum)
        high = _completion_delta(pair, metric.minimum, metric.maximum)
        if metric.orientation < 0:
            low, high = -high, -low
        lower_changes.append(low)
        upper_changes.append(high)
    lower, upper = finite_mean(lower_changes), finite_mean(upper_changes)
    if upper < -metric.margin:
        conclusion = "regression_for_all_completions"
    elif lower >= -metric.margin:
        conclusion = "within_margin_for_all_completions"
    else:
        conclusion = "could_change_decision"
    return replace(base, lower=lower, upper=upper, conclusion=conclusion)


def describe(pairs: Sequence[CasePair], metric: MetricSpec) -> ComparisonDiagnostics:
    """Compute descriptive evidence without modifying eligibility or verdicts."""
    baseline = []
    candidate = []
    changes = []
    for pair in pairs:
        if not pair.eligible:
            continue
        baseline.append(finite_mean(pair.baseline))
        candidate.append(finite_mean(pair.candidate))
        changes.append(
            finite_mean(
                tuple(
                    (after - before) * metric.orientation
                    for before, after in zip(pair.baseline, pair.candidate)
                )
            )
        )
    return ComparisonDiagnostics(
        baseline=_distribution(baseline),
        candidate=_distribution(candidate),
        changes=_changes(changes, metric.margin),
        repeats=_stability(pairs, metric),
        missing_outcomes=missing_outcome_bounds(pairs, metric),
    )
