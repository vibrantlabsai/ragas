"""Pure paired-comparison orchestration, with coverage-aware CI decisions."""

import hashlib
from collections import Counter
from dataclasses import asdict, replace
from typing import Any, Iterable, Optional, Sequence, Tuple

from .alignment import CasePair, Group, align
from .config import ComparisonLimits, ComparisonPlan, MetricSpec
from .diagnostics import describe
from .observations import canonical_bytes, capture
from .reports import CaseDetail, ComparisonReport, Coverage, Estimate, MetricComparison
from .statistics import bootstrap_interval, finite_mean, stable_seed


def _case_values(pair: CasePair, metric: MetricSpec) -> Tuple[float, float, float]:
    before = finite_mean(pair.baseline)
    after = finite_mean(pair.candidate)
    # Average paired differences directly: subtracting large aggregate means
    # first could erase a small but consistently observed change.
    change = finite_mean(
        tuple(after - before for before, after in zip(pair.baseline, pair.candidate))
    )
    return before, after, change * metric.orientation


def _coverage(pairs: Sequence[CasePair]) -> Coverage:
    reasons: Counter = Counter()
    for pair in pairs:
        reasons.update(pair.reasons)
    eligible = sum(pair.eligible for pair in pairs)
    return Coverage(
        total_cases=len(pairs),
        eligible_cases=eligible,
        excluded_cases=len(pairs) - eligible,
        expected_repeats=sum(pair.expected_repeats for pair in pairs),
        baseline_rows=sum(pair.baseline_rows for pair in pairs),
        candidate_rows=sum(pair.candidate_rows for pair in pairs),
        reason_counts=tuple(sorted(reasons.items())),
    )


def _bootstrap_reasons(
    differences: Tuple[float, ...], plan: ComparisonPlan
) -> Tuple[str, ...]:
    reasons = []
    if len(differences) < plan.min_cases:
        reasons.append("insufficient_cases")
    if differences and min(differences) == max(differences):
        reasons.append("degenerate_differences")
    family = plan.family_size if plan.adjustment == "bonferroni" else 1
    if plan.resamples * ((1 - plan.confidence) / (2 * family)) < 20:
        reasons.append("insufficient_tail_resamples")
    return tuple(reasons)


def _estimate(
    pairs: Sequence[CasePair],
    metric: MetricSpec,
    group: Group,
    plan: ComparisonPlan,
) -> Tuple[Estimate, Tuple[str, ...]]:
    eligible = [pair for pair in pairs if pair.eligible]
    if not eligible:
        return Estimate(0, 0, None, None, None, None), ("no_eligible_cases",)
    values = [_case_values(pair, metric) for pair in eligible]
    differences = tuple(value[2] for value in values)
    oriented_delta = finite_mean(differences)
    estimate = Estimate(
        cases=len(eligible),
        paired_repeats=sum(pair.expected_repeats for pair in eligible),
        baseline_mean=finite_mean(tuple(value[0] for value in values)),
        candidate_mean=finite_mean(tuple(value[1] for value in values)),
        raw_delta=oriented_delta * metric.orientation,
        oriented_delta=oriented_delta,
    )
    if plan.mode == "observed":
        return estimate, ()
    reasons = _bootstrap_reasons(differences, plan)
    family = plan.family_size if plan.adjustment == "bonferroni" else 1
    effective_confidence = 1 - (1 - plan.confidence) / family
    if reasons:
        return replace(estimate, effective_confidence=effective_confidence), reasons
    interval = bootstrap_interval(
        differences,
        resamples=plan.resamples,
        confidence=effective_confidence,
        seed=stable_seed(plan.seed, metric.name, group.name),
    )
    return replace(
        estimate,
        interval_lower=interval.lower,
        interval_upper=interval.upper,
        standard_error=interval.standard_error,
        effective_confidence=effective_confidence,
    ), ()


def _decision(
    estimate: Estimate,
    coverage: Coverage,
    metric: MetricSpec,
    plan: ComparisonPlan,
    inference_reasons: Tuple[str, ...],
) -> Tuple[str, Tuple[str, ...]]:
    reasons = list(inference_reasons)
    if coverage.fraction < plan.minimum_coverage:
        reasons.append("insufficient_coverage")
    if not coverage.total_cases:
        reasons.append("empty_group")
    if estimate.oriented_delta is None or inference_reasons:
        return "inconclusive", tuple(reasons)
    lower = (
        estimate.oriented_delta if plan.mode == "observed" else estimate.interval_lower
    )
    upper = (
        estimate.oriented_delta if plan.mode == "observed" else estimate.interval_upper
    )
    if lower is None or upper is None:
        return "inconclusive", tuple(reasons)
    if upper < -metric.margin:
        return "regress", tuple(reasons + ["eligible_cohort_regression"])
    if lower >= -metric.margin and not reasons:
        return "pass", ("within_tolerated_degradation",)
    if lower < -metric.margin:
        reasons.append("interval_crosses_margin")
    return "inconclusive", tuple(reasons)


def _details(
    pairs: Sequence[CasePair], metric: MetricSpec, limit: int
) -> Tuple[CaseDetail, ...]:
    details = []
    # IDs are already canonical: truncation must not depend on completion order.
    for pair in pairs[:limit]:
        before, after, change = (
            _case_values(pair, metric) if pair.eligible else (None, None, None)
        )
        details.append(
            CaseDetail(
                case_id=pair.case_id,
                eligible=pair.eligible,
                expected_repeats=pair.expected_repeats,
                baseline_rows=pair.baseline_rows,
                candidate_rows=pair.candidate_rows,
                baseline_mean=before,
                candidate_mean=after,
                oriented_delta=change,
                reasons=pair.reasons,
            )
        )
    return tuple(details)


def compare(
    baseline: Iterable[Any],
    candidate: Iterable[Any],
    *,
    plan: ComparisonPlan,
    limits: Optional[ComparisonLimits] = None,
) -> ComparisonReport:
    """Compare numeric evaluation observations paired by explicit identities.

    Both arguments may be Experiment objects, mappings, or Pydantic model
    iterables. No model, backend, save, or experiment runner is invoked. Cases
    absent on both sides are detectable only with ``plan.expected_cases``.

    Results with incomplete coverage describe the eligible cohort. A regression
    may conservatively block CI even when missing outcomes prevent conclusions
    about the full benchmark. Relaxing coverage is always explicit in the plan.
    """
    if not isinstance(plan, ComparisonPlan):
        raise ValueError("plan must be a ComparisonPlan")
    limits = limits or ComparisonLimits()
    plan.check_limits(limits)
    before = capture(baseline, plan, limits)
    after = capture(candidate, plan, limits)
    cohort = align(before, after, plan, limits)
    memberships = sum(len(group.case_ids) for group in cohort.groups) * len(
        plan.metrics
    )
    if memberships > limits.max_group_memberships:
        raise ValueError("comparison metric-group membership limit exceeded")
    jobs = []
    work = 0
    for metric, metric_pairs in zip(plan.metrics, cohort.pairs):
        indexed = {pair.case_id: pair for pair in metric_pairs}
        for group in cohort.groups:
            pairs = tuple(indexed[case_id] for case_id in group.case_ids)
            jobs.append((metric, group, pairs))
            if plan.mode == "bootstrap":
                work += sum(pair.eligible for pair in pairs) * plan.resamples
                if work > limits.max_bootstrap_work:
                    raise ValueError("comparison bootstrap work limit exceeded")

    comparisons = []
    remaining_details = plan.detail_limit
    for metric, group, pairs in jobs:
        coverage = _coverage(pairs)
        estimate, inference_reasons = _estimate(pairs, metric, group, plan)
        verdict, reasons = _decision(
            estimate, coverage, metric, plan, inference_reasons
        )
        details = _details(pairs, metric, remaining_details)
        remaining_details -= len(details)
        comparisons.append(
            MetricComparison(
                metric=metric.name,
                group=group.name,
                required=metric.required and group.required,
                direction=metric.direction,
                margin=metric.margin,
                verdict=verdict,
                reasons=reasons,
                coverage=coverage,
                estimate=estimate,
                conditional_on_eligible=bool(coverage.excluded_cases),
                details=details,
                omitted_details=len(pairs) - len(details),
                diagnostics=describe(pairs, metric),
            )
        )
    return ComparisonReport(
        plan_fingerprint=hashlib.sha256(canonical_bytes(asdict(plan))).hexdigest(),
        baseline_fingerprint=before.fingerprint,
        candidate_fingerprint=after.fingerprint,
        cohort_provenance=cohort.provenance,
        baseline_rows=cohort.baseline_rows,
        candidate_rows=cohort.candidate_rows,
        baseline_unsuccessful=cohort.baseline_unsuccessful,
        candidate_unsuccessful=cohort.candidate_unsuccessful,
        mode=plan.mode,
        confidence=plan.confidence,
        adjustment=plan.adjustment,
        family_size=plan.family_size,
        seed=plan.seed,
        resamples=plan.resamples,
        min_cases=plan.min_cases,
        minimum_coverage=plan.minimum_coverage,
        comparisons=tuple(comparisons),
        algorithm="observed-case-mean-v1"
        if plan.mode == "observed"
        else "paired-case-percentile-v1",
    )
