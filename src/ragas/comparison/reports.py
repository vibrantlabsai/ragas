"""Immutable comparison results; reports contain selected scores, not payloads."""

import json
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

from .diagnostics import ComparisonDiagnostics


@dataclass(frozen=True)
class Coverage:
    """Counts against the fixed group denominator, before outcome filtering."""

    total_cases: int
    eligible_cases: int
    excluded_cases: int
    expected_repeats: int
    baseline_rows: int
    candidate_rows: int
    reason_counts: Tuple[Tuple[str, int], ...]

    @property
    def fraction(self) -> float:
        """Empty cohorts have no usable coverage, never a vacuous 100%."""
        return self.eligible_cases / self.total_cases if self.total_cases else 0.0


@dataclass(frozen=True)
class Estimate:
    """Equal-case estimates; repeats are averaged within each case first."""

    cases: int
    paired_repeats: int
    baseline_mean: Optional[float]
    candidate_mean: Optional[float]
    raw_delta: Optional[float]
    oriented_delta: Optional[float]
    interval_lower: Optional[float] = None
    interval_upper: Optional[float] = None
    standard_error: Optional[float] = None
    effective_confidence: Optional[float] = None


@dataclass(frozen=True)
class CaseDetail:
    """One bounded diagnostic record, keyed by caller-supplied case ID."""

    case_id: str
    eligible: bool
    expected_repeats: int
    baseline_rows: int
    candidate_rows: int
    baseline_mean: Optional[float]
    candidate_mean: Optional[float]
    oriented_delta: Optional[float]
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class MetricComparison:
    """One metric/group decision with its denominator and uncertainty."""

    metric: str
    group: str
    required: bool
    direction: str
    margin: float
    verdict: str
    reasons: Tuple[str, ...]
    coverage: Coverage
    estimate: Estimate
    conditional_on_eligible: bool
    details: Tuple[CaseDetail, ...]
    omitted_details: int
    diagnostics: ComparisonDiagnostics


@dataclass(frozen=True)
class ComparisonReport:
    """A deterministic, auditable result for one predeclared comparison plan.

    Supplied-row counts do not recover failed attempts omitted by an experiment
    runner. An expected manifest is needed to count cases missing on both sides.
    """

    plan_fingerprint: str
    baseline_fingerprint: str
    candidate_fingerprint: str
    cohort_provenance: str
    baseline_rows: int
    candidate_rows: int
    baseline_unsuccessful: int
    candidate_unsuccessful: int
    mode: str
    confidence: float
    adjustment: str
    family_size: int
    seed: int
    resamples: int
    min_cases: int
    minimum_coverage: float
    comparisons: Tuple[MetricComparison, ...]
    algorithm: str = "paired-case-percentile-v1"
    format_version: int = 1

    @property
    def verdict(self) -> str:
        """Required comparisons gate independently; none are averaged away."""
        required = [result.verdict for result in self.comparisons if result.required]
        if "regress" in required:
            return "regress"
        if not required or "inconclusive" in required:
            return "inconclusive"
        return "pass"

    @property
    def exit_code(self) -> int:
        """CI status: pass=0, regression=1, inconclusive=2."""
        return {"pass": 0, "regress": 1, "inconclusive": 2}[self.verdict]

    def to_dict(self) -> dict:
        """Return a detached JSON-compatible mapping, with derived conclusions."""
        result = asdict(self)
        result["verdict"] = self.verdict
        result["exit_code"] = self.exit_code
        for comparison in result["comparisons"]:
            coverage = comparison["coverage"]
            coverage["fraction"] = (
                coverage["eligible_cases"] / coverage["total_cases"]
                if coverage["total_cases"]
                else 0.0
            )
        return result

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Serialize without nonstandard NaN/Infinity or an implicit timestamp."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            indent=indent,
        )
