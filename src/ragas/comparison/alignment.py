"""Freeze cohorts before filtering outcomes, then pair exact observation keys."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .config import ComparisonLimits, ComparisonPlan, Scalar, scalar_key
from .observations import Observation, RunSnapshot


@dataclass(frozen=True)
class Group:
    """A fixed denominator independent of metric values or execution success."""

    name: str
    case_ids: Tuple[str, ...]
    required: bool


@dataclass(frozen=True)
class CasePair:
    """Paired repeats for one metric; incomplete cases are never partly scored."""

    case_id: str
    baseline: Tuple[float, ...]
    candidate: Tuple[float, ...]
    expected_repeats: int
    baseline_rows: int
    candidate_rows: int
    reasons: Tuple[str, ...]

    @property
    def eligible(self) -> bool:
        """All expected repeats on both sides must be usable."""
        return not self.reasons and self.expected_repeats > 0


@dataclass(frozen=True)
class AlignedCohort:
    """Metric-specific complete pairs over one immutable case population."""

    groups: Tuple[Group, ...]
    pairs: Tuple[Tuple[CasePair, ...], ...]
    provenance: str
    baseline_rows: int
    candidate_rows: int
    baseline_unsuccessful: int
    candidate_unsuccessful: int


def _attributes_key(attributes: Tuple[Tuple[str, Scalar], ...]) -> tuple:
    return tuple((name, scalar_key(value)) for name, value in attributes)


def _index(snapshot: RunSnapshot) -> Dict[str, Dict[Optional[str], Observation]]:
    result: Dict[str, Dict[Optional[str], Observation]] = {}
    for row in snapshot.rows:
        result.setdefault(row.case_id, {})[row.repeat_id] = row
    return result


def _case_pair(
    case_id: str,
    repeats: Tuple[Optional[str], ...],
    baseline: Dict[Optional[str], Observation],
    candidate: Dict[Optional[str], Observation],
    metric_index: int,
) -> CasePair:
    reasons = set()
    values = ([], [])
    for side_index, (label, rows) in enumerate(
        (("baseline", baseline), ("candidate", candidate))
    ):
        if not rows:
            reasons.add(f"{label}_absent")
            continue
        for repeat in repeats:
            row = rows.get(repeat)
            if row is None:
                reasons.add(f"{label}_repeat_absent")
                continue
            if not row.successful:
                reasons.add(f"{label}_unsuccessful")
            score = row.scores[metric_index]
            if score.value is None:
                reasons.add(f"{label}_{score.reason}")
            elif row.successful:
                values[side_index].append(score.value)
    return CasePair(
        case_id=case_id,
        baseline=tuple(values[0]),
        candidate=tuple(values[1]),
        expected_repeats=len(repeats),
        baseline_rows=len(baseline),
        candidate_rows=len(candidate),
        reasons=tuple(sorted(reasons)),
    )


def align(
    baseline: RunSnapshot,
    candidate: RunSnapshot,
    plan: ComparisonPlan,
    limits: Optional[ComparisonLimits] = None,
) -> AlignedCohort:
    """Align captured observations without choosing a favorable subset.

    Case membership and attributes are resolved before checking any score. The
    manifest, when supplied, is authoritative for both case and repeat coverage.
    Otherwise only the union of supplied case/repeat IDs can be known.
    """
    limits = limits or ComparisonLimits()
    plan.check_limits(limits)
    before, after = _index(baseline), _index(candidate)
    supplied = set(before) | set(after)
    manifest = {case.case_id: case for case in plan.expected_cases or ()}
    if plan.expected_cases is not None:
        if supplied - set(manifest):
            raise ValueError("unexpected case_id outside expected manifest")
        case_ids = tuple(sorted(manifest))
        provenance = "expected_manifest"
    else:
        case_ids = tuple(sorted(supplied))
        provenance = "observed_union"
    if len(case_ids) > limits.max_cases:
        raise ValueError("comparison union cases limit exceeded")
    if len(case_ids) * len(plan.metrics) > limits.max_alignment_cells:
        raise ValueError("comparison alignment cell limit exceeded")
    if len(case_ids) * (1 + len(plan.slices)) > limits.max_group_memberships:
        raise ValueError("comparison group membership limit exceeded")

    attributes: Dict[str, Tuple[Tuple[str, Scalar], ...]] = {}
    repeats_by_case: Dict[str, Tuple[Optional[str], ...]] = {}
    for case_id in case_ids:
        observations = tuple(before.get(case_id, {}).values()) + tuple(
            after.get(case_id, {}).values()
        )
        declared = manifest.get(case_id)
        expected_attributes = (
            declared.attributes if declared else observations[0].attributes
        )
        expected_key = _attributes_key(expected_attributes)
        for row in observations:
            if _attributes_key(row.attributes) != expected_key:
                raise ValueError(
                    "case attributes differ across observations or manifest"
                )
            if len(row.scores) != len(plan.metrics):
                raise ValueError("snapshot metric count does not match plan")
        attributes[case_id] = expected_attributes
        supplied_repeats = set(before.get(case_id, {})) | set(after.get(case_id, {}))
        if declared is not None:
            repeats: Tuple[Optional[str], ...] = (
                declared.repeats if plan.repeat_id else (None,)
            )
            if supplied_repeats - set(repeats):
                raise ValueError("unexpected repeat_id outside expected manifest")
        else:
            repeats = tuple(sorted(supplied_repeats, key=lambda value: value or ""))
        if len(repeats) > limits.max_repeats:
            raise ValueError("comparison union repeats limit exceeded")
        repeats_by_case[case_id] = repeats

    groups = [Group("overall", case_ids, True)]
    for slice_spec in plan.slices:
        allowed = {scalar_key(value) for value in slice_spec.values}
        members = tuple(
            case_id
            for case_id in case_ids
            if scalar_key(dict(attributes[case_id])[slice_spec.field]) in allowed
        )
        groups.append(Group(slice_spec.name, members, slice_spec.required))

    paired_metrics = []
    for index, _ in enumerate(plan.metrics):
        paired_metrics.append(
            tuple(
                _case_pair(
                    case_id,
                    repeats_by_case[case_id],
                    before.get(case_id, {}),
                    after.get(case_id, {}),
                    index,
                )
                for case_id in case_ids
            )
        )
    return AlignedCohort(
        groups=tuple(groups),
        pairs=tuple(paired_metrics),
        provenance=provenance,
        baseline_rows=len(baseline.rows),
        candidate_rows=len(candidate.rows),
        baseline_unsuccessful=sum(not row.successful for row in baseline.rows),
        candidate_unsuccessful=sum(not row.successful for row in candidate.rows),
    )
