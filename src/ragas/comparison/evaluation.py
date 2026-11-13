"""Explicit identity adapters for results returned by ragas.evaluate()."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple

from .alignment import align
from .config import ComparisonLimits, ComparisonPlan, ExpectedCase, sequence, text
from .engine import compare
from .observations import capture, field_value, native_score
from .reports import ComparisonReport

if TYPE_CHECKING:
    from ragas.dataset_schema import EvaluationDataset, EvaluationResult


@dataclass(frozen=True)
class EvaluationKey:
    """Caller-assigned identity for one row of an EvaluationResult.

    Keys must travel with their original evaluation samples when rows are
    reordered. They are never inferred from a positional index or response text.
    """

    case_id: str
    repeat_id: Optional[str] = None

    def __post_init__(self) -> None:
        text(self.case_id, "case_id")
        if self.repeat_id is not None:
            text(self.repeat_id, "repeat_id")


def _keys(
    identities: Sequence[EvaluationKey],
    count: int,
    plan: ComparisonPlan,
    limits: ComparisonLimits,
) -> Tuple[EvaluationKey, ...]:
    keys = sequence(identities, "evaluation keys", limits.max_rows)
    if len(keys) != count:
        raise ValueError("evaluation keys and data length must match")
    seen = set()
    for key in keys:
        if not isinstance(key, EvaluationKey):
            raise ValueError("evaluation keys must contain EvaluationKey objects")
        if (key.repeat_id is not None) != (plan.repeat_id is not None):
            raise ValueError("evaluation repeat keys must match the plan")
        identity = (key.case_id, key.repeat_id)
        if identity in seen:
            raise ValueError("duplicate evaluation case/repeat key")
        seen.add(identity)
    return keys


def _identity_row(sample: Any, key: EvaluationKey, plan: ComparisonPlan) -> dict:
    row: dict[str, Any] = {plan.case_id: key.case_id}
    if plan.repeat_id is not None:
        row[plan.repeat_id] = key.repeat_id
    owned = set(row)
    for field in plan.selected_fields:
        if field in owned:
            raise ValueError(
                "evaluation identity fields cannot collide with sample fields"
            )
        present, value = field_value(sample, field)
        if not present:
            raise ValueError("selected case field is absent from the evaluation sample")
        row[field] = value
    if set(row) & {metric.score_column for metric in plan.metrics}:
        raise ValueError("evaluation metric columns cannot replace identity fields")
    return row


def evaluation_rows(
    result: "EvaluationResult",
    identities: Sequence[EvaluationKey],
    *,
    plan: ComparisonPlan,
    limits: Optional[ComparisonLimits] = None,
) -> Tuple[dict, ...]:
    """Extract paired-comparison rows without DataFrames or trace serialization.

    ``EvaluationResult`` stores metric scores and the corresponding dataset,
    but no stable comparison IDs. Supply keys in that result's original sample
    order. Different results may have different orders and different subsets.

    Selected identity/slice fields come from the original samples, not from
    generated responses or metric outputs. Non-scalar fields cannot be selected
    by the comparison plan. Without selected identity fields, key correctness is
    the caller's responsibility.

    Legacy results do not retain an authoritative execution status. A plan with
    a status field is rejected rather than manufacturing successful statuses.
    Missing/NaN metric scores remain missing evidence.
    """
    from ragas.dataset_schema import EvaluationResult

    if not isinstance(result, EvaluationResult):
        raise ValueError("result must be an EvaluationResult")
    limits = limits or ComparisonLimits()
    plan.check_limits(limits)
    if plan.status_field is not None:
        raise ValueError(
            "EvaluationResult does not provide authoritative execution status"
        )
    if result.dataset is None or len(result.dataset) != len(result.scores):
        raise ValueError("evaluation score and dataset length must match")
    keys = _keys(identities, len(result.scores), plan, limits)
    rows = []
    for index, key in enumerate(keys):
        row = _identity_row(result.dataset[index], key, plan)
        scores = result.scores[index]
        if not isinstance(scores, dict):
            raise ValueError("evaluation score rows must be dictionaries")
        for metric in plan.metrics:
            if metric.score_column in scores:
                row[metric.score_column] = native_score(scores[metric.score_column])
        rows.append(row)
    capture(rows, plan, limits)
    return tuple(rows)


def plan_for_evaluation_dataset(
    dataset: "EvaluationDataset",
    identities: Sequence[EvaluationKey],
    plan: ComparisonPlan,
    *,
    limits: Optional[ComparisonLimits] = None,
) -> ComparisonPlan:
    """Freeze the expected cohort before legacy evaluate() calls are made.

    The evaluation dataset may contain explicit repeated cases. Repeats must
    agree on every selected case attribute. Only the selected attributes and
    caller keys enter the manifest; responses, contexts and traces are omitted.
    """
    from ragas.dataset_schema import EvaluationDataset

    if not isinstance(dataset, EvaluationDataset):
        raise ValueError("dataset must be an EvaluationDataset")
    limits = limits or ComparisonLimits()
    plan.check_limits(limits)
    if plan.status_field is not None:
        raise ValueError("legacy evaluation plans cannot infer execution status")
    keys = _keys(identities, len(dataset), plan, limits)
    if not keys:
        raise ValueError("cannot plan an empty evaluation dataset")
    rows = tuple(
        _identity_row(dataset[index], key, plan) for index, key in enumerate(keys)
    )
    # Missing metric columns are intentional here: this is a planned cohort,
    # built before scoring. Alignment validates stable identity across repeats.
    snapshot = capture(rows, plan, limits)
    align(snapshot, snapshot, plan, limits)
    grouped: dict = {}
    for row in snapshot.rows:
        grouped.setdefault(row.case_id, []).append(row)
    expected = []
    for case_id, observations in sorted(grouped.items()):
        repeats = tuple(row.repeat_id for row in observations) if plan.repeat_id else ()
        expected.append(ExpectedCase(case_id, repeats, observations[0].attributes))
    prepared = replace(plan, expected_cases=tuple(expected))
    if (
        plan.expected_cases is not None
        and prepared.expected_cases != plan.expected_cases
    ):
        raise ValueError("evaluation dataset does not match the expected manifest")
    return prepared


def compare_evaluations(
    baseline: "EvaluationResult",
    candidate: "EvaluationResult",
    *,
    baseline_keys: Sequence[EvaluationKey],
    candidate_keys: Sequence[EvaluationKey],
    plan: ComparisonPlan,
    limits: Optional[ComparisonLimits] = None,
) -> ComparisonReport:
    """Compare two legacy results using explicit per-result identity mappings.

    Selected case fields detect changed samples under a reused key. The full
    expected denominator requires a prebuilt manifest; observed results alone
    cannot reveal cases missing from both versions.
    """
    before = evaluation_rows(baseline, baseline_keys, plan=plan, limits=limits)
    after = evaluation_rows(candidate, candidate_keys, plan=plan, limits=limits)
    return compare(before, after, plan=plan, limits=limits)
