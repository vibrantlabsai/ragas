"""Bounded, immutable snapshots of only the columns needed by a comparison."""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

from pydantic import BaseModel

from .config import ComparisonLimits, ComparisonPlan, Scalar, number, scalar_key, text


@dataclass(frozen=True)
class Score:
    """A numeric observation or an explicitly identified absence."""

    value: Optional[float]
    reason: Optional[str] = None


@dataclass(frozen=True)
class Observation:
    """One uniquely keyed run; all containers are immutable copies."""

    case_id: str
    repeat_id: Optional[str]
    attributes: Tuple[Tuple[str, Scalar], ...]
    scores: Tuple[Score, ...]
    successful: bool
    status: str


@dataclass(frozen=True)
class RunSnapshot:
    """Canonical selected evidence, with a digest independent of row order."""

    rows: Tuple[Observation, ...]
    fingerprint: str
    selected_bytes: int


def canonical_bytes(value: Any) -> bytes:
    """Encode internal validated data deterministically, without NaN extensions."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def field_value(row: Any, field: str) -> Tuple[bool, Any]:
    """Read a selected field while distinguishing absent values from null.

    Avoid serializing an entire model: unrelated output fields can be large or
    have custom serialization behavior that a numeric comparison does not need.
    """
    if isinstance(row, Mapping):
        return (True, row[field]) if field in row else (False, None)
    if isinstance(row, BaseModel):
        if field in type(row).model_fields or field in (row.model_extra or {}):
            return True, getattr(row, field)
        return False, None
    raise ValueError("comparison rows must be mappings or Pydantic models")


def native_score(value: Any) -> Any:
    """Normalize NumPy scalars returned by native Ragas metric implementations.

    Restrict this conversion to adapter boundaries. Arrays, booleans, strings
    and arbitrary objects with a float conversion are not numeric observations.
    NumPy's item conversion preserves integer precision for later validation;
    extended-precision scalars that cannot become Python floats remain rejected.
    """
    import numpy as np

    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _score(row: Any, spec: Any) -> Score:
    present, raw = field_value(row, spec.score_column)
    if not present:
        return Score(None, "missing")
    if raw is None:
        return Score(None, "null")
    if type(raw) is float and math.isnan(raw):
        return Score(None, "nan")
    value = number(raw, "metric score")
    if spec.minimum is not None and value < spec.minimum:
        raise ValueError("metric score is below its configured minimum")
    if spec.maximum is not None and value > spec.maximum:
        raise ValueError("metric score exceeds its configured maximum")
    return Score(value)


def _status(row: Any, plan: ComparisonPlan) -> Tuple[bool, str]:
    if plan.status_field is None:
        return True, "not_configured"
    present, value = field_value(row, plan.status_field)
    if not present:
        return False, "missing"
    key = scalar_key(value)
    successful = key in {scalar_key(v) for v in plan.success_values}
    return successful, "success" if successful else "unsuccessful"


def observation_data(row: Observation) -> dict:
    """Selected evidence representation used for integrity fingerprints."""
    return {
        "case_id": row.case_id,
        "repeat_id": row.repeat_id,
        "attributes": [(name, scalar_key(value)) for name, value in row.attributes],
        "scores": [(score.value, score.reason) for score in row.scores],
        "successful": row.successful,
        "status": row.status,
    }


def capture(
    rows: Iterable[Any],
    plan: ComparisonPlan,
    limits: Optional[ComparisonLimits] = None,
) -> RunSnapshot:
    """Copy and validate selected evidence from an iterable without side effects.

    Duplicate keys, invalid numeric values and missing identity fields are input
    errors. Missing scores and unsuccessful statuses remain recorded evidence.
    """
    limits = limits or ComparisonLimits()
    plan.check_limits(limits)
    return _capture(rows, plan, limits)


def _capture(
    rows: Iterable[Any], plan: ComparisonPlan, limits: ComparisonLimits
) -> RunSnapshot:
    """Capture rows after the immutable plan has already passed its limits.

    This private entry point lets collection adapters reuse a validated plan
    without traversing its complete expected manifest for every individual job.
    All row, score, identity and byte limits are still enforced on each call.
    """
    captured = []
    keys = set()
    case_counts: dict = {}
    total_bytes = 0
    for index, row in enumerate(rows):
        if index >= limits.max_rows:
            raise ValueError("comparison rows limit exceeded")
        _, case_id = field_value(row, plan.case_id)
        text(case_id, "case_id")
        repeat_id = None
        if plan.repeat_id is not None:
            _, repeat_id = field_value(row, plan.repeat_id)
            text(repeat_id, "repeat_id")
        key = (case_id, repeat_id)
        if key in keys:
            raise ValueError("duplicate case_id/repeat_id observation")
        keys.add(key)
        case_counts[case_id] = case_counts.get(case_id, 0) + 1
        if len(case_counts) > limits.max_cases:
            raise ValueError("comparison cases limit exceeded")
        if case_counts[case_id] > limits.max_repeats:
            raise ValueError("comparison repeats limit exceeded")
        attributes = []
        for field in plan.selected_fields:
            present, value = field_value(row, field)
            if not present:
                raise ValueError("selected case field is missing")
            scalar_key(value)
            attributes.append((field, value))
        strings = [case_id, repeat_id] + [v for _, v in attributes]
        for value in strings:
            if (
                isinstance(value, str)
                and len(value.encode("utf-8")) > limits.max_string_bytes
            ):
                raise ValueError("comparison string limit exceeded")
        successful, status = _status(row, plan)
        observation = Observation(
            case_id,
            repeat_id,
            tuple(attributes),
            tuple(_score(row, metric) for metric in plan.metrics),
            successful,
            status,
        )
        total_bytes += len(canonical_bytes(observation_data(observation)))
        if total_bytes > limits.max_selected_bytes:
            raise ValueError("selected evidence byte limit exceeded")
        captured.append(observation)
    captured.sort(key=lambda row: (row.case_id, row.repeat_id or ""))
    digest = hashlib.sha256()
    for observation in captured:
        digest.update(canonical_bytes(observation_data(observation)))
        digest.update(b"\n")
    return RunSnapshot(tuple(captured), digest.hexdigest(), total_bytes)
