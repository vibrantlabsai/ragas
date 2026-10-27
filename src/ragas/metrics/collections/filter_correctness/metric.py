"""FilterCorrectness — deterministic check for metadata-filtered retrieval leaks."""

from __future__ import annotations

import typing as t

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import inspect_retrieved_metadatas, normalize_allowed_values


class FilterCorrectness(BaseMetric):
    """
    Structural correctness of a retrieval metadata filter (non-LLM).

    Given per-chunk metadata and the filter that *should* have constrained
    retrieval (``filter_field`` + equality / ``$in``-style ``allowed_values``),
    score how many returned chunks violate that filter.

    Score semantics (higher is better):
    - ``1.0`` — every *checked* chunk respects the filter
    - ``1 - violations/checked`` — leak ratio among chunks that have the field
    - ``0.0`` with status ``empty_retrieval`` — no chunks returned (not proof of
      a correct filter)
    - ``0.0`` with status ``missing_metadata`` — chunks exist but none expose
      ``filter_field`` (not proof of a correct filter)

    Chunks missing ``filter_field`` are counted under ``missing_metadata`` and
    are **not** counted as violations. This metric only inspects returned
    chunks; it cannot prove that a filter could never leak under all queries.

    Example:
        >>> from ragas.metrics.collections import FilterCorrectness
        >>> metric = FilterCorrectness()
        >>> result = await metric.ascore(
        ...     retrieved_metadatas=[
        ...         {"tenant_id": "A"},
        ...         {"tenant_id": "B"},
        ...     ],
        ...     filter_field="tenant_id",
        ...     allowed_values="A",
        ... )
        >>> result.value  # 0.5
        0.5
    """

    def __init__(self, name: str = "filter_correctness", **kwargs):
        super().__init__(name=name, allowed_values=(0.0, 1.0), **kwargs)

    async def ascore(
        self,
        retrieved_metadatas: t.List[t.Dict[str, t.Any]],
        filter_field: str,
        allowed_values: t.Any,
    ) -> MetricResult:
        """
        Score filter adherence for one retrieval result.

        Args:
            retrieved_metadatas: Metadata dict per retrieved chunk (order preserved).
            filter_field: Metadata key the where-clause should have constrained.
            allowed_values: Allowed scalar (equality) or sequence (``$in`` / membership).

        Returns:
            MetricResult with correctness in ``[0, 1]`` and structured traces.
        """
        if not isinstance(retrieved_metadatas, list):
            raise ValueError("retrieved_metadatas must be a list of metadata dicts")
        if not isinstance(filter_field, str) or not filter_field.strip():
            raise ValueError("filter_field must be a non-empty string")

        allowed = normalize_allowed_values(allowed_values)

        if len(retrieved_metadatas) == 0:
            return self._result(
                value=0.0,
                reason=(
                    "empty retrieval: no chunks to check "
                    "(not evidence of a correct filter)"
                ),
                status="empty_retrieval",
                checked=0,
                violations=0,
                missing_metadata=0,
                offending_values=[],
                filter_field=filter_field,
            )

        stats = inspect_retrieved_metadatas(
            retrieved_metadatas, filter_field=filter_field, allowed=allowed
        )
        checked = stats["checked"]
        violations = stats["violations"]
        missing = stats["missing_metadata"]
        offending = stats["offending_values"]

        if checked == 0:
            return self._result(
                value=0.0,
                reason=(
                    f"missing_metadata: none of {len(retrieved_metadatas)} retrieved "
                    f"chunks expose field {filter_field!r} "
                    f"(not evidence of a correct filter)"
                ),
                status="missing_metadata",
                checked=0,
                violations=0,
                missing_metadata=missing,
                offending_values=[],
                filter_field=filter_field,
            )

        correctness = round((checked - violations) / checked, 4)
        if violations == 0:
            reason = (
                f"checked={checked} violations=0 missing_metadata={missing}; "
                f"all checked chunks respect {filter_field!r}"
            )
        else:
            reason = (
                f"checked={checked} violations={violations} "
                f"missing_metadata={missing}; "
                f"offending {filter_field} values={offending!r}"
            )

        return self._result(
            value=correctness,
            reason=reason,
            status="scored",
            checked=checked,
            violations=violations,
            missing_metadata=missing,
            offending_values=offending,
            filter_field=filter_field,
        )

    @staticmethod
    def _result(
        *,
        value: float,
        reason: str,
        status: str,
        checked: int,
        violations: int,
        missing_metadata: int,
        offending_values: t.List[t.Any],
        filter_field: str,
    ) -> MetricResult:
        return MetricResult(
            value=value,
            reason=reason,
            traces={
                "output": {
                    "status": status,
                    "checked": checked,
                    "violations": violations,
                    "missing_metadata": missing_metadata,
                    "offending_values": offending_values,
                    "filter_field": filter_field,
                }
            },
        )
