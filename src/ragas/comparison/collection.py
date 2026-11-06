"""Plan denominators before evaluation and retain unsuccessful observations.

These helpers compose with Ragas Executor or an existing experiment runner.
They do not schedule, retry, checkpoint, or claim exactly-once external effects.
Execution inputs stay with the caller; only selected identity fields enter the
manifest and only numeric metric values enter returned observations.
"""

import asyncio
import inspect
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from pydantic import BaseModel

from .config import (
    ComparisonLimits,
    ComparisonPlan,
    ExpectedCase,
    number,
    scalar_key,
    sequence,
    text,
)
from .observations import _capture, canonical_bytes, field_value, native_score

STATUS_FIELD = "__ragas_comparison_status"


def _collection_fields(plan: ComparisonPlan) -> Tuple[str, ...]:
    owned = (plan.case_id,) + plan.selected_fields
    if plan.repeat_id is not None:
        owned += (plan.repeat_id,)
    if plan.status_field is not None:
        owned += (plan.status_field,)
    if len(set(owned)) != len(owned):
        raise ValueError(
            "collection identity, repeat and status fields must be distinct"
        )
    if set(owned) & {metric.score_column for metric in plan.metrics}:
        raise ValueError(
            "collection metric columns cannot replace identity or status fields"
        )
    if tuple(scalar_key(value) for value in plan.success_values) != (
        ("str", "success"),
    ):
        raise ValueError("collection requires success_values=('success',)")
    return owned


def _case_from_row(
    row: Any,
    plan: ComparisonPlan,
    repeats: Tuple[str, ...],
    limits: ComparisonLimits,
) -> ExpectedCase:
    _, case_id = field_value(row, plan.case_id)
    text(case_id, "case_id")
    attributes = []
    for field in plan.selected_fields:
        present, value = field_value(row, field)
        if not present:
            raise ValueError("selected case field is missing from dataset")
        scalar_key(value)
        attributes.append((field, value))
    for value in [case_id] + [value for _, value in attributes]:
        if (
            isinstance(value, str)
            and len(value.encode("utf-8")) > limits.max_string_bytes
        ):
            raise ValueError("dataset selected string limit exceeded")
    return ExpectedCase(case_id, repeats, tuple(attributes))


def _case_key(case: ExpectedCase) -> tuple:
    return (
        case.case_id,
        case.repeats,
        tuple((field, scalar_key(value)) for field, value in case.attributes),
    )


def plan_for_dataset(
    dataset: Iterable[Any],
    plan: ComparisonPlan,
    *,
    repeat_ids: Tuple[str, ...] = (),
    limits: Optional[ComparisonLimits] = None,
) -> ComparisonPlan:
    """Build a detached expected manifest before invoking any evaluation code.

    The dataset must have one row per case and explicit string case IDs. Only
    identity and slice columns are read; prompts and other payloads are not
    copied. The iterable is consumed once, so keep a reusable dataset separately
    if it will also be passed to an executor.

    When ``repeat_id`` is configured, supply all planned repeat IDs explicitly.
    A missing status field becomes ``__ragas_comparison_status``. Existing
    manifests must agree exactly rather than being silently replaced.
    """
    limits = limits or ComparisonLimits()
    plan.check_limits(limits)
    repeats = sequence(repeat_ids, "repeat_ids", limits.max_repeats)
    for repeat in repeats:
        text(repeat, "repeat_id")
    if len(set(repeats)) != len(repeats):
        raise ValueError("duplicate planned repeat_id")
    if bool(repeats) != (plan.repeat_id is not None):
        raise ValueError(
            "explicit repeat_ids must match the configured repeat_id field"
        )
    prepared = replace(plan, status_field=plan.status_field or STATUS_FIELD)
    _collection_fields(prepared)
    cases = []
    seen = set()
    selected_bytes = 0
    for index, row in enumerate(dataset):
        if index >= min(limits.max_rows, limits.max_cases):
            raise ValueError("dataset case limit exceeded")
        case = _case_from_row(row, prepared, repeats, limits)
        if case.case_id in seen:
            raise ValueError("duplicate case_id in dataset manifest")
        seen.add(case.case_id)
        selected_bytes += len(canonical_bytes(_case_key(case)))
        if selected_bytes > limits.max_selected_bytes:
            raise ValueError("dataset selected evidence byte limit exceeded")
        cases.append(case)
    if not cases:
        raise ValueError("cannot plan a comparison from an empty dataset")
    if len(cases) * max(1, len(repeats)) > limits.max_rows:
        raise ValueError("planned observation rows limit exceeded")
    cases.sort(key=lambda case: case.case_id)
    if plan.expected_cases is not None:
        if tuple(_case_key(case) for case in cases) != tuple(
            _case_key(case) for case in plan.expected_cases
        ):
            raise ValueError("dataset does not match the existing expected manifest")
    return replace(prepared, expected_cases=tuple(cases))


def _numeric_result(value: Any) -> Any:
    """Unwrap known Ragas result types without exporting reasoning or traces."""
    from ragas.metrics.result import MetricResult

    if isinstance(value, MetricResult):
        return native_score(value.value)
    # Collections metrics use their own Pydantic MetricResult. Import lazily so
    # manifest preparation itself does not initialize metric integrations.
    from ragas.metrics.collections.base import MetricResult as CollectionMetricResult

    if isinstance(value, CollectionMetricResult):
        return native_score(value.value)
    return native_score(value)


class ComparisonAdapter:
    """Wrap one async evaluator with authoritative comparison identities.

    Submit instances to ``Executor`` exactly like any async callable. Ordinary
    exceptions, cooperative timeouts and invalid outputs produce unsuccessful
    rows instead of disappearing. External cancellation propagates, with the
    prebuilt manifest retaining the missing job in the denominator.

    No retries occur. A timeout or cancellation does not prove that a remote
    side effect did not happen. Async functions must cooperate with cancellation;
    this adapter does not provide a hard process-level execution deadline.
    """

    def __init__(
        self,
        evaluator: Callable,
        *,
        plan: ComparisonPlan,
        timeout: Optional[float] = None,
        pass_repeat_id: bool = False,
        limits: Optional[ComparisonLimits] = None,
    ) -> None:
        self.limits = limits or ComparisonLimits()
        plan.check_limits(self.limits)
        if plan.expected_cases is None or plan.status_field is None:
            raise ValueError("ComparisonAdapter requires a plan prepared for a dataset")
        if not (
            inspect.iscoroutinefunction(evaluator)
            or inspect.iscoroutinefunction(getattr(evaluator, "__call__", None))
        ):
            raise ValueError("comparison evaluator must be an async callable")
        if type(pass_repeat_id) is not bool:
            raise ValueError("pass_repeat_id must be boolean")
        if timeout is not None:
            timeout = number(timeout, "timeout")
            if timeout <= 0:
                raise ValueError("timeout must be positive")
        self.owned_fields = _collection_fields(plan)
        self.plan = plan
        self.status_field = plan.status_field
        self.evaluator = evaluator
        self.timeout = timeout
        self.pass_repeat_id = pass_repeat_id
        self.cases = {case.case_id: case for case in plan.expected_cases}

    def _identity(self, row: Any, repeat_id: Optional[str]) -> ExpectedCase:
        _, case_id = field_value(row, self.plan.case_id)
        text(case_id, "case_id")
        case = self.cases.get(case_id)
        if case is None:
            raise ValueError("case_id is outside the prepared manifest")
        if self.plan.repeat_id is None:
            if repeat_id is not None:
                raise ValueError("repeat_id supplied for an unreplicated plan")
        elif repeat_id not in case.repeats:
            raise ValueError("repeat_id is outside the prepared manifest")
        current = _case_from_row(row, self.plan, case.repeats, self.limits)
        if _case_key(current) != _case_key(case):
            raise ValueError("case attributes changed after manifest preparation")
        return case

    def _observation(
        self,
        case: ExpectedCase,
        repeat_id: Optional[str],
        status: str,
        output: Any = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            self.plan.case_id: case.case_id,
            **dict(case.attributes),
        }
        if self.plan.repeat_id is not None:
            result[self.plan.repeat_id] = repeat_id
        result[self.status_field] = status
        if status == "success":
            if not isinstance(output, (Mapping, BaseModel)):
                raise ValueError("evaluator output must contain numeric metric columns")
            for field in self.owned_fields:
                if field_value(output, field)[0]:
                    raise ValueError(
                        "evaluator output cannot replace collection-owned fields"
                    )
            for metric in self.plan.metrics:
                present, value = field_value(output, metric.score_column)
                if present:
                    result[metric.score_column] = _numeric_result(value)
        else:
            for metric in self.plan.metrics:
                result[metric.score_column] = None
        # Reuse the same score/range validation as offline comparisons. No raw
        # exception text, reasoning, traces or unselected outputs are retained.
        _capture([result], self.plan, self.limits)
        return result

    async def __call__(
        self, row: Any, *, repeat_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate one planned observation without allowing identity overrides."""
        case = self._identity(row, repeat_id)
        kwargs = {"repeat_id": repeat_id} if self.pass_repeat_id else {}
        try:
            invocation = self.evaluator(row, **kwargs)
            if self.timeout is None:
                output = await invocation
            else:
                output = await asyncio.wait_for(invocation, timeout=self.timeout)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            return self._observation(case, repeat_id, "timeout")
        except Exception:
            return self._observation(case, repeat_id, "error")
        try:
            return self._observation(case, repeat_id, "success", output)
        except (TypeError, ValueError, OverflowError):
            return self._observation(case, repeat_id, "invalid_output")
