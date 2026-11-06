"""Export selected evaluation evidence for reproducible offline comparisons."""

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .config import ComparisonLimits, ComparisonPlan
from .io import PathLike, _check_structure, reject_input_alias
from .observations import _capture, field_value


def _selected_rows(rows: Iterable[Any], plan: ComparisonPlan) -> Iterator[dict]:
    fields = [plan.case_id, *plan.selected_fields]
    if plan.repeat_id is not None:
        fields.append(plan.repeat_id)
    if plan.status_field is not None:
        fields.append(plan.status_field)
    fields.extend(metric.score_column for metric in plan.metrics)
    fields = list(dict.fromkeys(fields))
    metric_columns = {metric.score_column for metric in plan.metrics}
    for row in rows:
        selected = {}
        for field in fields:
            present, value = field_value(row, field)
            if present:
                if (
                    field in metric_columns
                    and type(value) is float
                    and math.isnan(value)
                ):
                    value = None
                selected[field] = value
        yield selected


def write_observations(
    path: PathLike,
    rows: Iterable[Any],
    *,
    plan: ComparisonPlan,
    input_paths: Iterable[PathLike] = (),
    limits: Optional[ComparisonLimits] = None,
) -> int:
    """Atomically export validated selected columns as strict UTF-8 JSONL.

    Accepts Experiment rows, ComparisonAdapter outputs and ``evaluation_rows``.
    Identity, selected case attributes, configured status and metric columns are
    retained. Unselected responses, reasoning and trace fields are never read.
    Caller-selected case attributes can still contain sensitive information.

    NaN scores become JSON null, preserving eligibility but changing the missing
    reason from ``nan`` to ``null`` and therefore the evidence fingerprint. To
    compare reports with identical fingerprints, compare the exported files on
    both occasions. Omitted score/status fields remain omitted; failed statuses
    remain unchanged. Row order is preserved, though pairing is order-independent.

    The iterable is consumed once. Duplicate keys, invalid observations, format
    limits and iterator failures leave any existing destination untouched. A
    private temporary file is published only after full validation and fsync.
    Declare source paths to prevent overwriting evidence through path aliases.
    Returns the number of exported observations.
    """
    limits = limits or ComparisonLimits()
    plan.check_limits(limits)
    reject_input_alias(path, input_paths)
    destination = Path(path)
    temporary: Optional[str] = None
    count = 0
    total_bytes = 0
    total_nodes = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name

            def selected_evidence() -> Iterator[dict]:
                nonlocal count, total_bytes, total_nodes
                for row in _selected_rows(rows, plan):
                    # The comparison validator checks this row before resuming
                    # the iterator and writing it to the unpublished file.
                    yield row
                    total_nodes += _check_structure(row, limits)
                    if total_nodes > limits.max_json_nodes:
                        raise ValueError("JSONL nodes limit exceeded")
                    try:
                        content = (
                            json.dumps(
                                row,
                                ensure_ascii=False,
                                allow_nan=False,
                                separators=(",", ":"),
                            ).encode("utf-8")
                            + b"\n"
                        )
                    except (TypeError, ValueError, UnicodeError) as exc:
                        raise ValueError(
                            "selected evidence must be valid JSON"
                        ) from exc
                    if len(content) > limits.max_line_bytes:
                        raise ValueError("JSONL line byte limit exceeded")
                    total_bytes += len(content)
                    if total_bytes > limits.max_file_bytes:
                        raise ValueError("JSONL file byte limit exceeded")
                    stream.write(content)
                    count += 1

            _capture(selected_evidence(), plan, limits)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
        return count
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
