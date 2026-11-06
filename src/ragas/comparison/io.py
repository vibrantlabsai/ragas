"""Strict bounded plan/JSONL reading and atomic comparison report writing."""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Tuple, Union

from .config import (
    ComparisonLimits,
    ComparisonPlan,
    ExpectedCase,
    MetricSpec,
    SliceSpec,
)
from .reports import ComparisonReport

PathLike = Union[str, Path]
PLAN_FORMAT = "ragas.comparison.plan"
PLAN_VERSION = 1


def _object(
    value: Any, allowed: Iterable[str], required: Iterable[str], label: str
) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if set(value) - set(allowed):
        raise ValueError(f"{label} contains unknown fields")
    if set(required) - set(value):
        raise ValueError(f"{label} is missing required fields")
    return dict(value)


def _array(value: Any, label: str) -> list:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{label} must be an array")
    return list(value)


def plan_to_dict(plan: ComparisonPlan) -> dict:
    """Create the versioned, human-editable plan representation."""
    data = asdict(plan)
    if data["expected_cases"] is not None:
        data["expected_cases"] = [
            {**case, "attributes": dict(case["attributes"])}
            for case in data["expected_cases"]
        ]
    return {"format": PLAN_FORMAT, "version": PLAN_VERSION, "plan": data}


def plan_from_dict(
    envelope: Any,
    *,
    limits: Optional[ComparisonLimits] = None,
) -> ComparisonPlan:
    """Decode a strict plan; no executable expressions or imported callables."""
    limits = limits or ComparisonLimits()
    envelope = _object(
        envelope,
        ("format", "version", "plan"),
        ("format", "version", "plan"),
        "plan envelope",
    )
    if envelope["format"] != PLAN_FORMAT:
        raise ValueError("unsupported comparison plan format")
    if type(envelope["version"]) is not int or envelope["version"] != PLAN_VERSION:
        raise ValueError("unsupported comparison plan version")
    allowed = tuple(ComparisonPlan.__dataclass_fields__)
    data = _object(envelope["plan"], allowed, ("metrics",), "plan")
    metrics = _array(data["metrics"], "metrics")
    if len(metrics) > limits.max_metrics:
        raise ValueError("comparison metrics limit exceeded")
    data["metrics"] = tuple(
        MetricSpec(
            **_object(
                metric, tuple(MetricSpec.__dataclass_fields__), ("name",), "metric"
            )
        )
        for metric in metrics
    )
    slices = _array(data.get("slices", []), "slices")
    if len(slices) > limits.max_slices:
        raise ValueError("comparison slices limit exceeded")
    data["slices"] = tuple(
        SliceSpec(
            **_object(
                item,
                tuple(SliceSpec.__dataclass_fields__),
                ("name", "field", "values"),
                "slice",
            )
        )
        for item in slices
    )
    cases = data.get("expected_cases")
    if cases is not None:
        cases = _array(cases, "expected_cases")
        if len(cases) > limits.max_cases:
            raise ValueError("comparison cases limit exceeded")
        expected = []
        for item in cases:
            case = _object(
                item,
                ("case_id", "repeats", "attributes"),
                ("case_id",),
                "expected case",
            )
            attributes = case.get("attributes", {})
            if not isinstance(attributes, dict):
                raise ValueError("expected attributes must be an object")
            case["attributes"] = tuple(attributes.items())
            expected.append(ExpectedCase(**case))
        data["expected_cases"] = tuple(expected)
    plan = ComparisonPlan(**data)
    plan.check_limits(limits)
    return plan


def _unique_object(pairs: list) -> dict:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON object key")
        result[name] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite JSON constants are not supported")


def _check_structure(value: Any, limits: ComparisonLimits) -> int:
    """Count JSON nodes iteratively after bounded-byte decoding."""
    stack = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > limits.max_json_nodes:
            raise ValueError("JSON nodes limit exceeded")
        if depth > limits.max_json_depth:
            raise ValueError("JSON depth limit exceeded")
        if isinstance(item, dict):
            stack.extend((key, depth + 1) for key in item)
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            if len(item.encode("utf-8")) > limits.max_string_bytes:
                raise ValueError("JSON string limit exceeded")
    return nodes


def _decode(data: bytes, limits: ComparisonLimits) -> Tuple[Any, int]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid UTF-8 JSON input") from exc
    return value, _check_structure(value, limits)


def read_plan(
    path: PathLike, *, limits: Optional[ComparisonLimits] = None
) -> ComparisonPlan:
    """Read a bounded versioned plan file with caller-owned limits."""
    limits = limits or ComparisonLimits()
    with Path(path).open("rb") as stream:
        data = stream.read(limits.max_file_bytes + 1)
    if len(data) > limits.max_file_bytes:
        raise ValueError("plan file byte limit exceeded")
    value, _ = _decode(data, limits)
    return plan_from_dict(value, limits=limits)


def read_jsonl(
    path: PathLike, *, limits: Optional[ComparisonLimits] = None
) -> Iterator[dict]:
    """Stream numeric JSONL observations, including LocalJSONLBackend files.

    Empty lines are ignored. Limits cover the complete file, not just each row.
    Missing scores use null or an omitted field; JSON NaN extensions are invalid.
    """
    limits = limits or ComparisonLimits()
    total_bytes = total_nodes = rows = 0
    with Path(path).open("rb") as stream:
        while True:
            line = stream.readline(limits.max_line_bytes + 1)
            if not line:
                break
            if len(line) > limits.max_line_bytes:
                raise ValueError("JSONL line byte limit exceeded")
            total_bytes += len(line)
            if total_bytes > limits.max_file_bytes:
                raise ValueError("JSONL file byte limit exceeded")
            if not line.strip():
                continue
            rows += 1
            if rows > limits.max_rows:
                raise ValueError("JSONL rows limit exceeded")
            value, nodes = _decode(line, limits)
            total_nodes += nodes
            if total_nodes > limits.max_json_nodes:
                raise ValueError("JSONL nodes limit exceeded")
            if not isinstance(value, dict):
                raise ValueError("each JSONL observation must be an object")
            yield value


def reject_input_alias(output: PathLike, inputs: Iterable[PathLike]) -> None:
    """Reject lexical, symlink and existing hard-link aliases of an input."""
    destination = Path(output)
    resolved = destination.resolve()
    for item in inputs:
        source = Path(item)
        if resolved == source.resolve():
            raise ValueError("output path must not alias an input")
        if (
            destination.exists()
            and source.exists()
            and os.path.samefile(destination, source)
        ):
            raise ValueError("output path must not alias an input")


def _atomic_write(path: PathLike, content: str, limits: ComparisonLimits) -> None:
    data = content.encode("utf-8")
    if len(data) > limits.max_file_bytes:
        raise ValueError("output file byte limit exceeded")
    destination = Path(path)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def write_plan(
    path: PathLike, plan: ComparisonPlan, *, limits: Optional[ComparisonLimits] = None
) -> None:
    """Atomically write a reusable versioned plan in a private temporary file."""
    limits = limits or ComparisonLimits()
    plan.check_limits(limits)
    content = (
        json.dumps(
            plan_to_dict(plan),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    _atomic_write(path, content, limits)


def write_report(
    path: PathLike,
    report: ComparisonReport,
    *,
    output_format: str = "json",
    input_paths: Iterable[PathLike] = (),
    limits: Optional[ComparisonLimits] = None,
) -> None:
    """Write JSON, Markdown, CSV or JUnit without overwriting declared inputs."""
    limits = limits or ComparisonLimits()
    reject_input_alias(path, input_paths)
    if output_format == "json":
        content = report.to_json() + "\n"
    elif output_format == "markdown":
        from .render import render_markdown

        content = render_markdown(report)
    elif output_format in ("csv", "junit"):
        from .formats import render_csv, render_junit

        content = render_csv(report) if output_format == "csv" else render_junit(report)
    else:
        raise ValueError("output format must be json, markdown, csv, or junit")
    _atomic_write(path, content, limits)
