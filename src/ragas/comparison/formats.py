"""CSV interoperability and CI-native exports for comparison results."""

import csv
import io
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Optional

from .config import ComparisonLimits, ComparisonPlan, number
from .io import PathLike
from .render import safe_text
from .reports import ComparisonReport


def _csv_lines(path: PathLike, limits: ComparisonLimits) -> Iterator[str]:
    total_bytes = 0
    with Path(path).open("rb") as stream:
        while True:
            line = stream.readline(limits.max_line_bytes + 1)
            if not line:
                break
            if len(line) > limits.max_line_bytes:
                raise ValueError("CSV physical line byte limit exceeded")
            total_bytes += len(line)
            if total_bytes > limits.max_file_bytes:
                raise ValueError("CSV file byte limit exceeded")
            try:
                yield line.decode("utf-8")
            except UnicodeError as exc:
                raise ValueError("CSV input must be UTF-8") from exc


def _csv_number(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        parsed = int(value) if re.fullmatch(r"[+-]?[0-9]+", value) else float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("CSV metric fields must be numeric or empty") from exc
    if type(parsed) is float and math.isnan(parsed):
        return parsed
    return number(parsed, "CSV metric score")


def read_csv(
    path: PathLike,
    *,
    plan: ComparisonPlan,
    limits: Optional[ComparisonLimits] = None,
) -> Iterator[dict]:
    """Read a deliberately selected CSV input format with numeric metric columns.

    IDs, case attributes and statuses remain strings, including leading zeros.
    Only explicitly configured metric columns are converted to numbers. Empty
    metric cells become null; NaN cells are missing evidence. Infinity, boolean
    spellings, duplicate headers and malformed row widths are input errors.

    CSV has no type schema for attributes. Use JSONL or typed Experiment rows
    when slices or identity attributes require booleans, numbers or null values.
    The standard library's parser field-size ceiling also applies.
    """
    limits = limits or ComparisonLimits()
    plan.check_limits(limits)
    reader = csv.reader(_csv_lines(path, limits), strict=True)
    try:
        header = next(reader, None)
        if (
            not header
            or len(set(header)) != len(header)
            or any(not name for name in header)
        ):
            raise ValueError("CSV requires unique nonempty column headers")
        if len(header) > limits.max_fields + limits.max_metrics + 3:
            raise ValueError("CSV column count limit exceeded")
        if plan.case_id not in header:
            raise ValueError("CSV is missing the case_id column")
        selected = {plan.case_id, *plan.selected_fields}
        if plan.repeat_id:
            selected.add(plan.repeat_id)
        if plan.status_field:
            selected.add(plan.status_field)
        metric_columns = {metric.score_column for metric in plan.metrics}
        selected.update(metric_columns)
        count = 0
        for cells in reader:
            if not cells:
                continue
            count += 1
            if count > limits.max_rows:
                raise ValueError("CSV rows limit exceeded")
            if len(cells) != len(header):
                raise ValueError("CSV row width does not match its header")
            if any(
                len(cell.encode("utf-8")) > limits.max_string_bytes for cell in cells
            ):
                raise ValueError("CSV field string limit exceeded")
            row = {}
            for name, value in zip(header, cells):
                if name in selected:
                    row[name] = _csv_number(value) if name in metric_columns else value
            yield row
    except csv.Error as exc:
        raise ValueError(
            "invalid CSV structure or parser field limit exceeded"
        ) from exc


def _spreadsheet_text(value: str) -> str:
    value = safe_text(value)
    if value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def render_csv(report: ComparisonReport) -> str:
    """Export one row per metric/group for spreadsheet or analytical review.

    Text fields that spreadsheets could interpret as formulas are prefixed with
    an apostrophe. Numeric columns remain numeric; missing values remain empty.
    This summary includes all comparisons, regardless of detail truncation.
    """
    columns = (
        "metric",
        "group",
        "required",
        "verdict",
        "direction",
        "margin",
        "total_cases",
        "eligible_cases",
        "excluded_cases",
        "coverage",
        "baseline_mean",
        "candidate_mean",
        "raw_delta",
        "oriented_delta",
        "interval_lower",
        "interval_upper",
        "conditional_on_eligible",
        "improved_cases",
        "regressed_cases",
        "within_margin_cases",
        "repeat_direction_flip_cases",
        "missing_bound_lower",
        "missing_bound_upper",
        "missing_bound_conclusion",
        "reasons",
        "cohort_provenance",
        "plan_fingerprint",
        "baseline_fingerprint",
        "candidate_fingerprint",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for result in report.comparisons:
        coverage, estimate, diagnostics = (
            result.coverage,
            result.estimate,
            result.diagnostics,
        )
        writer.writerow(
            {
                "metric": _spreadsheet_text(result.metric),
                "group": _spreadsheet_text(result.group),
                "required": result.required,
                "verdict": result.verdict,
                "direction": result.direction,
                "margin": result.margin,
                "total_cases": coverage.total_cases,
                "eligible_cases": coverage.eligible_cases,
                "excluded_cases": coverage.excluded_cases,
                "coverage": coverage.fraction,
                "baseline_mean": estimate.baseline_mean,
                "candidate_mean": estimate.candidate_mean,
                "raw_delta": estimate.raw_delta,
                "oriented_delta": estimate.oriented_delta,
                "interval_lower": estimate.interval_lower,
                "interval_upper": estimate.interval_upper,
                "conditional_on_eligible": result.conditional_on_eligible,
                "improved_cases": diagnostics.changes.improved_cases,
                "regressed_cases": diagnostics.changes.regressed_cases,
                "within_margin_cases": diagnostics.changes.within_margin_cases,
                "repeat_direction_flip_cases": diagnostics.repeats.direction_flip_cases,
                "missing_bound_lower": diagnostics.missing_outcomes.lower,
                "missing_bound_upper": diagnostics.missing_outcomes.upper,
                "missing_bound_conclusion": diagnostics.missing_outcomes.conclusion,
                "reasons": ";".join(result.reasons),
                "cohort_provenance": report.cohort_provenance,
                "plan_fingerprint": report.plan_fingerprint,
                "baseline_fingerprint": report.baseline_fingerprint,
                "candidate_fingerprint": report.candidate_fingerprint,
            }
        )
    return output.getvalue()


def render_junit(report: ComparisonReport) -> str:
    """Export required regressions as failures and inconclusive gates as errors.

    Optional comparisons remain diagnostic and do not fail a CI test suite.
    No execution duration is fabricated: comparison evidence is not a test timer.
    """
    required = [item for item in report.comparisons if item.required]
    suite = ET.Element(
        "testsuite",
        {
            "name": "ragas.paired_comparison",
            "tests": str(len(report.comparisons)),
            "failures": str(sum(item.verdict == "regress" for item in required)),
            "errors": str(sum(item.verdict == "inconclusive" for item in required)),
            "skipped": str(sum(not item.required for item in report.comparisons)),
        },
    )
    properties = ET.SubElement(suite, "properties")
    for name, value in (
        ("algorithm", report.algorithm),
        ("mode", report.mode),
        ("cohort_provenance", report.cohort_provenance),
        ("plan_fingerprint", report.plan_fingerprint),
        ("baseline_fingerprint", report.baseline_fingerprint),
        ("candidate_fingerprint", report.candidate_fingerprint),
    ):
        ET.SubElement(properties, "property", {"name": name, "value": value})
    for item in report.comparisons:
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "ragas.comparison." + safe_text(item.group),
                "name": safe_text(item.metric),
            },
        )
        message = "; ".join(item.reasons)
        if not item.required:
            ET.SubElement(
                case, "skipped", {"message": "Diagnostic comparison: " + item.verdict}
            )
        elif item.verdict == "regress":
            ET.SubElement(
                case, "failure", {"type": "EvaluationRegression", "message": message}
            )
        elif item.verdict == "inconclusive":
            ET.SubElement(
                case, "error", {"type": "InconclusiveComparison", "message": message}
            )
        evidence = ET.SubElement(case, "system-out")
        evidence.text = (
            f"coverage={item.coverage.eligible_cases}/{item.coverage.total_cases}; "
            f"oriented_delta={item.estimate.oriented_delta}; "
            f"conditional_on_eligible={item.conditional_on_eligible}; "
            f"missing_outcome_bounds=[{item.diagnostics.missing_outcomes.lower}, "
            f"{item.diagnostics.missing_outcomes.upper}]"
        )
    return ET.tostring(suite, encoding="unicode", xml_declaration=True) + "\n"
