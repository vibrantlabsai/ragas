"""Human-readable comparison reports with explicit denominators and caveats."""

import html
import unicodedata
from typing import Optional

from rich.table import Table
from rich.text import Text

from .reports import ComparisonReport, MetricComparison


def safe_text(value: str) -> str:
    """Display identifiers without terminal controls or hidden bidi overrides."""
    pieces = []
    for character in value:
        if character in "\r\n\t":
            pieces.append(" ")
        elif unicodedata.category(character).startswith("C"):
            pieces.append(f"\\u{ord(character):04x}")
        else:
            pieces.append(character)
    return "".join(pieces)


def markdown_text(value: str) -> str:
    """Escape data used in Markdown tables, headings and paragraphs."""
    escaped = html.escape(safe_text(value), quote=True)
    for character in ("\\", "|", "[", "]", "(", ")", "`", "*", "_", "#", "!"):
        escaped = escaped.replace(character, "\\" + character)
    return escaped


def format_number(value: Optional[float]) -> str:
    """Use a consistent display precision, without replacing missing with zero."""
    return "—" if value is None else f"{value:.6g}"


def _interval(result: MetricComparison) -> str:
    estimate = result.estimate
    if estimate.interval_lower is None or estimate.interval_upper is None:
        return "not estimated"
    return f"[{format_number(estimate.interval_lower)}, {format_number(estimate.interval_upper)}]"


def render_table(report: ComparisonReport) -> Table:
    """Build a Rich table; caller-supplied names are literal Text, never markup."""
    table = Table(title=f"Paired comparison: {report.verdict}")
    for name in (
        "Metric / group",
        "Coverage",
        "Baseline",
        "Candidate",
        "Oriented Δ",
        "Interval",
        "Gate",
    ):
        table.add_column(name)
    for result in report.comparisons:
        coverage = result.coverage
        label = f"{safe_text(result.metric)} / {safe_text(result.group)}"
        if not result.required:
            label += " (diagnostic)"
        if result.conditional_on_eligible:
            label += " *"
        table.add_row(
            Text(label),
            f"{coverage.eligible_cases}/{coverage.total_cases}",
            format_number(result.estimate.baseline_mean),
            format_number(result.estimate.candidate_mean),
            format_number(result.estimate.oriented_delta),
            _interval(result),
            Text(
                result.verdict,
                style={"pass": "green", "regress": "red", "inconclusive": "yellow"}[
                    result.verdict
                ],
            ),
        )
    table.caption = "Positive oriented changes are better. * Conditional on eligible cases; omitted outcomes remain unknown."
    return table


def render_markdown(report: ComparisonReport) -> str:
    """Render a standalone reviewable artifact with coverage and provenance."""
    lines = [
        f"# Paired evaluation comparison: {report.verdict}",
        "",
        f"Mode: **{report.mode}**. Cohort: **{report.cohort_provenance}**.",
        f"Supplied rows: baseline **{report.baseline_rows}**, candidate **{report.candidate_rows}**.",
        f"Unsuccessful supplied rows: baseline **{report.baseline_unsuccessful}**, candidate **{report.candidate_unsuccessful}**.",
        "",
        "Each case has equal weight. Repeated observations are paired by repeat ID and averaged within a case.",
        "Positive oriented changes are better, including for metrics where lower values are preferred.",
        "",
        "| Metric | Group | Required | Eligible / total cases | Baseline | Candidate | Oriented change | Interval | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for result in report.comparisons:
        estimate, coverage = result.estimate, result.coverage
        values = (
            markdown_text(result.metric),
            markdown_text(result.group),
            "yes" if result.required else "no",
            f"{coverage.eligible_cases} / {coverage.total_cases}",
            format_number(estimate.baseline_mean),
            format_number(estimate.candidate_mean),
            format_number(estimate.oriented_delta),
            _interval(result),
            result.verdict,
        )
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Interpretation", ""])
    if report.mode == "bootstrap":
        lines.extend(
            [
                f"Paired case-level percentile intervals use {report.resamples} resamples and seed {report.seed}.",
                f"Requested confidence: {report.confidence:.6g}; adjustment: {report.adjustment}; fixed family size: {report.family_size}.",
                f"At least {report.min_cases} eligible cases are required. Degenerate differences and insufficient tail samples are inconclusive.",
                "Intervals are approximate and assume independent, representative cases. Repeated runs do not increase the independent sample count.",
                "The interval does not estimate unobserved within-case randomness. Repeated tuning against this benchmark does not establish out-of-sample quality.",
            ]
        )
    else:
        lines.append(
            "Observed mode describes only this finite benchmark; it makes no population-level confidence claim."
        )
    lines.extend(
        [
            f"Minimum required case coverage: {report.minimum_coverage:.6g}.",
            "Any result with exclusions describes the eligible cohort, including a regression that conservatively blocks CI.",
            "Missing outcomes could change the full-cohort result. No missing score is replaced with zero.",
            "",
        ]
    )
    if report.cohort_provenance == "observed_union":
        lines.append(
            "No expected manifest was supplied. Cases absent from both versions are unknown and cannot appear in this denominator."
        )
    else:
        lines.append(
            "The supplied manifest defines the expected case and repeat denominator, including cases absent from both versions."
        )
    lines.append(
        "Supplied-row counts do not reconstruct attempts omitted by an experiment runner."
    )
    for result in report.comparisons:
        lines.extend(
            [
                "",
                f"## {markdown_text(result.metric)} / {markdown_text(result.group)}",
                "",
                f"Direction: {result.direction}; tolerated degradation: {format_number(result.margin)}.",
                "Reasons: "
                + ", ".join(markdown_text(reason) for reason in result.reasons)
                + ".",
                f"Expected repeats: {result.coverage.expected_repeats}; paired eligible repeats: {result.estimate.paired_repeats}.",
            ]
        )
        if result.conditional_on_eligible:
            lines.append(
                "**Conditional on eligible cases:** excluded outcomes are not evidence of improvement."
            )
        lines.extend(_diagnostic_lines(result))
        if result.coverage.reason_counts:
            lines.extend(["", "| Exclusion reason | Cases |", "| --- | ---: |"])
            for reason, count in result.coverage.reason_counts:
                lines.append(f"| {markdown_text(reason)} | {count} |")
            lines.append(
                "A case can have several exclusion reasons; these counts are not disjoint."
            )
        if result.details:
            lines.extend(
                [
                    "",
                    "| Case ID | Eligible | Expected repeats | Baseline | Candidate | Oriented change | Reasons |",
                    "| --- | --- | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for detail in result.details:
                values = (
                    markdown_text(detail.case_id),
                    "yes" if detail.eligible else "no",
                    str(detail.expected_repeats),
                    format_number(detail.baseline_mean),
                    format_number(detail.candidate_mean),
                    format_number(detail.oriented_delta),
                    ", ".join(markdown_text(reason) for reason in detail.reasons),
                )
                lines.append("| " + " | ".join(values) + " |")
        if result.omitted_details:
            lines.append(
                f"{result.omitted_details} case details omitted by the report-wide detail limit; all cases contributed to the decision."
            )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"Algorithm: `{report.algorithm}`; report format version: {report.format_version}.",
            f"Plan fingerprint: `{report.plan_fingerprint}`.",
            f"Baseline selected-data fingerprint: `{report.baseline_fingerprint}`.",
            f"Candidate selected-data fingerprint: `{report.candidate_fingerprint}`.",
            "Fingerprints identify selected evidence; they do not authenticate its producer or include unselected payloads.",
            "",
        ]
    )
    return "\n".join(lines)


def _diagnostic_lines(result: MetricComparison) -> list:
    """Explain heterogeneity and attrition alongside each mean comparison."""
    diagnostic = result.diagnostics
    changes = diagnostic.changes
    repeats = diagnostic.repeats
    bounds = diagnostic.missing_outcomes
    lines = [
        "",
        "### Descriptive diagnostics",
        "",
        f"Cases improved beyond the margin: {changes.improved_cases}; regressed beyond the margin: {changes.regressed_cases}; within the margin: {changes.within_margin_cases}.",
        "These counts use the eligible paired cohort and do not add independent statistical tests.",
        "",
        "| Distribution | Minimum | 5th percentile | Median | 95th percentile | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, distribution in (
        ("Baseline case means", diagnostic.baseline),
        ("Candidate case means", diagnostic.candidate),
        ("Oriented case changes", changes),
    ):
        values = (
            distribution.minimum,
            distribution.p05,
            distribution.median,
            distribution.p95,
            distribution.maximum,
        )
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(format_number(value) for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            f"Eligible cases with repeated runs: {repeats.repeated_cases}.",
            f"Cases whose paired repeat changes cross both sides of the tolerance margin: {repeats.direction_flip_cases}.",
            "",
            "| Repeat variability | Baseline | Candidate | Oriented difference |",
            "| --- | ---: | ---: | ---: |",
            "| RMS within-case standard deviation | "
            + " | ".join(
                format_number(value)
                for value in (
                    repeats.baseline_rms_standard_deviation,
                    repeats.candidate_rms_standard_deviation,
                    repeats.difference_rms_standard_deviation,
                )
            )
            + " |",
            "| Maximum within-case range | "
            + " | ".join(
                format_number(value)
                for value in (
                    repeats.maximum_baseline_range,
                    repeats.maximum_candidate_range,
                    repeats.maximum_difference_range,
                )
            )
            + " |",
            "Within-case variability is descriptive and does not increase the bootstrap's independent case count.",
            "",
            "### Missing-outcome sensitivity",
            "",
        ]
    )
    if bounds.lower is None or bounds.upper is None:
        lines.append(
            "Bounds unavailable: "
            + markdown_text(bounds.reason or "not_available")
            + "."
        )
    else:
        lines.extend(
            [
                f"Possible oriented finite-cohort mean change: **[{format_number(bounds.lower)}, {format_number(bounds.upper)}]**.",
                f"Assumed score range: [{format_number(bounds.score_minimum)}, {format_number(bounds.score_maximum)}].",
                f"Conclusion under these bounds: **{bounds.conclusion}**.",
                f"Unknown planned repeats: {bounds.unknown_baseline_repeats} baseline, {bounds.unknown_candidate_repeats} candidate.",
                "These are completion bounds over the declared cohort, not a confidence interval or an imputed estimate.",
                "They include usable scores from partial cases, assume all unknown scores respect the declared range, and do not override the configured coverage gate.",
            ]
        )
    return lines
