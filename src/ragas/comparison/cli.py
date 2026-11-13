"""CLI adapter for saved comparisons; independent of experiment execution."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.text import Text

from .engine import compare
from .io import read_jsonl, read_plan, reject_input_alias, write_report
from .render import render_markdown, render_table


def compare_command(
    baseline: Path = typer.Option(
        ..., "--baseline", help="Baseline observation JSONL file."
    ),
    candidate: Path = typer.Option(
        ..., "--candidate", help="Candidate observation JSONL file."
    ),
    plan: Path = typer.Option(
        ..., "--plan", help="Versioned comparison plan JSON file."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", help="Write a JSON or Markdown report atomically."
    ),
    output_format: str = typer.Option(
        "table", "--format", help="Console format: table, json, or markdown."
    ),
    file_format: str = typer.Option(
        "json",
        "--output-format",
        help="Report file format: json, markdown, csv, or junit.",
    ),
    input_format: str = typer.Option(
        "jsonl",
        "--input-format",
        help="Observation format for both files: jsonl or csv.",
    ),
) -> None:
    """Compare saved observations by case ID, with coverage-aware CI exits.

    Exit codes: 0 pass, 1 regression, 2 inconclusive, 3 invalid input or file error.
    This command makes no model calls and does not execute an experiment.
    """
    console = Console()
    try:
        if output_format not in ("table", "json", "markdown"):
            raise ValueError("console format must be table, json, or markdown")
        if file_format not in ("json", "markdown", "csv", "junit"):
            raise ValueError("output format must be json, markdown, csv, or junit")
        if input_format not in ("jsonl", "csv"):
            raise ValueError("input format must be jsonl or csv")
        inputs = (baseline, candidate, plan)
        if output is not None:
            reject_input_alias(output, inputs)
        configuration = read_plan(plan)
        if input_format == "csv":
            from .formats import read_csv

            before = read_csv(baseline, plan=configuration)
            after = read_csv(candidate, plan=configuration)
        else:
            before, after = read_jsonl(baseline), read_jsonl(candidate)
        report = compare(before, after, plan=configuration)
        if output is not None:
            write_report(output, report, output_format=file_format, input_paths=inputs)
    except (OSError, ValueError, TypeError) as exc:
        # Validation messages describe fields and types, never raw row values.
        Console(stderr=True).print(Text(f"Comparison input error: {exc}", style="red"))
        raise typer.Exit(3) from exc
    if output_format == "json":
        typer.echo(report.to_json())
    elif output_format == "markdown":
        typer.echo(render_markdown(report))
    else:
        console.print(render_table(report))
        console.print(
            Text(
                f"Cohort: {report.cohort_provenance}; supplied rows: {report.baseline_rows} baseline, {report.candidate_rows} candidate."
            )
        )
    raise typer.Exit(report.exit_code)


def main() -> None:
    """Run the same command as a module for hermetic CI environments."""
    typer.run(compare_command)
