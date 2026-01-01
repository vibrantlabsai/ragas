"""
Ragas CLI for running experiments from command line.
"""

import asyncio
import importlib.util
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

# from ragas.experimental.project.core import Project  # TODO: Project module not implemented yet
from ragas.utils import console

app = typer.Typer(help="Ragas CLI for running LLM evaluations")


# Create a callback for the main app to make it a group
@app.callback()
def main():
    """Ragas CLI for running LLM evaluations"""
    pass


# Rich utility functions
def success(text: str) -> None:
    """Print text in green color for success messages."""
    console.print(text, style="green")


def error(text: str) -> None:
    """Print text in red color for error messages."""
    console.print(text, style="red")


def info(text: str) -> None:
    """Print text in cyan color for info messages."""
    console.print(text, style="cyan")


def warning(text: str) -> None:
    """Print text in yellow color for warning messages."""
    console.print(text, style="yellow")


def create_numerical_metrics_table(
    metrics_data: Dict[str, Dict], has_baseline: bool = False
) -> Table:
    """Create a Rich table for numerical metrics."""
    table = Table(title="Numerical Metrics")

    # Add columns based on whether we have baseline comparison
    table.add_column("Metric", style="yellow", no_wrap=True)
    table.add_column("Current", justify="right")

    if has_baseline:
        table.add_column("Baseline", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Gate", justify="center")

    for metric_name, values in metrics_data.items():
        current_value = values["current"]

        if has_baseline:
            baseline_value = values["baseline"]
            delta = current_value - baseline_value

            is_improvement = delta > 0
            # Format delta with arrow and color
            arrow = "▲" if delta > 0 else "▼"
            delta_str = f"{arrow}{abs(delta):.3f}"
            delta_color = "green" if is_improvement else "red"

            # Determine if test passes (allow small regression)
            passed = is_improvement or abs(delta) < 0.01
            gate_str = (
                Text("pass", style="green") if passed else Text("fail", style="red")
            )

            table.add_row(
                metric_name.replace("_", " "),
                f"{current_value:.3f}",
                f"{baseline_value:.3f}",
                Text(delta_str, style=delta_color),
                gate_str,
            )
        else:
            table.add_row(metric_name.replace("_", " "), f"{current_value:.3f}")

    return table


def create_categorical_metrics_table(
    metrics_data: Dict[str, Dict], has_baseline: bool = False
) -> Table:
    """Create a Rich table for categorical metrics."""
    table = Table(title="Categorical Metrics")

    # Add columns
    table.add_column("Metric", style="yellow", no_wrap=True)
    table.add_column("Category", style="cyan")
    table.add_column("Current", justify="right")

    if has_baseline:
        table.add_column("Baseline", justify="right")
        table.add_column("Delta", justify="right")

    for metric_name, values in metrics_data.items():
        current_value = values["current"]

        if has_baseline:
            baseline_value = values["baseline"]

            # Get all unique categories
            all_categories = set(current_value.keys()) | set(baseline_value.keys())

            for i, category in enumerate(sorted(all_categories)):
                current_count = current_value.get(category, 0)
                baseline_count = baseline_value.get(category, 0)
                delta = current_count - baseline_count

                if delta > 0:
                    delta_str = Text(f"▲{delta}", style="green")
                elif delta < 0:
                    delta_str = Text(f"▼{abs(delta)}", style="red")
                else:
                    delta_str = Text("→", style="dim")

                # Only show metric name on first row for this metric
                metric_display = metric_name.replace("_", " ") if i == 0 else ""

                table.add_row(
                    metric_display,
                    category,
                    str(current_count),
                    str(baseline_count),
                    delta_str,
                )
        else:
            # Sort by count (descending) for better readability
            if current_value:
                sorted_items = sorted(
                    current_value.items(), key=lambda x: x[1], reverse=True
                )
                for i, (category, count) in enumerate(sorted_items):
                    # Only show metric name on first row for this metric
                    metric_display = metric_name.replace("_", " ") if i == 0 else ""
                    table.add_row(metric_display, category, str(count))
            else:
                table.add_row(metric_name.replace("_", " "), "N/A", "0")

    return table


def extract_metrics_from_experiment(experiment, metric_fields: list) -> Dict[str, list]:
    """Extract metric values from experiment entries."""
    metrics_data = {field_name: [] for field_name in metric_fields}
    for entry in experiment:
        for field_name in metric_fields:
            field_value = getattr(entry, field_name)
            metrics_data[field_name].append(field_value)
    return metrics_data


def calculate_aggregated_metrics(metrics_data: Dict[str, list]) -> Dict[str, Dict]:
    """Calculate aggregated scores for metrics (numeric average or categorical frequency)."""
    agg_metrics = {}
    for metric_name, scores in metrics_data.items():
        # Remove None values
        scores = [score for score in scores if score is not None]
        if not scores:
            avg_score = 0
        elif isinstance(scores[0], (int, float)):
            # Numeric metric - calculate average
            avg_score = sum(scores) / len(scores)
        else:
            # Categorical metric - create frequency distribution
            avg_score = dict(Counter(scores))
        agg_metrics[metric_name] = {"score": avg_score}
    return agg_metrics


def separate_metrics_by_type(
    current_metrics: Dict, baseline_metrics: Optional[Dict] = None
) -> tuple:
    """Separate metrics into numeric and categorical dictionaries."""
    numeric_metrics = {}
    categorical_metrics = {}

    for metric_name, current_metric in current_metrics.items():
        current_value = current_metric.get("score", 0)

        if baseline_metrics and metric_name in baseline_metrics:
            baseline_value = baseline_metrics[metric_name].get("score", 0)

            if isinstance(current_value, dict) and isinstance(baseline_value, dict):
                categorical_metrics[metric_name] = {
                    "current": current_value,
                    "baseline": baseline_value,
                }
            else:
                numeric_metrics[metric_name] = {
                    "current": current_value,
                    "baseline": baseline_value,
                }
        else:
            # No baseline comparison
            if isinstance(current_value, dict):
                categorical_metrics[metric_name] = {"current": current_value}
            else:
                numeric_metrics[metric_name] = {"current": current_value}

    return numeric_metrics, categorical_metrics


def display_metrics_tables(
    numeric_metrics: Dict, categorical_metrics: Dict, has_baseline: bool = False
) -> None:
    """Display metrics tables for numeric and categorical data."""
    if numeric_metrics:
        table = create_numerical_metrics_table(
            numeric_metrics, has_baseline=has_baseline
        )
        console.print(table)

    if categorical_metrics:
        table = create_categorical_metrics_table(
            categorical_metrics, has_baseline=has_baseline
        )
        console.print(table)


def load_eval_module(eval_path: str) -> Any:
    """Load an evaluation module from a file path."""
    eval_path_obj = Path(eval_path).resolve()
    if not eval_path_obj.exists():
        error(f"Error: Evaluation file not found: {eval_path_obj}")
        raise typer.Exit(1)

    # Add the eval directory to Python path so imports work
    eval_dir = eval_path_obj.parent
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    # Load the module
    spec = importlib.util.spec_from_file_location("eval_module", eval_path_obj)
    if spec is None or spec.loader is None:
        error(f"Error: Could not load evaluation file: {eval_path_obj}")
        raise typer.Exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def run_experiments(
    project,
    experiment_func,
    dataset_name: str,
    input_data_class: type,
    baseline_name: Optional[str] = None,
    metrics: Optional[str] = None,
    name: Optional[str] = None,
):
    """Run experiments using ragas dataset system."""
    console.print(f"Getting dataset: {dataset_name}")

    # Get the dataset using project's get_dataset method
    try:
        dataset = project.get_dataset(dataset_name=dataset_name, model=input_data_class)
        dataset.load()  # Load the dataset data
        success(f"✓ Loaded dataset with {len(dataset)} rows")
    except Exception as e:
        error(f"Error loading dataset '{dataset_name}': {e}")
        raise typer.Exit(1)

    # Run the experiment using the run_async method
    try:
        experiment_result = await experiment_func.run_async(dataset, name=name)
        success("✓ Completed experiments successfully")
    except Exception as e:
        error(f"Error running experiments: {e}")
        raise typer.Exit(1)

    # Parse metrics from provided list
    metric_fields = [
        metric.strip() for metric in (metrics or "").split(",") if metric.strip()
    ]

    # Extract metrics from current experiment
    current_metrics_data = extract_metrics_from_experiment(
        experiment_result, metric_fields
    )
    current_agg_metrics = calculate_aggregated_metrics(current_metrics_data)

    # Handle baseline comparison if specified
    if baseline_name:
        console.print(f"Comparing against baseline: {baseline_name}")
        try:
            # The experiment model should be the return type or we can infer it
            baseline = project.get_experiment(
                baseline_name, model=experiment_result.model
            )
            baseline.load()

            # Create comparison header with panel
            header_content = f"Experiment: {experiment_result.name}\nDataset: {dataset_name} ({len(dataset)} rows)\nBaseline: {baseline_name}"
            console.print(
                Panel(
                    header_content,
                    title="Ragas Evaluation Results",
                    style="bold white",
                    width=80,
                )
            )

            # Extract metrics from baseline experiment
            baseline_metrics_data = extract_metrics_from_experiment(
                baseline, metric_fields
            )
            baseline_agg_metrics = calculate_aggregated_metrics(baseline_metrics_data)

            # Separate metrics by type with baseline comparison
            numeric_metrics, categorical_metrics = separate_metrics_by_type(
                current_agg_metrics, baseline_agg_metrics
            )

            # Display metrics tables
            display_metrics_tables(
                numeric_metrics, categorical_metrics, has_baseline=True
            )

            success("✓ Comparison completed")

        except Exception as e:
            error(f"Error comparing with baseline: {e}")
            traceback.print_exc()  # Print the full traceback with line numbers
            # Continue without comparison
    else:
        # No baseline provided, just print the current experiment metrics
        header_content = f"Experiment: {experiment_result.name}\nDataset: {dataset_name} ({len(dataset)} rows)"
        console.print(
            Panel(
                header_content,
                title="Ragas Evaluation Results",
                style="bold white",
                width=80,
            )
        )

        # Separate metrics by type without baseline comparison
        numeric_metrics, categorical_metrics = separate_metrics_by_type(
            current_agg_metrics
        )

        # Display metrics tables
        display_metrics_tables(numeric_metrics, categorical_metrics, has_baseline=False)

        success("✓ Experiment results displayed")


@app.command()
def evals(
    eval_file: str = typer.Argument(..., help="Path to the evaluation file"),
    dataset: str = typer.Option(
        ..., "--dataset", help="Name of the dataset in the project"
    ),
    metrics: str = typer.Option(
        ..., "--metrics", help="Comma-separated list of metric field names to evaluate"
    ),
    baseline: Optional[str] = typer.Option(
        None, "--baseline", help="Baseline experiment name to compare against"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", help="Name of the experiment run"
    ),
):
    """Run evaluations on a dataset."""
    console.print(f"Running evaluation: {eval_file}")
    console.print(f"Dataset: {dataset}")
    if baseline:
        console.print(f"Baseline: {baseline}")

    try:
        # Load the evaluation module
        eval_module = load_eval_module(eval_file)

        # Find the project and experiment function
        project = None
        experiment_func = None
        input_data_class = None

        # Look for project and experiment in the module
        for attr_name in dir(eval_module):
            attr = getattr(eval_module, attr_name)
            # TODO: Project class not implemented yet
            # if isinstance(attr, Project):
            #     project = attr
            if hasattr(attr, "get_dataset") and hasattr(attr, "get_experiment"):
                project = attr
            elif hasattr(attr, "run_async"):
                experiment_func = attr
                # Get input type from the experiment function's signature
                import inspect

                sig = inspect.signature(attr)
                if sig.parameters:
                    # Get the first parameter's annotation
                    first_param = next(iter(sig.parameters.values()))
                    if (
                        first_param.annotation
                        and first_param.annotation != inspect.Parameter.empty
                    ):
                        input_data_class = first_param.annotation

        if project is None:
            error("Error: No Project instance found in evaluation file")
            raise typer.Exit(1)

        if experiment_func is None:
            error(
                "Error: No experiment function with run_async method found in evaluation file"
            )
            raise typer.Exit(1)

        if input_data_class is None:
            error(
                "Error: Could not determine input data class from experiment function"
            )
            raise typer.Exit(1)

        # Run the experiments
        asyncio.run(
            run_experiments(
                project,
                experiment_func,
                dataset,
                input_data_class,
                baseline,
                metrics,
                name,
            )
        )
        success("✓ Evaluation completed successfully")

    except Exception as e:
        error(f"Error running evaluation: {e}")
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def quickstart(
    template: Optional[str] = typer.Argument(
        None,
        help="Template name (e.g., 'rag_eval', 'agent_evals'). Leave empty to see available templates.",
    ),
    output_dir: str = typer.Option(
        ".", "--output-dir", "-o", help="Directory to create the project in"
    ),
):
    """
    Clone a complete example project to get started with Ragas.

    Similar to 'uvx hud-python quickstart', this creates a complete example
    project with all necessary files and dependencies.

    Examples:
        ragas quickstart                    # List available templates
        ragas quickstart rag_eval           # Create a RAG evaluation project
        ragas quickstart agent_evals -o ./my-project
    """
    import shutil
    import time
    from pathlib import Path

    # Define available templates with descriptions
    templates = {
        "rag_eval": {
            "name": "RAG Evaluation",
            "description": "Evaluate a RAG (Retrieval Augmented Generation) system with custom metrics",
            "source_path": "ragas_examples/rag_eval",
        },
        "improve_rag": {
            "name": "Improve RAG",
            "description": "Compare naive vs agentic RAG using BM25 retrieval and HuggingFace docs",
            "source_path": "ragas_examples/improve_rag",
        },
        "agent_evals": {
            "name": "Agent Evaluation",
            "description": "Evaluate AI agents solving math problems with correctness metrics",
            "source_path": "ragas_examples/agent_evals",
        },
        "llamaIndex_agent_evals": {
            "name": "LlamaIndex Agent Evaluation",
            "description": "Evaluate LlamaIndex agents with tool call accuracy metrics",
            "source_path": "ragas_examples/llamaIndex_agent_evals",
        },
        "text2sql": {
            "name": "Text-to-SQL Evaluation",
            "description": "Evaluate text-to-SQL systems with execution accuracy comparison",
            "source_path": "ragas_examples/text2sql",
        },
        "workflow_eval": {
            "name": "Workflow Evaluation",
            "description": "Evaluate complex LLM workflows with email classification and routing",
            "source_path": "ragas_examples/workflow_eval",
        },
        "prompt_evals": {
            "name": "Prompt Evaluation",
            "description": "Evaluate and compare prompt variations with sentiment analysis",
            "source_path": "ragas_examples/prompt_evals",
        },
        "judge_alignment": {
            "name": "Judge Alignment",
            "description": "Measure LLM-as-judge alignment with human evaluation standards",
            "source_path": "ragas_examples/judge_alignment",
        },
        "benchmark_llm": {
            "name": "LLM Benchmarking",
            "description": "Benchmark and compare different LLM models on discount calculation tasks",
            "source_path": "ragas_examples/benchmark_llm",
        },
    }

    # If no template specified, list available templates
    if template is None:
        console.print(
            "\n[bold cyan]Available Ragas Quickstart Templates:[/bold cyan]\n"
        )

        # Create a table of templates
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("Template", style="cyan", no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("Description", style="white")

        for template_id, template_info in templates.items():
            table.add_row(
                template_id, template_info["name"], template_info["description"]
            )

        console.print(table)
        console.print("\n[bold]Usage:[/bold]")
        console.print("  ragas quickstart [template_name]")
        console.print("\n[bold]Example:[/bold]")
        console.print("  ragas quickstart rag_eval")
        console.print("  ragas quickstart rag_eval --output-dir ./my-project\n")
        return

    # Validate template name
    if template not in templates:
        error(f"Unknown template: {template}")
        console.print(f"\nAvailable templates: {', '.join(templates.keys())}")
        console.print("Run 'ragas quickstart' to see all available templates.")
        raise typer.Exit(1)

    template_info = templates[template]
    template_path = template_info["source_path"].replace("ragas_examples/", "")

    # Try to find examples locally first (for development and testing)
    # Look for examples in the installed ragas-examples package or local dev environment
    source_path = None
    temp_dir = None

    try:
        import ragas_examples

        if ragas_examples.__file__ is not None:
            examples_root = Path(ragas_examples.__file__).parent
            local_source = examples_root / template_path
            if local_source.exists():
                source_path = local_source
                info("Using locally installed examples")
    except ImportError:
        pass

    # If not found locally, check if we're in the ragas repository (dev mode)
    if source_path is None:
        # Try to find examples directory relative to this file (development mode)
        cli_file = Path(__file__).resolve()
        repo_root = cli_file.parent.parent.parent  # Go up from src/ragas/cli.py
        local_examples = repo_root / "examples" / "ragas_examples" / template_path
        if local_examples.exists():
            source_path = local_examples
            info("Using local development examples")

    # If still not found, download from GitHub
    if source_path is None:
        import tempfile
        import urllib.request
        import zipfile

        github_repo = "vibrantlabsai/ragas"
        branch = "main"

        # Create temporary directory for download
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Download the specific template folder from GitHub
            archive_url = (
                f"https://github.com/{github_repo}/archive/refs/heads/{branch}.zip"
            )

            zip_path = temp_dir / "repo.zip"
            urllib.request.urlretrieve(archive_url, zip_path)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            extracted_folders = [
                f
                for f in temp_dir.iterdir()
                if f.is_dir() and f.name.startswith("ragas-")
            ]
            if not extracted_folders:
                error("Failed to extract template from GitHub archive")
                raise typer.Exit(1)

            repo_dir = extracted_folders[0]
            source_path = repo_dir / "examples" / "ragas_examples" / template_path

            if not source_path.exists():
                error(f"Template not found in repository: {template_path}")
                console.print(f"Looking for: {source_path}")
                raise typer.Exit(1)

        except Exception as e:
            error(f"Failed to download template from GitHub: {e}")
            console.print("\nYou can also manually clone the repository:")
            console.print(f"  git clone https://github.com/{github_repo}.git")
            console.print(
                f"  cp -r ragas/examples/ragas_examples/{template_path} ./{template}"
            )
            raise typer.Exit(1)

    # Determine output directory
    output_path = Path(output_dir) / template

    if output_path.exists():
        warning(f"Directory already exists: {output_path}")
        overwrite = typer.confirm("Do you want to overwrite it?", default=False)
        if not overwrite:
            info("Operation cancelled.")
            raise typer.Exit(0)
        shutil.rmtree(output_path)

    # Copy the template
    with Live(
        Spinner(
            "dots", text=f"Creating {template_info['name']} project...", style="green"
        ),
        console=console,
    ) as live:
        live.update(Spinner("dots", text="Copying template files...", style="green"))

        # Copy template but exclude .venv and __pycache__
        def ignore_patterns(directory, files):
            return {
                f for f in files if f in {".venv", "__pycache__", "*.pyc", "uv.lock"}
            }

        shutil.copytree(source_path, output_path, ignore=ignore_patterns)
        time.sleep(0.3)

        live.update(
            Spinner("dots", text="Setting up project structure...", style="green")
        )

        evals_dir = output_path / "evals"
        evals_dir.mkdir(exist_ok=True)
        (evals_dir / "datasets").mkdir(exist_ok=True)
        (evals_dir / "experiments").mkdir(exist_ok=True)
        (evals_dir / "logs").mkdir(exist_ok=True)

        datasets_src = output_path / "datasets"
        if datasets_src.exists() and datasets_src.is_dir():
            for item in datasets_src.iterdir():
                if item.is_file():
                    shutil.copy2(item, evals_dir / "datasets" / item.name)
            shutil.rmtree(datasets_src)

        contexts_src = output_path / "contexts"
        if contexts_src.exists() and contexts_src.is_dir():
            shutil.copytree(contexts_src, evals_dir / "datasets" / "contexts")
            shutil.rmtree(contexts_src)

        time.sleep(0.2)

        # Create a README.md with setup instructions
        live.update(Spinner("dots", text="Creating documentation...", style="green"))

        # Template-specific README content
        if template == "improve_rag":
            readme_content = f"""# {template_info["name"]}

{template_info["description"]}

## Quick Start

### 1. Set Your API Key

```bash
export OPENAI_API_KEY="your-openai-key"
```

### 2. Install Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or using `pip`:

```bash
pip install -e .
```

### 3. (Optional) Start MLflow for tracing

```bash
mlflow ui --port 5000
```

### 4. Run the Evaluation

Naive RAG mode (default):

```bash
uv run python evals.py
```

Agentic RAG mode:

```bash
uv run python evals.py --agentic
```

## Project Structure

```
{template}/
├── README.md           # This file
├── pyproject.toml      # Project configuration
├── rag.py              # RAG implementation (naive & agentic modes)
├── evals.py            # Evaluation workflow
├── __init__.py         # Makes this a Python package
└── evals/              # Evaluation-related data
    ├── datasets/       # Test datasets (hf_doc_qa_eval.csv)
    ├── experiments/    # Experiment results
    └── logs/           # Evaluation logs
```

## Features

- **Naive RAG**: Single retrieval + generation
- **Agentic RAG**: Agent-controlled retrieval with multiple searches
- **BM25 Retrieval**: Uses HuggingFace documentation as knowledge base
- **MLflow Tracing**: Automatic tracing of all LLM calls

## Documentation

Visit https://docs.ragas.io for more information.
"""
        else:
            readme_content = f"""# {template_info["name"]}

{template_info["description"]}

## Quick Start

### 1. Set Your API Key

Choose your LLM provider:

```bash
# OpenAI (default)
export OPENAI_API_KEY="your-openai-key"

# Or use Anthropic Claude
export ANTHROPIC_API_KEY="your-anthropic-key"

# Or use Google Gemini
export GOOGLE_API_KEY="your-google-key"
```

### 2. Install Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or using `pip`:

```bash
pip install -e .
```

### 3. Run the Evaluation

Using `uv`:

```bash
uv run python evals.py
```

Or using `pip`:

```bash
python evals.py
```

## Project Structure

```
{template}/
├── README.md           # This file
├── pyproject.toml      # Project configuration
├── rag.py              # Your RAG application code
├── evals.py            # Evaluation workflow
├── __init__.py         # Makes this a Python package
└── evals/              # Evaluation-related data
    ├── datasets/       # Test datasets
    ├── experiments/    # Experiment results
    └── logs/           # Evaluation logs and traces
```

## Customization

### Modify the LLM Provider

In `evals.py`, update the LLM configuration:

```python
from ragas.llms import llm_factory

# Use Anthropic Claude
llm = llm_factory("claude-3-5-sonnet-20241022", provider="anthropic")

# Use Google Gemini
llm = llm_factory("gemini-1.5-pro", provider="google")

# Use local Ollama
llm = llm_factory("mistral", provider="ollama", base_url="http://localhost:11434")
```

### Customize Test Cases

Edit the `load_dataset()` function in `evals.py` to add or modify test cases.

### Change Evaluation Metrics

Update the `my_metric` definition in `evals.py` to use different grading criteria.

## Documentation

Visit https://docs.ragas.io for more information.
"""

        readme_path = output_path / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        time.sleep(0.2)

        # live.update(Spinner("dots", text="Finalizing project...", style="green"))
        time.sleep(0.3)

    # Cleanup temporary directory if we downloaded from GitHub
    if temp_dir is not None:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    # Success message with next steps
    success(f"\n✓ Created {template_info['name']} project at: {output_path}")
    console.print("\n[bold cyan]Next Steps:[/bold cyan]")
    console.print(f"  cd {output_path}")
    console.print("  uv sync")
    console.print("  export OPENAI_API_KEY='your-api-key'")
    console.print("  uv run python evals.py")
    console.print("\n📚 For detailed instructions, see:")
    console.print("  https://docs.ragas.io/en/latest/getstarted/quickstart/\n")


@app.command()
def hello_world(
    directory: str = typer.Argument(
        ".", help="Directory to run the hello world example in"
    ),
):
    import os
    import time

    import pandas as pd

    if not os.path.exists(directory):
        console.print(f"Directory {directory} does not exist.", style="red")
        raise typer.Exit(1)

    with Live(
        Spinner("dots", text="Creating hello world example...", style="green"),
        console=console,
    ) as live:
        live.update(Spinner("dots", text="Creating directories...", style="green"))
        Path(directory).joinpath("hello_world").mkdir(parents=True, exist_ok=True)
        os.makedirs(os.path.join(directory, "hello_world", "datasets"), exist_ok=True)
        os.makedirs(
            os.path.join(directory, "hello_world", "experiments"), exist_ok=True
        )
        time.sleep(0.5)  # Brief pause to show spinner

        live.update(Spinner("dots", text="Creating test dataset...", style="green"))
        hello_world_data = [
            {
                "id": 1,
                "query": "What is the capital of France?",
                "expected_output": "Paris",
            },
            {"id": 2, "query": "What is 2 + 2?", "expected_output": "4"},
            {
                "id": 3,
                "query": "What is the largest mammal?",
                "expected_output": "Blue Whale",
            },
            {
                "id": 4,
                "query": "Who developed the theory of relativity?",
                "expected_output": "Einstein",
            },
            {
                "id": 5,
                "query": "What is the programming language used for data science?",
                "expected_output": "Python",
            },
            {
                "id": 6,
                "query": "What is the highest mountain in the world?",
                "expected_output": "Mount Everest",
            },
            {
                "id": 7,
                "query": "Who wrote 'Romeo and Juliet'?",
                "expected_output": "Shakespeare",
            },
            {
                "id": 8,
                "query": "What is the fourth planet from the Sun?",
                "expected_output": "Mars",
            },
            {
                "id": 9,
                "query": "What is the name of the fruit that keeps the doctor away?",
                "expected_output": "Apple",
            },
            {
                "id": 10,
                "query": "Who painted the Mona Lisa?",
                "expected_output": "Leonardo da Vinci",
            },
        ]
        df = pd.DataFrame(hello_world_data)
        df.to_csv(
            os.path.join(directory, "hello_world", "datasets", "test_data.csv"),
            index=False,
        )
        time.sleep(0.5)  # Brief pause to show spinner

        live.update(
            Spinner("dots", text="Creating evaluation script...", style="green")
        )
        # Create evals.py file
        evals_content = '''import typing as t

import numpy as np
from pydantic import BaseModel
# from ragas.experimental.project.backends import LocalCSVProjectBackend  # TODO: Not implemented yet
from ragas.metrics.result import MetricResult
from ragas.metrics.numeric import numeric_metric

# TODO: Project class not implemented yet  
# p = Project(
#     project_id="hello_world", 
#     project_backend=LocalCSVProjectBackend("."),
# )


@numeric_metric(name="accuracy_score", allowed_values=(0, 1))
def accuracy_score(response: str, expected: str):
    """
    Is the response a good response to the query?
    """
    result = 1 if expected.lower().strip() == response.lower().strip() else 0
    return MetricResult(
        result=result,
        reason=(
            f"Response contains {expected}"
            if result
            else f"Response does not contain {expected}"
        ),
    )


def mock_app_endpoint(**kwargs) -> str:
    """Mock AI endpoint for testing purposes."""
    mock_responses = [
        "Paris","4","Blue Whale","Einstein","Python","Mount Everest","Shakespeare",
        "Mars","Apple","Leonardo da Vinci",]
    return np.random.choice(mock_responses)


class TestDataRow(BaseModel):
    id: t.Optional[int]
    query: str
    expected_output: str


class ExperimentDataRow(TestDataRow):
    response: str
    accuracy: int
    accuracy_reason: t.Optional[str] = None


# @p.experiment(ExperimentDataRow)  # TODO: Project not implemented
async def run_experiment(row: TestDataRow):
    response = mock_app_endpoint(query=row.query)
    accuracy = accuracy_score.score(response=response, expected=row.expected_output)

    experiment_view = ExperimentDataRow(
        **row.model_dump(),
        response=response,
        accuracy=accuracy.result,
        accuracy_reason=accuracy.reason,
    )
    return experiment_view
'''

        evals_path = os.path.join(directory, "hello_world", "evals.py")
        with open(evals_path, "w", encoding="utf-8") as f:
            f.write(evals_content)
        time.sleep(0.5)  # Brief pause to show spinner

        live.update(Spinner("dots", text="Finalizing hello world example..."))
        time.sleep(0.5)  # Brief pause to show spinner

    hello_world_path = os.path.join(directory, "hello_world")
    success(f"✓ Created hello world example in {hello_world_path}")
    success(
        "✓ You can now run: ragas evals hello_world/evals.py --dataset test_data --metrics accuracy"
    )


if __name__ == "__main__":
    app()
