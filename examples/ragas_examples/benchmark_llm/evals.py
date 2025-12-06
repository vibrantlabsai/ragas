import argparse
import datetime
import json
import os
import sys
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")

from ragas import experiment
from ragas.dataset import Dataset
from ragas.metrics.discrete import discrete_metric
from ragas.metrics.result import MetricResult

from .prompt import DEFAULT_MODEL, run_prompt


@discrete_metric(name="discount_accuracy", allowed_values=["correct", "incorrect"])
def discount_accuracy(prediction: str, expected_discount):
    """Check if the discount prediction is correct."""
    parsed_json = json.loads(prediction)
    predicted_discount = parsed_json.get("discount_percentage")
    expected_discount_int = int(expected_discount)

    if predicted_discount == expected_discount_int:
        return MetricResult(
            value="correct",
            reason=f"Correctly calculated discount={expected_discount_int}%",
        )
    else:
        return MetricResult(
            value="incorrect",
            reason=f"Expected discount={expected_discount_int}%; Got discount={predicted_discount}%",
        )


@experiment()
async def benchmark_experiment(row, model_name: str):
    """Benchmark experiment function that evaluates a model on discount calculation."""
    # Get model response
    response = await run_prompt(row["customer_profile"], model=model_name)

    # Parse response (strict JSON mode expected)
    try:
        parsed_json = json.loads(response)
        predicted_discount = parsed_json.get("discount_percentage")
    except Exception:
        predicted_discount = None

    # Score the response
    score = discount_accuracy.score(
        prediction=response, expected_discount=row["expected_discount"]
    )

    return {
        **row,
        "model": model_name,
        "response": response,
        "predicted_discount": predicted_discount,
        "score": score.value,
        "score_reason": score.reason,
    }


def load_dataset():
    """Load the dataset from CSV file. Downloads from GitHub if not found locally."""
    import urllib.request
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "datasets", "discount_benchmark.csv")
    # Download dataset from GitHub if it doesn't exist locally
    if not os.path.exists(dataset_path):
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        urllib.request.urlretrieve("https://raw.githubusercontent.com/vibrantlabsai/ragas/main/examples/ragas_examples/benchmark_llm/datasets/discount_benchmark.csv", dataset_path)
    return Dataset.load(name="discount_benchmark", backend="local/csv", root_dir=current_dir)


def compare_inputs_to_output(
    inputs: List[str], output_path: Optional[str] = None
) -> str:
    """Compare multiple experiment CSVs and write a combined CSV.

    - Requires 'id' column in all inputs; uses it as the alignment key
    - Builds output with id + canonical columns + per-experiment response/score/reason columns
    - Returns the full output path
    """
    if not inputs or len(inputs) < 2:
        raise ValueError("At least two input CSV files are required for comparison")

    # Load all inputs
    dataframes = []
    experiment_names = []
    for path in inputs:
        df = pd.read_csv(path)
        if "model" not in df.columns:
            raise ValueError(f"Missing 'model' column in {path}")
        exp_name = str(df["model"].iloc[0])
        experiment_names.append(exp_name)
        dataframes.append(df)

    canonical_cols = ["customer_profile", "description", "expected_discount"]
    base_df = dataframes[0]

    # Require 'id' in all inputs
    if not all("id" in df.columns for df in dataframes):
        raise ValueError(
            "All input CSVs must contain an 'id' column to align rows. Re-run experiments after adding 'id' to your dataset."
        )

    # Validate duplicates and matching sets of IDs
    key_sets = []
    for idx, df in enumerate(dataframes):
        keys = df["id"].astype(str)
        if keys.duplicated().any():
            dupes = keys[keys.duplicated()].head(3).tolist()
            raise ValueError(
                f"Input {inputs[idx]} contains duplicate id values. Examples: {dupes}"
            )
        key_sets.append(set(keys.tolist()))

    base_keys = key_sets[0]
    for i, ks in enumerate(key_sets[1:], start=1):
        if ks != base_keys:
            missing_in_other = list(base_keys - ks)[:5]
            missing_in_base = list(ks - base_keys)[:5]
            raise ValueError(
                "Inputs do not contain the same set of IDs.\n"
                f"- Missing in file {i + 1}: {missing_in_other}\n"
                f"- Extra in file {i + 1}: {missing_in_base}"
            )

    # Validate canonical columns exist in base
    missing = [c for c in canonical_cols if c not in base_df.columns]
    if missing:
        raise ValueError(f"First CSV missing required columns: {missing}")

    # Build combined on base order using 'id' as alignment key
    base_ids_str = base_df["id"].astype(str)
    combined = base_df[["id"] + canonical_cols].copy()

    # Append per-experiment outputs by aligned ID
    for df, exp_name in zip(dataframes, experiment_names):
        df = df.copy()
        df["id"] = df["id"].astype(str)
        df = df.set_index("id")
        for col in ["response", "score", "score_reason"]:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in one input. Please provide per-row '{col}'."
                )
        combined[f"{exp_name}_response"] = base_ids_str.map(df["response"])
        combined[f"{exp_name}_score"] = base_ids_str.map(df["score"])
        combined[f"{exp_name}_score_reason"] = base_ids_str.map(df["score_reason"])

    # Determine output path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(current_dir, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)

    if output_path is None or output_path.strip() == "":
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(experiments_dir, f"{run_id}-comparison.csv")
    else:
        # If relative path, place under experiments dir
        if not os.path.isabs(output_path):
            output_path = os.path.join(experiments_dir, output_path)

    # Sort by id for user-friendly reading
    if "id" in combined.columns:
        combined = combined.sort_values(by="id").reset_index(drop=True)
    combined.to_csv(output_path, index=False)

    # Print per-experiment accuracy summary
    for df, exp_name in zip(dataframes, experiment_names):
        try:
            acc = (df["score"] == "correct").mean()
            print(f"{exp_name} Accuracy: {acc:.2%}")
        except Exception:
            pass

    return output_path


async def run_command(model: str, name: Optional[str]) -> None:
    """Run a single experiment using the provided model and name."""
    if "OPENAI_API_KEY" not in os.environ:
        print("❌ Error: OpenAI API key not found!")
        print("Please set your API key: export OPENAI_API_KEY=your_actual_key")
        return

    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Dataset loaded with {len(dataset)} samples")

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = name or model

    # Ensure output directory exists (experiment framework saves under experiments/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(current_dir, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)

    print(f"Running model evaluation ({model})...")
    results = await benchmark_experiment.arun(
        dataset, 
        name=f"{run_id}-{exp_name}",
        model_name=model
    )
    print(f"✅ {exp_name}: {len(results)} cases evaluated")
    print(f"Results saved to: {os.path.join(experiments_dir, results.name)}.csv")

    # Accuracy summary
    accuracy = sum(1 for r in results if r["score"] == "correct") / max(1, len(results))
    print(f"{exp_name} Accuracy: {accuracy:.2%}")


def compare_command(inputs: List[str], output: Optional[str]) -> None:
    output_path = compare_inputs_to_output(inputs, output)
    print(f"Combined comparison saved to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark LLM evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run a single experiment")
    run_parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="Model name to evaluate"
    )
    run_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (defaults to model name)",
    )

    # compare subcommand
    cmp_parser = subparsers.add_parser(
        "compare", help="Combine multiple experiment CSVs"
    )
    cmp_parser.add_argument(
        "--inputs", nargs="+", required=True, help="Input CSV files to compare"
    )
    cmp_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (defaults to experiments/<timestamp>-comparison.csv)",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        import asyncio

        asyncio.run(run_command(model=args.model, name=args.name))
        sys.exit(0)
    elif args.command == "compare":
        compare_command(inputs=args.inputs, output=args.output)
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(2)
