"""
AG-UI Agent Experiment Script

This script demonstrates how to run experiments on agents built with the AG-UI protocol
using Ragas metrics with the modern @experiment decorator pattern.

It includes two experiment scenarios:

1. Scientist Biographies (Single-turn) - Tests factual correctness and answer relevancy
2. Weather Tool Usage (Multi-turn) - Tests tool calling accuracy and agent goal achievement

Metrics used:
- FactualCorrectness: Measures factual accuracy of responses
- AnswerRelevancy: Measures how relevant the response is to the question
- ToolCallF1: Rule-based metric for tool call accuracy
- AgentGoalAccuracyWithReference: LLM-based metric for whether the agent achieved the user's goal

Prerequisites:
- An AG-UI compatible agent running at the specified endpoint URL
- See https://docs.ag-ui.com/quickstart/applications for agent setup

Usage:
    python experiments.py --endpoint-url http://localhost:8000/chat
    python experiments.py --endpoint-url http://localhost:8000/chat --skip-tool-experiment
    python experiments.py --endpoint-url http://localhost:8000 --skip-factual
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.dataset import Dataset
from ragas.embeddings.base import embedding_factory
from ragas.experiment import experiment
from ragas.integrations.ag_ui import run_ag_ui_row
from ragas.llms import llm_factory
from ragas.messages import ToolCall
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import (
    AgentGoalAccuracyWithReference,
    AnswerRelevancy,
    FactualCorrectness,
    ToolCallF1,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
load_dotenv(REPO_ROOT / ".env")
TEST_DATA_DIR = SCRIPT_DIR / "test_data"


def load_scientist_dataset() -> Dataset:
    """
    Load the scientist biographies dataset from CSV.

    Returns:
        Dataset with entries for testing factual correctness.
    """
    csv_path = TEST_DATA_DIR / "scientist_biographies.csv"
    logger.info(f"Loading scientist biographies dataset from {csv_path}")

    dataset = Dataset.load(
        name="scientist_biographies",
        backend="local/csv",
        root_dir=str(TEST_DATA_DIR),
    )

    logger.info(f"Loaded {len(dataset)} scientist biography samples")
    return dataset


def load_weather_dataset() -> Dataset:
    """
    Load the weather tool call dataset from CSV.

    Returns:
        Dataset with entries for testing tool call accuracy and agent goal accuracy.
    """
    csv_path = TEST_DATA_DIR / "weather_tool_calls.csv"
    logger.info(f"Loading weather tool call dataset from {csv_path}")

    dataset = Dataset.load(
        name="weather_tool_calls",
        backend="local/csv",
        root_dir=str(TEST_DATA_DIR),
    )

    logger.info(f"Loaded {len(dataset)} weather tool call samples")
    return dataset


def create_evaluator_components(model_name: str):
    """Instantiate a fresh evaluator LLM and embeddings for the current loop."""

    llm_client = AsyncOpenAI()
    evaluator_llm = llm_factory(model_name, client=llm_client, max_tokens=6000)
    setattr(evaluator_llm, "is_async", True)
    embedding_client = AsyncOpenAI()
    evaluator_embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=embedding_client,
        interface="modern",
    )
    return evaluator_llm, evaluator_embeddings


async def run_scientist_experiment(
    endpoint_url: str, evaluator_model: str
) -> tuple:
    """
    Run an experiment to test the agent's ability to provide factually correct
    information about scientists using the @experiment pattern.

    Args:
        endpoint_url: The AG-UI endpoint URL
        evaluator_model: The evaluator LLM model name

    Returns:
        Tuple of (experiment_result, dataframe) where experiment_result is the Experiment
        and dataframe is the pandas DataFrame with results.
    """
    logger.info("=" * 80)
    logger.info("Starting Scientist Biographies Experiment")
    logger.info("=" * 80)

    # Load dataset
    dataset = load_scientist_dataset()

    # Create evaluator components
    evaluator_llm, evaluator_embeddings = create_evaluator_components(evaluator_model)

    # Define metrics using the modern collections portfolio
    factual_correctness = FactualCorrectness(
        llm=evaluator_llm, mode="f1", atomicity="high", coverage="high"
    )
    answer_relevancy = AnswerRelevancy(
        llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=2
    )
    conciseness_metric = DiscreteMetric(
        name="conciseness",
        allowed_values=["verbose", "concise"],
        prompt=(
            "Is the response concise and efficiently conveys information?\n\n"
            "Response: {response}\n\n"
            "Answer with only 'verbose' or 'concise'."
        ),
    )

    @experiment()
    async def scientist_experiment(row):
        """Single-turn Q&A experiment with factual correctness scoring."""
        # Call AG-UI endpoint and get enriched row
        enriched = await run_ag_ui_row(row, endpoint_url, timeout=300.0)

        # Score with factual correctness metric
        fc_result = await factual_correctness.ascore(
            response=enriched["response"],
            reference=row["reference"],
        )

        # Score with answer relevancy metric
        ar_result = await answer_relevancy.ascore(
            user_input=row["user_input"],
            response=enriched["response"],
        )

        # Score with conciseness metric
        concise_result = await conciseness_metric.ascore(
            response=enriched["response"],
            llm=evaluator_llm,
        )

        return {
            **enriched,
            "factual_correctness": fc_result.value,
            "answer_relevancy": ar_result.value,
            "conciseness": concise_result.value,
        }

    # Run evaluation using @experiment pattern
    logger.info(f"Evaluating against endpoint: {endpoint_url}")
    result = await scientist_experiment.arun(dataset, name="scientist_biographies_eval")

    # Convert to DataFrame for analysis
    df = result.to_pandas()

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Scientist Biographies Experiment Results")
    logger.info("=" * 80)
    logger.info(f"\nDataFrame shape: {df.shape}")
    logger.info(f"\n{df.to_string()}")

    metric_columns = [
        "factual_correctness",
        "answer_relevancy",
    ]
    for column in metric_columns:
        if column in df.columns:
            logger.info(f"Average {column}: {df[column].mean():.4f}")

    if "factual_correctness" in df.columns:
        logger.info(
            f"Perfect factual scores (1.0): {(df['factual_correctness'] == 1.0).sum()}/{len(df)}"
        )
    if "conciseness" in df.columns:
        concise_ratio = (df["conciseness"] == "concise").mean()
        logger.info(f"Concise responses: {concise_ratio:.2%}")

    return result, df


async def run_tool_experiment(endpoint_url: str, evaluator_model: str) -> tuple:
    """
    Run an experiment to test the agent's ability to correctly call the weather tool
    and achieve the user's goal using the @experiment pattern.

    Args:
        endpoint_url: The AG-UI endpoint URL
        evaluator_model: The evaluator LLM model name

    Returns:
        Tuple of (experiment_result, dataframe) where experiment_result is the Experiment
        and dataframe is the pandas DataFrame with results.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Starting Weather Tool Usage Experiment")
    logger.info("=" * 80)

    # Load dataset
    dataset = load_weather_dataset()

    # Create evaluator LLM for goal accuracy metric
    evaluator_llm, _ = create_evaluator_components(evaluator_model)

    # Define metrics:
    # - ToolCallF1: Rule-based metric for tool call accuracy
    # - AgentGoalAccuracyWithReference: LLM-based metric for goal achievement
    #   Note: This metric has some variance due to LLM non-determinism
    tool_call_f1 = ToolCallF1()
    goal_accuracy = AgentGoalAccuracyWithReference(llm=evaluator_llm)

    @experiment()
    async def tool_experiment(row):
        """Multi-turn experiment with tool call and goal accuracy scoring."""
        # Call AG-UI endpoint and get enriched row
        enriched = await run_ag_ui_row(row, endpoint_url, timeout=300.0)

        # Parse reference_tool_calls from JSON string (e.g., from CSV)
        ref_tool_calls_raw = row.get("reference_tool_calls")
        if isinstance(ref_tool_calls_raw, str):
            ref_tool_calls = [
                ToolCall(**tc) for tc in json.loads(ref_tool_calls_raw)
            ]
        else:
            ref_tool_calls = ref_tool_calls_raw or []

        # Score with tool metrics using the modern collections API
        f1_result = await tool_call_f1.ascore(
            user_input=enriched["messages"],
            reference_tool_calls=ref_tool_calls,
        )
        goal_result = await goal_accuracy.ascore(
            user_input=enriched["messages"],
            reference=row.get("reference", ""),
        )

        return {
            **enriched,
            "tool_call_f1": f1_result.value,
            "agent_goal_accuracy": goal_result.value,
        }

    # Run evaluation using @experiment pattern
    logger.info(f"Evaluating against endpoint: {endpoint_url}")
    result = await tool_experiment.arun(dataset, name="weather_tool_calls_eval")

    # Convert to DataFrame for analysis
    df = result.to_pandas()

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Weather Tool Usage Experiment Results")
    logger.info("=" * 80)
    logger.info(f"\nDataFrame shape: {df.shape}")
    logger.info(f"\n{df.to_string()}")

    if "tool_call_f1" in df.columns:
        avg_f1 = df["tool_call_f1"].mean()
        logger.info(f"\nAverage Tool Call F1: {avg_f1:.4f}")
        logger.info(
            f"Perfect scores (F1=1.0): {(df['tool_call_f1'] == 1.0).sum()}/{len(df)}"
        )
        logger.info(
            f"Failed scores (F1=0.0): {(df['tool_call_f1'] == 0.0).sum()}/{len(df)}"
        )

    if "agent_goal_accuracy" in df.columns:
        avg_goal = df["agent_goal_accuracy"].mean()
        logger.info(f"\nAverage Agent Goal Accuracy: {avg_goal:.4f}")
        logger.info(
            f"Goals achieved (1.0): {(df['agent_goal_accuracy'] == 1.0).sum()}/{len(df)}"
        )

    return result, df


async def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run AG-UI agent experiments using Ragas metrics with @experiment pattern"
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default="http://localhost:8000",
        help="AG-UI endpoint URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--evaluator-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for experiments (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--skip-factual",
        action="store_true",
        help="Skip the factual correctness experiment",
    )
    parser.add_argument(
        "--skip-tool-experiment",
        action="store_true",
        help="Skip the tool call experiment",
    )

    args = parser.parse_args()

    # Sanity check the embedding endpoint before experiments
    async def sanity_check():
        sanity_client = AsyncOpenAI()
        logger.info("Running embeddings sanity check before experiments")
        try:
            await sanity_client.embeddings.create(
                input="Sanity check",
                model="text-embedding-3-small",
                timeout=10.0,
            )
            logger.info("Embeddings sanity check succeeded")
        except Exception as exc:
            logger.warning("Embeddings sanity check failed: %s", exc)

    await sanity_check()

    # Run experiments
    try:
        if not args.skip_factual:
            result, df = await run_scientist_experiment(
                args.endpoint_url, args.evaluator_model
            )
            logger.info(f"\nResults saved to: {result.name}")

        if not args.skip_tool_experiment:
            result, df = await run_tool_experiment(
                args.endpoint_url, args.evaluator_model
            )
            logger.info(f"\nResults saved to: {result.name}")

        logger.info("\n" + "=" * 80)
        logger.info("All experiments completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\nExperiment failed with error: {e}")
        logger.error(
            "\nPlease ensure your AG-UI agent is running at the specified endpoint."
        )
        logger.error(
            "See https://docs.ag-ui.com/quickstart/applications for setup instructions."
        )
        raise


if __name__ == "__main__":
    asyncio.run(main())
