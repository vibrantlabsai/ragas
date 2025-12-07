"""
AG-UI Agent Evaluation Script

This script demonstrates how to evaluate agents built with the AG-UI protocol
using Ragas metrics. It includes two evaluation scenarios:

1. Scientist Biographies (Single-turn) - Tests factual correctness and answer relevancy
2. Weather Tool Usage (Multi-turn) - Tests tool calling accuracy and agent goal achievement

Metrics used:
- FactualCorrectness: Measures factual accuracy of responses
- AnswerRelevancy: Measures how relevant the response is to the question
- ToolCallF1: Rule-based metric for tool call accuracy
- AgentGoalAccuracyWithReference: LLM-based metric evaluating whether the agent achieved the user's goal

Prerequisites:
- An AG-UI compatible agent running at the specified endpoint URL
- See https://docs.ag-ui.com/quickstart/applications for agent setup

Usage:
    python evals.py --endpoint-url http://localhost:8000/agentic_chat
    python evals.py --endpoint-url http://localhost:8000/chat --skip-tool-eval
    python evals.py --endpoint-url http://localhost:8000 --skip-factual
"""

import argparse
import asyncio
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from ragas.dataset_schema import (
    EvaluationDataset,
    MultiTurnSample,
    SingleTurnSample,
)
from ragas.embeddings.base import embedding_factory
from ragas.integrations.ag_ui import evaluate_ag_ui_agent
from ragas.llms import llm_factory
from ragas.messages import HumanMessage, ToolCall
from ragas.metrics import (
    AgentGoalAccuracyWithReference,
    DiscreteMetric,
    ToolCallF1,
)
from ragas.metrics.collections import AnswerRelevancy, FactualCorrectness
from ragas.run_config import RunConfig

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


def load_scientist_dataset() -> EvaluationDataset:
    """
    Load the scientist biographies dataset from CSV.

    Returns:
        EvaluationDataset with SingleTurnSample entries for testing factual correctness.
    """
    csv_path = TEST_DATA_DIR / "scientist_biographies.csv"
    logger.info(f"Loading scientist biographies dataset from {csv_path}")

    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = SingleTurnSample(
                user_input=row["user_input"], reference=row["reference"]
            )
            samples.append(sample)

    logger.info(f"Loaded {len(samples)} scientist biography samples")
    return EvaluationDataset(samples=samples)


def load_weather_dataset() -> EvaluationDataset:
    """
    Load the weather tool call dataset from CSV.

    Returns:
        EvaluationDataset with MultiTurnSample entries for testing tool call accuracy
        and agent goal accuracy.
    """
    csv_path = TEST_DATA_DIR / "weather_tool_calls.csv"
    logger.info(f"Loading weather tool call dataset from {csv_path}")

    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse the reference_tool_calls JSON
            tool_calls_data = json.loads(row["reference_tool_calls"])
            tool_calls = [
                ToolCall(name=tc["name"], args=tc["args"]) for tc in tool_calls_data
            ]

            # Create MultiTurnSample with user_input as a list of HumanMessage
            # Include reference for AgentGoalAccuracyWithReference metric
            sample = MultiTurnSample(
                user_input=[HumanMessage(content=row["user_input"])],
                reference_tool_calls=tool_calls,
                reference=row.get("reference"),  # Expected outcome for goal accuracy
            )
            samples.append(sample)

    logger.info(f"Loaded {len(samples)} weather tool call samples")
    return EvaluationDataset(samples=samples)


def create_evaluator_components(model_name: str):
    """Instantiate a fresh evaluator LLM and embeddings for the current loop."""

    llm_client = AsyncOpenAI()
    evaluator_llm = llm_factory(model_name, client=llm_client, max_tokens=6000)
    setattr(evaluator_llm, "is_async", True)
    embedding_client = OpenAI()
    evaluator_embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=embedding_client,
        interface="modern",
    )
    return evaluator_llm, evaluator_embeddings


async def evaluate_scientist_biographies(
    endpoint_url: str, evaluator_model: str
) -> tuple:
    """
    Evaluate the agent's ability to provide factually correct information
    about scientists.

    Args:
        endpoint_url: The AG-UI endpoint URL
        evaluator_model: The evaluator LLM model name

    Returns:
        Tuple of (result, dataframe) where result is the EvaluationResult
        and dataframe is the pandas DataFrame with results.
    """
    logger.info("=" * 80)
    logger.info("Starting Scientist Biographies Evaluation")
    logger.info("=" * 80)

    # Load dataset
    dataset = load_scientist_dataset()

    # Define metrics using the modern collections portfolio
    evaluator_llm, evaluator_embeddings = create_evaluator_components(
        evaluator_model
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
    metrics = [
        FactualCorrectness(
            llm=evaluator_llm, mode="f1", atomicity="high", coverage="high"
        ),
        AnswerRelevancy(
            llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=2
        ),
    ]

    # Run evaluation
    logger.info(f"Evaluating against endpoint: {endpoint_url}")
    run_config = RunConfig(max_workers=10, timeout=300)
    result = await evaluate_ag_ui_agent(
        endpoint_url=endpoint_url,
        dataset=dataset,
        metrics=metrics,
        evaluator_llm=evaluator_llm,
        run_config=run_config,
    )

    # Convert to DataFrame and clean up
    df = result.to_pandas()
    df = df.drop(columns=["retrieved_contexts"], errors="ignore")

    if "response" in df.columns:
        conciseness_scores = []
        for response_text in df["response"].fillna(""):
            conciseness_result = await conciseness_metric.ascore(
                response=response_text,
                llm=evaluator_llm,
            )
            conciseness_scores.append(conciseness_result.value)
        df["conciseness"] = conciseness_scores

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Scientist Biographies Evaluation Results")
    logger.info("=" * 80)
    logger.info(f"\nDataFrame shape: {df.shape}")
    logger.info(f"\n{df.to_string()}")

    metric_columns = [
        "factual_correctness(mode=f1)",
        "answer_relevancy",
    ]
    for column in metric_columns:
        if column in df.columns:
            logger.info(f"Average {column}: {df[column].mean():.4f}")

    if "factual_correctness(mode=f1)" in df.columns:
        logger.info(
            f"Perfect factual scores (1.0): {(df['factual_correctness(mode=f1)'] == 1.0).sum()}/{len(df)}"
        )
    if "conciseness" in df.columns:
        concise_ratio = (df["conciseness"] == "concise").mean()
        logger.info(f"Concise responses: {concise_ratio:.2%}")

    return result, df


async def evaluate_weather_tool_use(endpoint_url: str, evaluator_model: str) -> tuple:
    """
    Evaluate the agent's ability to correctly call the weather tool and achieve
    the user's goal.

    Args:
        endpoint_url: The AG-UI endpoint URL
        evaluator_model: The evaluator LLM model name

    Returns:
        Tuple of (result, dataframe) where result is the EvaluationResult
        and dataframe is the pandas DataFrame with results.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Starting Weather Tool Usage Evaluation")
    logger.info("=" * 80)

    # Load dataset
    dataset = load_weather_dataset()

    # Create evaluator LLM for goal accuracy metric
    evaluator_llm, _ = create_evaluator_components(evaluator_model)

    # Define metrics:
    # - ToolCallF1: Rule-based metric for tool call accuracy
    # - AgentGoalAccuracyWithReference: LLM-based metric for goal achievement
    #   Note: This metric has some variance due to LLM non-determinism
    metrics = [
        ToolCallF1(),
        AgentGoalAccuracyWithReference(llm=evaluator_llm),
    ]

    # Run evaluation
    logger.info(f"Evaluating against endpoint: {endpoint_url}")
    run_config = RunConfig(max_workers=10, timeout=300)
    result = await evaluate_ag_ui_agent(
        endpoint_url=endpoint_url,
        dataset=dataset,
        metrics=metrics,
        evaluator_llm=evaluator_llm,
        run_config=run_config,
    )

    # Convert to DataFrame and clean up
    df = result.to_pandas()
    columns_to_drop = [
        col for col in ["retrieved_contexts"] if col in df.columns
    ]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Weather Tool Usage Evaluation Results")
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


def save_results(df, scenario_name: str, output_dir: Path = None):
    """
    Save evaluation results to a timestamped CSV file.

    Args:
        df: The pandas DataFrame with evaluation results
        scenario_name: Name of the evaluation scenario
        output_dir: Directory to save results (defaults to script directory)
    """
    if output_dir is None:
        output_dir = SCRIPT_DIR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{scenario_name}_results_{timestamp}.csv"
    filepath = output_dir / filename

    df.to_csv(filepath, index=False)
    logger.info(f"\nResults saved to: {filepath}")


async def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate AG-UI agents using Ragas metrics"
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default="http://localhost:8000/agentic_chat",
        help="AG-UI endpoint URL (default: http://localhost:8000/agentic_chat)",
    )
    parser.add_argument(
        "--evaluator-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for evaluation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--skip-factual",
        action="store_true",
        help="Skip the factual correctness evaluation",
    )
    parser.add_argument(
        "--skip-tool-eval",
        action="store_true",
        help="Skip the tool call evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save results (default: script directory)",
    )

    args = parser.parse_args()

    # Sanity check the embedding endpoint before evaluation
    async def sanity_check():
        sanity_client = AsyncOpenAI()
        logger.info("Running embeddings sanity check before evaluation")
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

    # Run evaluations
    try:
        if not args.skip_factual:
            result, df = await evaluate_scientist_biographies(
                args.endpoint_url, args.evaluator_model
            )
            save_results(df, "scientist_biographies", args.output_dir)

        if not args.skip_tool_eval:
            result, df = await evaluate_weather_tool_use(
                args.endpoint_url, args.evaluator_model
            )
            save_results(df, "weather_tool_calls", args.output_dir)

        logger.info("\n" + "=" * 80)
        logger.info("All evaluations completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\nEvaluation failed with error: {e}")
        logger.error(
            "\nPlease ensure your AG-UI agent is running at the specified endpoint."
        )
        logger.error(
            "See https://docs.ag-ui.com/quickstart/applications for setup instructions."
        )
        raise


if __name__ == "__main__":
    asyncio.run(main())
