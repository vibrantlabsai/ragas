"""
Evaluation script for unified RAG system using HuggingFace documentation Q&A dataset.
This evaluates both naive and agentic RAG modes against a ground truth dataset.

The script creates a BM25Retriever and uses it with the RAG system for evaluation.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric

from .rag import RAG, BM25Retriever

# Load environment variables
load_dotenv(".env")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress HTTP request logs from OpenAI/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

def download_and_save_dataset() -> Path:
    """Download the HuggingFace doc Q&A dataset from GitHub."""
    dataset_path = Path("datasets/hf_doc_qa_eval.csv")
    dataset_path.parent.mkdir(exist_ok=True)
    
    if dataset_path.exists():
        logger.info(f"Dataset already exists at {dataset_path}")
        return dataset_path
    
    logger.info("Downloading HuggingFace doc Q&A evaluation dataset from GitHub...")
    github_url = "https://raw.githubusercontent.com/vibrantlabsai/ragas/main/examples/ragas_examples/improve_rag/datasets/hf_doc_qa_eval.csv"
    
    import urllib.request
    
    try:
        urllib.request.urlretrieve(github_url, dataset_path)
        logger.info(f"Dataset downloaded to {dataset_path}")
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise
    
    return dataset_path


def create_ragas_dataset(dataset_path: Path) -> Dataset:
    """Create a Ragas Dataset from the downloaded CSV file."""
    dataset = Dataset(name="hf_doc_qa_eval", backend="local/csv", root_dir=".")
    
    import pandas as pd
    df = pd.read_csv(dataset_path)
    
    for _, row in df.iterrows():
        dataset.append({"question": row["question"], "expected_answer": row["expected_answer"]})
    
    dataset.save()
    logger.info(f"Created Ragas dataset with {len(df)} samples")
    return dataset




def construct_mlflow_trace_url(trace_id: str, mlflow_host: str = "http://127.0.0.1:5000") -> str:
    """
    Construct MLflow trace URL for easy access to trace details.
    
    Args:
        trace_id: The MLflow trace ID
        mlflow_host: MLflow server host (default: http://127.0.0.1:5000)
        
    Returns:
        Full MLflow trace URL
    """
    base_url = f"{mlflow_host}/#/experiments/0"
    query_params = (
        "searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&"
        "startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&"
        "datasetsFilter=W10%3D&compareRunsMode=TRACES&"
        f"selectedEvaluationId={trace_id}"
    )
    return f"{base_url}?{query_params}"


# Define correctness metric
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt="""Compare the model response to the expected answer and determine if it's correct.
    
Consider the response correct if it:
1. Contains the key information from the expected answer
2. Is factually accurate based on the provided context
3. Adequately addresses the question asked

Return 'pass' if the response is correct, 'fail' if it's incorrect.

Question: {question}
Expected Answer: {expected_answer}
Model Response: {response}

Evaluation:""",
    allowed_values=["pass", "fail"],
)


@experiment()
async def evaluate_rag(row: Dict[str, Any], rag: RAG, llm) -> Dict[str, Any]:
    """
    Run RAG evaluation on a single row.
    
    Args:
        row: Dictionary containing question, context, and expected_answer
        rag: Pre-initialized RAG instance
        llm: Pre-initialized LLM client for evaluation
        
    Returns:
        Dictionary with evaluation results
    """
    question = row["question"]
    
    # Query the RAG system
    rag_response = await rag.query(question, top_k=4)
    model_response = rag_response.get("answer", "")
    
    # Evaluate correctness asynchronously
    score = await correctness_metric.ascore(
        question=question,
        expected_answer=row["expected_answer"],
        response=model_response,
        llm=llm
    )
    
    # Get trace ID and construct trace URL
    trace_id = rag_response.get("mlflow_trace_id", "N/A")
    trace_url = construct_mlflow_trace_url(trace_id) if trace_id != "N/A" else "N/A"
    
    # Return evaluation results
    result = {
        **row,
        "model_response": model_response,
        "correctness_score": score.value,
        "correctness_reason": score.reason,
        "mlflow_trace_id": trace_id,
        "mlflow_trace_url": trace_url,
        "retrieved_documents": [
            doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
            for doc in rag_response.get("retrieved_documents", [])
        ]
    }
    
    return result


async def run_experiment(mode: str = "naive", model: str = "gpt-5-mini", name: Optional[str] = None):
    """
    Simple function to run RAG evaluation experiment.
    
    Args:
        mode: RAG mode - "naive" or "agentic"
        model: OpenAI model to use
        name: Optional experiment name. If None, auto-generated with timestamp
        
    Returns:
        List of experiment results
    """
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set your OpenAI API key: export OPENAI_API_KEY='your_key'"
        )
    
    # Prepare dataset and initialize system
    logger.info("Initializing RAG system...")
    dataset = create_ragas_dataset(download_and_save_dataset())
    
    # Initialize RAG system with inline client creation
    openai_client = AsyncOpenAI(api_key=api_key)
    rag = RAG(
        llm_client=openai_client, 
        retriever=BM25Retriever(), 
        model=model, 
        mode=mode
    )
    logger.info("RAG system initialized!")
    
    # Run evaluation experiment
    experiment_results = await evaluate_rag.arun(
        dataset, 
        name=name or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{'agenticrag' if mode == 'agentic' else 'naiverag'}",
        rag=rag,
        llm=llm_factory("gpt-5-mini", client=openai_client, temperature=1, top_p=None)
    )
    
    # Print basic results
    if experiment_results:
        pass_count = sum(1 for result in experiment_results if result.get("correctness_score") == "pass")
        total_count = len(experiment_results)
        pass_rate = (pass_count / total_count) * 100 if total_count > 0 else 0
        
        logger.info(f"Results: {pass_count}/{total_count} passed ({pass_rate:.1f}%)")
    
    return experiment_results


if __name__ == "__main__":
    import sys
    
    # Simple command line argument parsing
    agentic_mode = "--agentic" in sys.argv
    mode = "agentic" if agentic_mode else "naive"
    
    if agentic_mode:
        logger.info("Running in AGENTIC mode")
    else:
        logger.info("Running in NAIVE mode")
    
    asyncio.run(run_experiment(mode=mode, model="gpt-5-mini"))


