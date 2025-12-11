# RAG Evaluation Quickstart

The `rag_eval` template provides a complete RAG evaluation setup with custom metrics, dataset management, and experiment tracking.

## Create the Project

```sh
# Using uvx (no installation required)
uvx ragas quickstart rag_eval
cd rag_eval

# Or with ragas installed
ragas quickstart rag_eval
cd rag_eval
```

## Install Dependencies

```sh
uv sync
```

Or with pip:

```sh
pip install -e .
```

## Set Your API Key

=== "OpenAI (Default)"
    ```sh
    export OPENAI_API_KEY="your-openai-key"
    ```

=== "Anthropic Claude"
    ```sh
    export ANTHROPIC_API_KEY="your-anthropic-key"
    ```

    Update `evals.py`:
    ```python
    from anthropic import Anthropic
    from ragas.llms import llm_factory

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    llm = llm_factory("claude-3-5-sonnet-20241022", provider="anthropic", client=client)
    ```

=== "Google Gemini"
    ```sh
    export GOOGLE_API_KEY="your-google-api-key"
    ```

    Update `evals.py`:
    ```python
    import google.generativeai as genai
    from ragas.llms import llm_factory

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    client = genai.GenerativeModel("gemini-2.0-flash")
    llm = llm_factory("gemini-2.0-flash", provider="google", client=client)
    ```

=== "Local Models (Ollama)"
    ```python
    from openai import OpenAI
    from ragas.llms import llm_factory

    client = OpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1"
    )
    llm = llm_factory("mistral", provider="openai", client=client)
    ```

## Run the Evaluation

```sh
uv run python evals.py
```

The evaluation will:

1. Load test data from the `load_dataset()` function
2. Query your RAG application with test questions
3. Evaluate responses using custom metrics
4. Display results in the console
5. Save results to CSV in `evals/experiments/`

## Project Structure

```
rag_eval/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── rag.py                 # RAG application implementation
├── evals.py               # Evaluation workflow
├── __init__.py            # Python package marker
└── evals/
    ├── datasets/          # Test data files
    ├── experiments/       # Evaluation results (CSV)
    └── logs/              # Execution logs and traces
```

## Understanding the Code

### The RAG Application (`rag.py`)

A simple RAG implementation with:

- **Document storage**: In-memory document collection
- **Keyword retrieval**: Simple keyword matching for document retrieval
- **Response generation**: OpenAI API for generating answers
- **Tracing**: Logs each query for debugging

```python
from rag import default_rag_client

# Initialize with OpenAI client
rag_client = default_rag_client(llm_client=openai_client, logdir="evals/logs")

# Query the RAG system
response = rag_client.query("What is Ragas?")
print(response["answer"])
```

### The Evaluation Script (`evals.py`)

The evaluation workflow:

1. **Dataset loading**: Creates test cases with questions and grading notes
2. **Metric definition**: Custom `DiscreteMetric` for pass/fail evaluation
3. **Experiment execution**: Runs queries and evaluates responses
4. **Result storage**: Saves to CSV for analysis

```python
from ragas import Dataset, experiment
from ragas.metrics import DiscreteMetric

# Define your metric
my_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response contains points from grading notes...",
    allowed_values=["pass", "fail"],
)

# Run experiment
@experiment()
async def run_experiment(row):
    response = rag_client.query(row["question"])
    score = my_metric.score(llm=llm, response=response["answer"], ...)
    return {**row, "response": response["answer"], "score": score.value}
```

## Customization

### Add Test Cases

Edit the `load_dataset()` function in `evals.py`:

```python
def load_dataset():
    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir="evals",
    )

    data_samples = [
        {
            "question": "What is Ragas?",
            "grading_notes": "- evaluation framework - LLM applications",
        },
        {
            "question": "How do experiments work?",
            "grading_notes": "- track results - compare runs - store metrics",
        },
        # Add more test cases...
    ]

    for sample in data_samples:
        dataset.append(sample)
    dataset.save()
    return dataset
```

### Modify the Metric

Change evaluation criteria by updating the metric prompt:

```python
my_metric = DiscreteMetric(
    name="quality",
    prompt="""Evaluate the response quality:

Response: {response}
Expected Points: {grading_notes}

Rate as:
- 'excellent': All points covered with clear explanation
- 'good': Most points covered
- 'poor': Missing key points

Rating:""",
    allowed_values=["excellent", "good", "poor"],
)
```

### Add Multiple Metrics

Create additional metrics for different evaluation aspects:

```python
from ragas.metrics import DiscreteMetric, NumericalMetric

correctness = DiscreteMetric(
    name="correctness",
    prompt="Is the response factually correct? {response}",
    allowed_values=["correct", "incorrect"],
)

relevance = NumericalMetric(
    name="relevance",
    prompt="Rate relevance 1-5: {response} for question: {question}",
    allowed_values=(1, 5),
)
```

### Use Your Own RAG System

Replace the example RAG with your production system:

```python
# In evals.py
from your_rag_module import YourRAGClient

rag_client = YourRAGClient(...)

@experiment()
async def run_experiment(row):
    # Call your RAG system
    response = await rag_client.query(row["question"])

    score = my_metric.score(
        llm=llm,
        response=response,
        grading_notes=row["grading_notes"],
    )

    return {
        **row,
        "response": response,
        "score": score.value,
    }
```

## Viewing Results

Results are saved to `evals/experiments/` as CSV files. Each experiment run creates a new file with:

- Input data (questions, grading notes)
- Model responses
- Evaluation scores
- Timestamps

```python
import pandas as pd

# Load results
results = pd.read_csv("evals/experiments/your_experiment.csv")

# Calculate pass rate
pass_rate = (results["score"] == "pass").mean()
print(f"Pass rate: {pass_rate:.1%}")
```

## Next Steps

- [Improve RAG Guide](improve_rag.md) - Compare naive vs agentic RAG
- [Custom Metrics](../customizations/metrics/_write_your_own_metric.md) - Write your own metrics
- [Datasets](../../concepts/datasets.md) - Learn about dataset management
- [Experimentation](../../concepts/experimentation.md) - Advanced experiment tracking
