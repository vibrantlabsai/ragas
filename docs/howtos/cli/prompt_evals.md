# Prompt Evaluation Quickstart

The `prompt_evals` template evaluates and compares different prompt variations with sentiment analysis.

## Create the Project

```sh
ragas quickstart prompt_evals
cd prompt_evals
```

## Install Dependencies

```sh
uv sync
```

## Set Your API Key

```sh
export OPENAI_API_KEY="your-openai-key"
```

## Run the Evaluation

```sh
uv run python evals.py
```

## Project Structure

```
prompt_evals/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── prompt.py              # Prompt implementation
├── evals.py               # Evaluation workflow
├── __init__.py            # Python package marker
└── evals/
    ├── datasets/          # Test datasets
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

## What It Evaluates

The template evaluates prompt effectiveness for sentiment classification:

- **Task**: Sentiment analysis (positive/negative)
- **Test Cases**: Movie reviews with expected sentiment labels
- **Metric**: Binary accuracy (pass/fail)

## Understanding the Code

### The Prompt (`prompt.py`)

Implements the sentiment analysis prompt:

```python
from prompt import run_prompt

sentiment = run_prompt("I loved the movie! It was fantastic.")
# Returns: "positive" or "negative"
```

### The Evaluation (`evals.py`)

Tests prompt accuracy:

```python
@discrete_metric(name="accuracy", allowed_values=["pass", "fail"])
def my_metric(prediction: str, actual: str):
    return (
        MetricResult(value="pass", reason="")
        if prediction == actual
        else MetricResult(value="fail", reason="")
    )
```

## Test Data

The dataset includes movie reviews:

```python
dataset_dict = [
    {"text": "I loved the movie! It was fantastic.", "label": "positive"},
    {"text": "The movie was terrible and boring.", "label": "negative"},
    # More examples...
]
```

## Customization

### Test Different Prompts

Modify `prompt.py` to test variations:

```python
# Version 1: Simple
prompt = f"Is this positive or negative: {text}"

# Version 2: With examples
prompt = f"""Classify sentiment:
Examples:
- "Great movie" -> positive
- "Boring film" -> negative

Text: {text}
Sentiment:"""

# Compare results across versions
```

### Add More Metrics

Evaluate additional aspects:

```python
from ragas.metrics import NumericalMetric

confidence = NumericalMetric(
    name="confidence",
    prompt="Rate confidence 1-5 in this classification: {prediction}",
    allowed_values=(1, 5),
)
```

## Next Steps

- [Judge Alignment](judge_alignment.md) - Measure LLM-as-judge alignment
- [LLM Benchmarking](benchmark_llm.md) - Compare different models
