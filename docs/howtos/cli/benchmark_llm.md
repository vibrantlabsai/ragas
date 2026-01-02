# LLM Benchmarking Quickstart

The `benchmark_llm` template benchmarks and compares different LLM models on discount calculation tasks.

## Create the Project

```sh
ragas quickstart benchmark_llm
cd benchmark_llm
```

## Install Dependencies

```sh
uv sync
```

## Set Your API Keys

```sh
export OPENAI_API_KEY="your-openai-key"
# Or other provider keys as needed
```

## Run the Evaluation

```sh
uv run python evals.py
```

To benchmark a specific model:

```sh
uv run python evals.py --model gpt-4o
uv run python evals.py --model gpt-3.5-turbo
```

## Project Structure

```
benchmark_llm/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── prompt.py              # Prompt implementation
├── evals.py               # Evaluation workflow
├── __init__.py            # Python package marker
└── evals/
    ├── datasets/
    │   └── discount_benchmark.csv  # Customer profiles and expected discounts
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

## What It Evaluates

The template benchmarks LLM performance on structured output tasks:

- **Task**: Calculate customer discount percentages based on profile
- **Models**: Compare GPT-4, GPT-3.5, Claude, Gemini, etc.
- **Output Format**: JSON with discount percentage
- **Metric**: Discount accuracy (correct/incorrect)

## Understanding the Code

### The Prompt (`prompt.py`)

Calculates discounts from customer profiles:

```python
from prompt import run_prompt

profile = "Premium customer, 5 years tenure, $50k annual spend"
result = await run_prompt(profile, model="gpt-4o")
# Returns: {"discount_percentage": 15}
```

### The Evaluation (`evals.py`)

Benchmarks model accuracy:

```python
@discrete_metric(name="discount_accuracy", allowed_values=["correct", "incorrect"])
def discount_accuracy(prediction: str, expected_discount):
    parsed_json = json.loads(prediction)
    predicted_discount = parsed_json.get("discount_percentage")

    if predicted_discount == int(expected_discount):
        return MetricResult(value="correct", ...)
    else:
        return MetricResult(value="incorrect", ...)
```

## Test Data

The template includes `evals/datasets/discount_benchmark.csv` with:

- Customer profiles (tenure, spend, tier, etc.)
- Expected discount percentages
- Business rules for discount calculation

## Benchmarking Multiple Models

Run the same evaluation across different models:

```sh
# GPT-4
uv run python evals.py --model gpt-4o

# GPT-3.5
uv run python evals.py --model gpt-3.5-turbo

# Claude
uv run python evals.py --model claude-3-5-sonnet-20241022

# Compare results
```

## Customization

### Add Your Own Task

Modify the prompt to benchmark different capabilities:

```python
# Code generation
prompt = "Generate Python code to {task}"

# Summarization
prompt = "Summarize this text in 50 words: {text}"

# Classification
prompt = "Classify this email as spam/not-spam: {email}"
```

### Compare Cost and Latency

Track additional metrics:

```python
import time

start = time.time()
response = await run_prompt(profile, model=model_name)
latency = time.time() - start

# Log cost and latency alongside accuracy
```

## Analyzing Results

Compare model performance:

```python
import pandas as pd

gpt4_results = pd.read_csv("evals/experiments/gpt4_benchmark.csv")
gpt35_results = pd.read_csv("evals/experiments/gpt35_benchmark.csv")

print(f"GPT-4 Accuracy: {(gpt4_results['discount_accuracy'] == 'correct').mean():.1%}")
print(f"GPT-3.5 Accuracy: {(gpt35_results['discount_accuracy'] == 'correct').mean():.1%}")
```

## Next Steps

- [Judge Alignment](judge_alignment.md) - Measure judge alignment
- [Prompt Evaluation](prompt_evals.md) - Compare different prompts
