# Judge Alignment Quickstart

The `judge_alignment` template measures how well an LLM-as-judge aligns with human evaluation standards.

## Create the Project

```sh
ragas quickstart judge_alignment
cd judge_alignment
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
judge_alignment/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── evals.py               # Evaluation workflow
├── __init__.py            # Python package marker
└── evals/
    ├── datasets/          # Test datasets
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

## What It Evaluates

The template evaluates LLM judge alignment:

- **Scenario**: Pre-existing responses are evaluated by an LLM judge
- **Human Labels**: Ground truth pass/fail labels
- **LLM Judge**: Evaluates same responses with grading criteria
- **Alignment Metric**: Agreement between human and LLM judgments

## Understanding the Code

### Judge Metrics (`evals.py`)

Two judge implementations to compare:

```python
# Baseline judge (simple prompt)
accuracy_metric = DiscreteMetric(
    name="accuracy",
    prompt="Check if response contains points from grading notes...",
    allowed_values=["pass", "fail"],
)

# Improved judge (enhanced with abbreviation guide)
accuracy_metric_v2 = DiscreteMetric(
    name="accuracy",
    prompt="""Evaluate if response covers ALL key concepts...

    ABBREVIATION GUIDE:
    • Financial: val=valuation, post-$=post-money, rev=revenue...
    • Business: mkt=market, reg=regulation...
    """,
    allowed_values=["pass", "fail"],
)
```

### The Evaluation

Tests alignment with human judgment:

```python
@discrete_metric(name="alignment", allowed_values=["aligned", "misaligned"])
def alignment_metric(llm_judgment: str, human_judgment: str):
    # Compares LLM judge output with human label
    return "aligned" if llm_judgment == human_judgment else "misaligned"
```

## Test Data

The dataset includes:
- Pre-evaluated responses
- Human pass/fail labels
- Grading notes with expected points
- Various abbreviations and business terminology

## Use Cases

### Compare Judge Versions

Run experiments with both judges:

```python
# Test baseline judge
results_v1 = await run_with_judge(accuracy_metric)

# Test improved judge
results_v2 = await run_with_judge(accuracy_metric_v2)

# Compare alignment rates
```

### Improve Judge Quality

Iterate on judge prompts to improve alignment:

1. Identify misalignment patterns
2. Update judge prompt with clearer criteria
3. Re-evaluate alignment
4. Repeat until satisfactory

## Next Steps

- [Prompt Evaluation](prompt_evals.md) - Compare different prompts
- [LLM Benchmarking](benchmark_llm.md) - Compare different models
