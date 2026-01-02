# Agent Evaluation Quickstart

The `agent_evals` template provides a setup for evaluating AI agents that solve mathematical problems with correctness metrics.

## Create the Project

```sh
ragas quickstart agent_evals
cd agent_evals
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
agent_evals/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── agent.py               # Math solving agent implementation
├── evals.py               # Evaluation workflow
├── __init__.py            # Python package marker
└── evals/
    ├── datasets/          # Test datasets
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

## What It Evaluates

The template evaluates an AI agent's ability to solve mathematical expressions:

- **Agent**: Uses tools to solve mathematical problems step-by-step
- **Test Cases**: Math expressions like `(2 + 3) * (6 - 2)`, `100 / 5 + 3 * 2`
- **Metric**: Binary correctness (1.0 if correct, 0.0 if incorrect)

## Understanding the Code

### The Agent (`agent.py`)

Implements a math-solving agent with calculator tools:

```python
from agent import get_default_agent

math_agent = get_default_agent()
result = math_agent.solve("15 - 3 / 4")
```

### The Evaluation (`evals.py`)

Tests the agent on various math problems:

```python
@numeric_metric(name="correctness", allowed_values=(0.0, 1.0))
def correctness_metric(prediction: float, actual: float):
    result = 1.0 if abs(prediction - actual) < 1e-5 else 0.0
    return MetricResult(value=result, reason=f"Prediction: {prediction}, Actual: {actual}")
```

## Next Steps

- [LlamaIndex Agent Evaluation](llamaIndex_agent_evals.md) - Evaluate LlamaIndex agents
- [Custom Metrics](../customizations/metrics/_write_your_own_metric.md) - Write your own metrics
