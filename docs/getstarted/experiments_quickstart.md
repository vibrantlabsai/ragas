# Run your first experiment

This tutorial walks you through running your first experiment with Ragas using the `@experiment` decorator and a local CSV backend.

## Prerequisites

- Python 3.9+
- Ragas installed (see [Installation](./install.md))

## Hello World 👋

![](/_static/imgs/experiments_quickstart/hello_world.gif)

### 1. Install (if you haven’t already)

```bash
pip install ragas
```

### 2. Create `hello_world.py`

Copy this into a new file and save as `hello_world.py`:

```python
import numpy as np
from ragas import Dataset, experiment
from ragas.metrics import MetricResult, discrete_metric


# Define a custom metric for accuracy
@discrete_metric(name="accuracy_score", allowed_values=["pass", "fail"])
def accuracy_score(response: str, expected: str):
    result = "pass" if expected.lower().strip() == response.lower().strip() else "fail"
    return MetricResult(value=result, reason=f"Match: {result == 'pass'}")


# Mock application endpoint that simulates an AI application response
def mock_app_endpoint(**kwargs) -> str:
    return np.random.choice(["Paris", "4", "Blue Whale", "Einstein", "Python"])


# Create an experiment that uses the mock application endpoint and the accuracy metric
@experiment()
async def run_experiment(row):
    response = mock_app_endpoint(query=row.get("query"))
    accuracy = accuracy_score.score(
        response=response, expected=row.get("expected_output")
    )
    return {**row, "response": response, "accuracy": accuracy.value}


if __name__ == "__main__":
    import asyncio

    # Create dataset inline
    dataset = Dataset(name="test_dataset", backend="local/csv", root_dir=".")
    test_data = [
        {"query": "What is the capital of France?", "expected_output": "Paris"},
        {"query": "What is 2 + 2?", "expected_output": "4"},
        {"query": "What is the largest animal?", "expected_output": "Blue Whale"},
        {
            "query": "Who developed the theory of relativity?",
            "expected_output": "Einstein",
        },
        {
            "query": "What programming language is named after a snake?",
            "expected_output": "Python",
        },
    ]

    for sample in test_data:
        dataset.append(sample)
    dataset.save()

    # Run experiment
    _ = asyncio.run(run_experiment.arun(dataset, name="first_experiment"))
```

### 3. Inspect the generated files

```bash
tree .
```

You should see:

```
├── datasets
│   └── test_dataset.csv
└── experiments
    └── first_experiment.csv
```

### 4. View the results of your first experiment

```bash
open experiments/first_experiment.csv
```

Output preview:

![](/_static/imgs/experiments_quickstart/output_first_experiment.png)

## Next steps

- Learn the concepts behind experiments in [Experiments (Concepts)](../concepts/experimentation.md)
- Explore evaluation metrics in [Metrics](../concepts/metrics/index.md)

