# Evaluate an AI workflow

This tutorial demonstrates how to evaluate an AI workflow using Ragas, here a simple custom email support triage workflow. By the end of this tutorial, you will learn how to evaluate and iterate on a workflow using evaluation-driven development.

```mermaid
flowchart LR
    A["Email Query"] --> B["Rule based Info Extractor"]
    B --> C["Template + LLM Response"]
    C --> D["Email Reply"]
```

We will start by testing our simple workflow that extracts the necessary information from an email, routes it to the correct template and generates response using an LLM.

```bash
python -m ragas_examples.workflow_eval.workflow
```


Next, we will write down a few sample email queries and expected outputs for our workflow. Then convert them to a CSV file.

```python
import pandas as pd

dataset_dict = [
    {
        "email": "Hi, I'm getting error code XYZ-123 when using version 2.1.4 of your software. Please help!",
        "pass_criteria": "category Bug Report; product_version 2.1.4; error_code XYZ-123; response references both version and error code",
    },
    {
        "email": "I need to dispute invoice #INV-2024-001 for 299.99 dollars. The charge seems incorrect.",
        "pass_criteria": "category Billing; invoice_number INV-2024-001; amount 299.99; response references invoice and dispute process",
    },
]
pd.DataFrame(dataset_dict).to_csv("datasets/test_dataset.csv", index=False)
```

To evaluate the performance of our workflow, we will define a llm based metric that compares the output of our workflow with the pass criteria and outputs pass/fail based on it.

```python
from ragas.metrics import DiscreteMetric

my_metric = DiscreteMetric(
    name="response_quality",
    prompt="Evaluate the response based on the pass criteria: {pass_criteria}. Does the response meet the criteria? Return 'pass' or 'fail'.\nResponse: {response}",
    allowed_values=["pass", "fail"],
)
```

Next, we will write the evaluation experiment loop that will run our workflow on the test dataset and evaluate it using the metric, and store the results in a CSV file.

```python
from ragas import experiment


@experiment()
async def run_experiment(row):
    response = workflow_client.process_email(row["email"])

    score = my_metric.score(
        llm=llm,
        response=response.get("response_template", " "),
        pass_criteria=row["pass_criteria"],
    )

    experiment_view = {
        **row,
        "response": response.get("response_template", " "),
        "score": score.value,
        "score_reason": score.reason,
    }
    return experiment_view
```

Now whenever you make a change to your workflow, you can run the experiment and see how it affects the performance of your workflow. Then compare it to the previous results to see how it has improved or degraded.

## Running the example end to end
1. Setup your OpenAI API key
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

2. Run the experiment
```bash
python -m ragas_examples.workflow_eval.evals
```

Voila! You have successfully run your first evaluation using Ragas. You can now inspect the results by opening the `experiments/experiment_name.csv` file.