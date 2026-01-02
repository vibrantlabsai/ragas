# Workflow Evaluation Quickstart

The `workflow_eval` template evaluates complex LLM workflows with email classification and routing.

## Create the Project

```sh
ragas quickstart workflow_eval
cd workflow_eval
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
workflow_eval/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── workflow.py            # Workflow implementation
├── evals.py               # Evaluation workflow
├── __init__.py            # Python package marker
└── evals/
    ├── datasets/          # Test datasets
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

## What It Evaluates

The template evaluates a customer support email classification workflow:

- **Workflow**: Multi-step email processing (classification → extraction → response)
- **Categories**: Bug Report, Feature Request, Billing
- **Test Cases**: Customer emails with expected categories and extracted fields
- **Metric**: Custom discrete metric checking classification accuracy

## Understanding the Code

### The Workflow (`workflow.py`)

Implements a customer support email workflow:

```python
from workflow import default_workflow_client

workflow = default_workflow_client()
result = workflow.process_email("I found a bug in version 2.1.4...")
# Returns: category, extracted fields, response
```

### The Evaluation (`evals.py`)

Tests workflow accuracy against pass criteria:

```python
def load_dataset():
    dataset_dict = [
        {
            "email": "Hi, I'm getting error code XYZ-123 when using version 2.1.4...",
            "pass_criteria": "category Bug Report; product_version 2.1.4; error_code XYZ-123",
        },
        # More test cases...
    ]
```

The metric evaluates if the workflow correctly:
- Classifies the email category
- Extracts relevant fields (version, error code, invoice number, etc.)
- Generates appropriate responses

## Test Cases

The template includes diverse scenarios:

- **Bug Reports**: With version numbers and error codes
- **Feature Requests**: With urgency levels and product areas
- **Billing Issues**: With invoice numbers and amounts

## Customization

### Add Your Own Workflow

Replace the example workflow with your own:

```python
from your_workflow import YourWorkflow

workflow = YourWorkflow()

@experiment()
async def run_experiment(row):
    result = await workflow.process(row["input"])
    # Evaluate result...
```

## Next Steps

- [Agent Evaluation](agent_evals.md) - Evaluate AI agents
- [LlamaIndex Agent Evaluation](llamaIndex_agent_evals.md) - Evaluate LlamaIndex workflows
