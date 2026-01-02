# LlamaIndex Agent Evaluation Quickstart

The `llamaIndex_agent_evals` template evaluates LlamaIndex workflow agents with tool call accuracy metrics.

## Create the Project

```sh
ragas quickstart llamaIndex_agent_evals
cd llamaIndex_agent_evals
```

## Install Dependencies

```sh
uv sync
```

## Set Your API Keys

```sh
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"  # For evaluator LLM
```

## Run the Evaluation

```sh
uv run python evals.py
```

## Project Structure

```
llamaIndex_agent_evals/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── llamaindex_agent.py    # LlamaIndex agent with tools
├── evals.py               # Evaluation workflow
├── __init__.py            # Python package marker
└── evals/
    ├── datasets/
    │   └── contexts/      # Test context files (JSON)
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

## What It Evaluates

The template evaluates a LlamaIndex agent's tool calling accuracy:

- **Agent**: LlamaIndex `FunctionAgent` with list management tools (add, remove, list items)
- **Test Cases**: Complex scenarios like duplicate additions, ambiguous removal requests
- **Metrics**: Tool call accuracy, response correctness

## Understanding the Code

### The Agent (`llamaindex_agent.py`)

LlamaIndex agent with simple tools:

```python
from llama_index.core.agent.workflow import FunctionAgent

agent = FunctionAgent(
    name="list_manager",
    tools=[add_item, remove_item, list_items],
    llm=llm
)
```

### The Evaluation (`evals.py`)

Tests tool call accuracy using F1 score:

```python
@numeric_metric(name="tool_call_accuracy")
def tool_call_accuracy_metric(predicted_calls: List[Dict], ground_truth_calls: List[Dict]):
    # Compares predicted vs ground truth tool calls
    # Returns F1 score between 0.0 and 1.0
```

## Test Data

The template includes JSON test contexts in `evals/datasets/contexts/`:

- `ambiguous_removal_request.json` - Tests handling of ambiguous requests
- `duplicate_addition.json` - Tests handling of duplicate operations
- `repeated_removal.json` - Tests repeated operations

## Next Steps

- [Agent Evaluation](agent_evals.md) - Evaluate general AI agents
- [Workflow Evaluation](workflow_eval.md) - Evaluate complex workflows
