# Ragas CLI

The Ragas Command Line Interface (CLI) provides tools for quickly setting up evaluation projects and running experiments from the terminal.

## Installation

The CLI is included with the ragas package:

```sh
pip install ragas
```

Or use `uvx` to run without installation:

```sh
uvx ragas --help
```

## Available Commands

### `ragas quickstart`

Create a complete evaluation project from a template. This is the fastest way to get started with Ragas.

```sh
ragas quickstart [TEMPLATE] [OPTIONS]
```

**Arguments:**

- `TEMPLATE`: Template name (optional). Leave empty to see available templates.

**Options:**

- `-o, --output-dir`: Directory to create the project in (default: current directory)

**Examples:**

```sh
# List available templates
ragas quickstart

# Create a RAG evaluation project
ragas quickstart rag_eval

# Create project in a specific directory
ragas quickstart rag_eval --output-dir ./my-project
```

### `ragas evals`

Run evaluations on a dataset using an evaluation file.

```sh
ragas evals EVAL_FILE [OPTIONS]
```

**Arguments:**

- `EVAL_FILE`: Path to the evaluation file (required)

**Options:**

- `--dataset`: Name of the dataset in the project (required)
- `--metrics`: Comma-separated list of metric field names to evaluate (required)
- `--baseline`: Baseline experiment name to compare against (optional)
- `--name`: Name of the experiment run (optional)

**Example:**

```sh
ragas evals evals.py --dataset test_data --metrics accuracy,relevance
```

### `ragas hello_world`

Create a simple hello world example to verify your installation.

```sh
ragas hello_world [DIRECTORY]
```

**Arguments:**

- `DIRECTORY`: Directory to create the example in (default: current directory)

## Quickstart Templates

### RAG & Retrieval
- [RAG Evaluation (`rag_eval`)](rag_eval.md) - Evaluate RAG systems with custom metrics
- [Improve RAG (`improve_rag`)](improve_rag.md) - Compare naive vs agentic RAG approaches

### Agent Evaluation
- [Agent Evaluation (`agent_evals`)](agent_evals.md) - Evaluate AI agents solving math problems
- [LlamaIndex Agent Evaluation (`llamaIndex_agent_evals`)](llamaIndex_agent_evals.md) - Evaluate LlamaIndex agents with tool call metrics

### Specialized Use Cases
- [Text-to-SQL Evaluation (`text2sql`)](text2sql.md) - Evaluate text-to-SQL systems with execution accuracy
- [Workflow Evaluation (`workflow_eval`)](workflow_eval.md) - Evaluate complex LLM workflows
- [Prompt Evaluation (`prompt_evals`)](prompt_evals.md) - Compare different prompt variations

### LLM Testing
- [Judge Alignment (`judge_alignment`)](judge_alignment.md) - Measure LLM-as-judge alignment with human standards
- [LLM Benchmarking (`benchmark_llm`)](benchmark_llm.md) - Benchmark and compare different LLM models

## Quick Start

Get running in 60 seconds:

```sh
# Create project
uvx ragas quickstart rag_eval
cd rag_eval

# Install dependencies
uv sync

# Set API key
export OPENAI_API_KEY="your-key"

# Run evaluation
uv run python evals.py
```

## Next Steps

- [RAG Evaluation Guide](rag_eval.md) - Detailed walkthrough of the rag_eval template
- [Improve RAG Guide](improve_rag.md) - Compare naive vs agentic RAG approaches
- [Custom Metrics](../customizations/metrics/_write_your_own_metric.md) - Create your own evaluation metrics
