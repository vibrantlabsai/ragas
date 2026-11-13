# Ragas Examples

Official examples demonstrating how to use Ragas for evaluating different types of AI applications including RAG systems, agents, prompts, workflows, and LLM benchmarking. These examples might be unstable and are subject to change.

## Installation

### From PyPI (after release)
```bash
pip install "ragas[examples]"
```

### Local Development
Install both main ragas and examples packages in editable mode:

```bash
cd /path/to/ragas
uv pip install -e . -e ./examples
```

Or using regular pip:
```bash
cd /path/to/ragas  
pip install -e . -e ./examples
```

## Available Examples

- **`ragas_examples.agent_evals`** - Agent evaluation examples
- **`ragas_examples.benchmark_llm`** - LLM benchmarking and comparison examples  
- **`ragas_examples.prompt_evals`** - Prompt evaluation examples
- **`ragas_examples.rag_eval`** - RAG system evaluation examples
- **`ragas_examples.workflow_eval`** - Workflow evaluation examples

## Usage

### Set Environment Variables

Most examples require API keys to be set:

```bash
export OPENAI_API_KEY=your_key_here
```

For Google Drive examples, also install the gdrive extra:
```bash
pip install "ragas[examples,gdrive]"
```

### Running Examples as Modules

After installation, you can run examples directly:

```bash
# Run benchmark LLM prompt example
python -m ragas_examples.benchmark_llm.prompt

# Run benchmark LLM evaluation
python -m ragas_examples.benchmark_llm.evals

# Run other examples
python -m ragas_examples.rag_eval.evals
python -m ragas_examples.agent_evals.evals
python -m ragas_examples.prompt_evals.evals
python -m ragas_examples.workflow_eval.evals
```

## Release process

- The examples package is released together with `ragas` and shares its version, which comes from the release tag (e.g., `v0.4.4`).
- Publishing is handled by the GitHub Actions workflow `python-publish.yml`. When a GitHub release is published it checks the tag, runs the unit tests and a clean-install check, and then uploads both `ragas` and `ragas-examples` to PyPI.

## Local Development & Testing


## Local Development & Testing

### Verify Installation
```bash

# Test module execution
python -m ragas_examples.benchmark_llm.prompt --help
```
