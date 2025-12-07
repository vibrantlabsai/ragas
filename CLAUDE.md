# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ragas is an evaluation toolkit for Large Language Model (LLM) applications. It provides objective metrics for evaluating LLM applications, test data generation capabilities, and integrations with popular LLM frameworks.

The repository contains:

1. **Ragas Library** - The main evaluation toolkit including experimental features (in `src/ragas/` directory)
   - Core evaluation metrics and test generation
   - Experimental features available at `ragas.experimental`

## Development Environment Setup

### Installation

Choose the appropriate installation based on your needs:

```bash
# RECOMMENDED: Minimal dev setup (79 packages - fast)
make install-minimal

# FULL: Complete dev environment (383 packages - comprehensive)  
make install

# OR manual installation:
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Minimal dev setup (uses [project.optional-dependencies].dev-minimal)
uv pip install -e ".[dev-minimal]"

# Full dev setup (uses [dependency-groups].dev)
uv sync --group dev
```

### Installation Methods Explained

- **Minimal setup**: Uses `uv pip install` with optional dependencies for selective installation
- **Full setup**: Uses `uv sync` with dependency groups for comprehensive environment management
- **No naming conflicts**: `dev-minimal` vs `dev` clearly distinguish the two approaches

### Workspace Structure

The project uses a UV workspace configuration for managing multiple packages:

```bash
# Install
uv sync

# Install examples separately
uv sync --package ragas-examples

# Build specific workspace package
uv build --package ragas-examples
```

**Workspace Members:**
- `ragas` (main package) - Located in `src/ragas/`
- `ragas-examples` (examples package) - Located in `examples/`

The workspace ensures consistent dependency versions across packages and enables editable installs of workspace members.

## Common Commands

### Commands (from root directory)

```bash
# Setup and installation  
make install-minimal # Minimal dev setup (79 packages - recommended)
make install        # Full dev environment (383 packages - complete)

# Code quality
make format         # Format and lint all code
make type           # Type check all code
make check          # Quick health check (format + type, no tests)

# Testing
make test           # Run all unit tests
make test-e2e       # Run end-to-end tests

# CI/Build
make run-ci         # Run complete CI pipeline
make clean          # Clean all generated files

# Documentation
make build-docs     # Build all documentation
make serve-docs     # Serve documentation locally

# Benchmarks
make benchmarks     # Run performance benchmarks
make benchmarks-docker # Run benchmarks in Docker
```

### Testing

```bash
# Run all tests (from root)
make test

# Run specific test (using pytest -k flag)
make test k="test_name"

# Run end-to-end tests
make test-e2e

# Direct pytest commands for more control
uv run pytest tests/unit -k "test_name"
uv run pytest tests/unit -v
```

### Documentation

```bash
# Build all documentation (from root)
make build-docs

# Serve documentation locally
make serve-docs
```

### Benchmarks

```bash
# Run all benchmarks locally
make benchmarks

# Run benchmarks in Docker
make benchmarks-docker
```

## Project Architecture

The repository has the following structure:

```sh
/                          # Main ragas project
├── src/ragas/             # Source code including experimental features
│   └── experimental/      # Experimental features
├── tests/                 # All tests (core + experimental)
│   └── experimental/      # Experimental tests
├── examples/              # Example code
├── pyproject.toml         # Build config
├── docs/                  # Documentation
├── scripts/               # Build/CI scripts
├── Makefile               # Build commands
└── README.md              # Repository overview
```

### Ragas Core Components

The Ragas core library provides metrics, test data generation and evaluation functionality for LLM applications:

1. **Metrics** - Various metrics for evaluating LLM applications including:

   - AspectCritic
   - AnswerCorrectness
   - ContextPrecision
   - ContextRecall
   - Faithfulness
   - and many more

2. **Test Data Generation** - Automatic creation of test datasets for LLM applications

3. **Integrations** - Integrations with popular LLM frameworks like LangChain, LlamaIndex, and observability tools

### Experimental Components

The experimental features are now integrated into the main ragas package:

1. **Experimental features** are available at `ragas.experimental`
2. **Dataset and Experiment management** - Enhanced data handling for experiments
3. **Advanced metrics** - Extended metric capabilities
4. **Backend support** - Multiple storage backends (CSV, JSONL, Google Drive, in-memory)

To use experimental features:

```python
from ragas import Dataset
from ragas import experiment
from ragas.backends import get_registry
```

## Debugging Logs

To view debug logs for any module:

```python
import logging

# Configure logging for a specific module (example with analytics)
analytics_logger = logging.getLogger('ragas._analytics')
analytics_logger.setLevel(logging.DEBUG)

# Create a console handler and set its level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter and add it to the handler
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
analytics_logger.addHandler(console_handler)
```

## Memories

- whenever you create such docs put in in /\_experiments because that is gitignored and you can use it as a scratchpad or tmp directory for storing these
- always use uv to run python and python related commandline tools like isort, ruff, pyright etc. This is because we are using uv to manage the .venv and dependencies.
- The project uses two distinct dependency management approaches:
  - **Minimal setup**: `[project.optional-dependencies].dev-minimal` for fast development (79 packages)
  - **Full setup**: `[dependency-groups].dev` for comprehensive development (383 packages)
- Use `make install-minimal` for most development tasks, `make install` for full ML stack work
- if the user asks you to save a plan, save it into the plan/ directory with an appropriate file name.
