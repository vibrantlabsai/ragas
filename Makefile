GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

# Optionally show commands being executed with V=1
Q := $(if $(V),,@)

# Common paths
RAGAS_PATHS := src tests docs

help: ## Show all Makefile targets
	$(Q)grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# SETUP & INSTALLATION
# =============================================================================

setup-venv: ## Set up uv virtual environment
	@echo "Setting up uv virtual environment..."
	$(Q)VIRTUAL_ENV= uv venv
	@echo "Virtual environment created at .venv"
	@echo "To activate: source .venv/bin/activate"

install-minimal: ## Install minimal dev dependencies (fast setup - 79 packages)
	@echo "Installing minimal development dependencies (fast setup)..."
	@if [ ! -d ".venv" ]; then \
		echo "Virtual environment not found, creating one..."; \
		$(MAKE) setup-venv; \
	fi
	@echo "Installing core ragas + essential dev tools..."
	$(Q)uv pip install -e ".[dev-minimal]"
	@echo "Setting up pre-commit hooks..."
	$(Q)uv run pre-commit install
	@echo "Minimal installation complete! (79 packages)"
	@echo "Note: For full features including ML packages, use 'make install'"

install: ## Install full dependencies with uv sync (backward compatible - modern approach)
	@echo "Installing full development dependencies with uv sync..."
	@if [ ! -d ".venv" ]; then \
		echo "Virtual environment not found, creating one..."; \
		$(MAKE) setup-venv; \
	fi
	@echo "Installing ragas with full dev environment..."
	$(Q)VIRTUAL_ENV= uv sync --group dev
	@echo "Setting up pre-commit hooks..."
	$(Q)uv run pre-commit install
	@echo "Full installation complete! (Modern uv sync approach)"

# =============================================================================
# CODE QUALITY
# =============================================================================

.PHONY: help setup-venv install-minimal install format type check clean test test-e2e benchmarks benchmarks-docker run-ci run-ci-fast run-ci-format-check run-ci-type run-ci-tests build-docs serve-docs
format: ## Format and lint all code
	@echo "Formatting and linting all code..."
	@echo "(ruff format) Formatting ragas..."
	$(Q)uv run --active ruff format src tests docs --exclude src/ragas/_version.py --config pyproject.toml
	@echo "(ruff) Auto-fixing ragas (includes import sorting and unused imports)..."
	$(Q)uv run --active ruff check src tests docs --exclude src/ragas/_version.py --fix-only --config pyproject.toml
	@echo "(ruff) Final linting check for ragas..."
	$(Q)uv run --active ruff check src tests docs --exclude src/ragas/_version.py --config pyproject.toml

type: ## Type check all code
	@echo "Type checking all code..."
	@echo "(pyright) Typechecking ragas..."
	$(Q)PYRIGHT_PYTHON_FORCE_VERSION=latest uv run --active pyright -p pyproject.toml src

check: format type ## Quick health check (format + type, no tests)
	@echo "Code quality check complete!"

# =============================================================================
# BENCHMARKS
# =============================================================================
benchmarks: ## Run all benchmarks locally
	@echo "Running all benchmarks..."
	@echo "Running evaluation benchmarks..."
	$(Q)cd $(GIT_ROOT)/tests/benchmarks && uv run python benchmark_eval.py
	@echo "Running testset generation benchmarks..."
	$(Q)cd $(GIT_ROOT)/tests/benchmarks && uv run python benchmark_testsetgen.py

benchmarks-docker: ## Run benchmarks in docker
	@echo "Running benchmarks in docker..."
	$(Q)cd $(GIT_ROOT) || exit 1
	docker buildx build --build-arg OPENAI_API_KEY=$(OPENAI_API_KEY) -t ragas-benchmark -f $(GIT_ROOT)/tests/benchmarks/Dockerfile .
	docker inspect ragas-benchmark:latest | jq ".[0].Size" | numfmt --to=si

benchmarks-test: ## Run benchmarks for ragas unit tests
	@echo "Running ragas unit tests with timing benchmarks..."
	$(Q)uv run --active pytest --nbmake tests/unit --durations=0 -v $(shell if [ -n "$(k)" ]; then echo "-k $(k)"; fi)

# =============================================================================
# CI/BUILD
# =============================================================================

run-ci: ## Run complete CI pipeline (mirrors GitHub CI exactly)
	@echo "Running complete CI pipeline..."
	@echo "Format check..."
	$(Q)uv run --active ruff format --check src tests docs --exclude src/ragas/_version.py --config pyproject.toml
	$(Q)uv run --active ruff check src tests docs --exclude src/ragas/_version.py --config pyproject.toml
	@echo "Type check..."
	$(Q)$(MAKE) type
	@echo "Unit tests..."
	$(Q)__RAGAS_DEBUG_TRACKING=true RAGAS_DO_NOT_TRACK=true uv run --active pytest --nbmake tests/unit --dist loadfile -n auto
	@echo "All CI checks passed!"

run-ci-format-check: ## Run format check in dry-run mode (like GitHub CI)
	@echo "Running format check (dry-run, like GitHub CI)..."
	@echo "Checking ragas formatting..."
	$(Q)uv run --active ruff format --check src tests docs --exclude src/ragas/_version.py --config pyproject.toml
	$(Q)uv run --active ruff check src docs tests --exclude src/ragas/_version.py --config pyproject.toml

run-ci-type: ## Run type checking (matches GitHub CI)
	@echo "Running type checking (matches GitHub CI)..."
	$(Q)$(MAKE) type

run-ci-tests: ## Run all tests with CI options
	@echo "Running all tests with CI options..."
	$(Q)__RAGAS_DEBUG_TRACKING=true RAGAS_DO_NOT_TRACK=true pytest --nbmake tests/unit --dist loadfile -n auto

run-ci-fast: ## Fast CI check for quick local validation (2-3 minutes)
	@echo "Running fast CI check for quick feedback..."
	@echo "Format check..."
	$(Q)uv run --active ruff format --check src tests docs --exclude src/ragas/_version.py --config pyproject.toml
	$(Q)uv run --active ruff check src docs tests --exclude src/ragas/_version.py --config pyproject.toml
	@echo "Core unit tests (no nbmake for speed)..."
	$(Q)uv run --active pytest tests/unit --dist loadfile -n auto -x
	@echo "Fast CI check completed!"

clean: ## Clean all generated files
	@echo "Cleaning all generated files..."
	$(Q)find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	$(Q)rm -rf site/ docs/site/ .mypy_cache .pytest_cache .ruff_cache
	$(Q)rm -rf dist/ build/ *.egg-info/ src/*.egg-info/
	$(Q)rm -rf .coverage htmlcov/ .tox/ .venv/
	$(Q)find . -name '*.log' -delete
	$(Q)find . -name '.DS_Store' -delete
	$(Q)find . -name 'temp*' -type d -exec rm -rf {} + 2>/dev/null || true
	$(Q)find . -name '.tmp*' -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup complete!"

# =============================================================================
# TESTING
# =============================================================================

test: ## Run all unit tests
	@echo "Running all unit tests..."
	$(Q)uv run --active pytest tests/unit $(shell if [ -n "$(k)" ]; then echo "-k $(k)"; fi)

test-all: ## Run all unit tests (including notebooks)
	@echo "Running all unit tests (including notebooks)..."
	$(Q)uv run --active pytest --nbmake tests/unit $(shell if [ -n "$(k)" ]; then echo "-k $(k)"; fi)

test-e2e: ## Run all end-to-end tests
	@echo "Running all end-to-end tests..."
	$(Q)uv run --active pytest --nbmake tests/e2e -s

# =============================================================================
# DOCUMENTATION
# =============================================================================

build-docs: ## Build all documentation
	@echo "Building all documentation..."
	@echo "Converting ipynb notebooks to md files..."
	$(Q)MKDOCS_CI=true uv run python $(GIT_ROOT)/docs/ipynb_to_md.py
	@echo "Building ragas documentation..."
	$(Q)MKDOCS_CI=false uv run --group docs mkdocs build

serve-docs: ## Build and serve documentation locally
	$(Q)MKDOCS_CI=false uv run --group docs mkdocs serve --dirtyreload
