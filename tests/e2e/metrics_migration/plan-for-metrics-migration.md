# Comprehensive Generalizable Metrics Migration Plan

## Overview
This document provides a complete, step-by-step plan for migrating any metric from legacy implementation to the modern collections pattern, incorporating all learnings from Context Recall migration, test infrastructure refactoring, and notebook-based testing approaches.

---

## Phase 0: Pre-Migration Study & Planning

### Study Existing Migrated Metrics

**Metrics to analyze**:
1. Answer Relevancy (LLM + Embeddings based)
2. Answer Similarity (Embeddings only)
3. BLEU/ROUGE (No LLM/embeddings)
4. String metrics (Simple comparison)
5. Context Recall (LLM with statement classification)

**What to look for in legacy metrics** (`src/ragas/metrics/_*.py`):
- [ ] **Core algorithm logic**: How is the score calculated?
- [ ] **LLM/Embeddings usage**: Which components are required?
- [ ] **Prompt structure**: PydanticPrompt classes and examples
- [ ] **Input parameters**: What data does it need?
- [ ] **Edge cases**: How are empty inputs, errors handled?
- [ ] **Ensembling**: Does it run multiple times and aggregate?
- [ ] **Deprecated methods**: Old APIs to maintain compatibility with
- [ ] **Output format**: Float score vs structured output

**Important patterns from legacy**:
1. `_single_turn_ascore()` is the main method to replicate
2. `MetricWithLLM`, `MetricWithEmbeddings` mixins show dependencies
3. `PydanticPrompt` examples become inline examples in new prompts
4. Score normalization and range validation (0.0-1.0)
5. Error handling and nan score returns

---

## Phase 1: Implement New Metric

### 1.1 Create Prompt Function
**File**: `src/ragas/prompts/metrics/{metric_name}.py`

**Structure**:
```python
"""Prompt for {MetricName} evaluation."""

import json

def {metric_name}_prompt(param1: str, param2: str, ...) -> str:
    """
    Generate prompt for {metric_name} evaluation.

    Args:
        param1: Description
        param2: Description

    Returns:
        Formatted prompt string for LLM
    """
    # Use json.dumps() for safe string escaping
    safe_param1 = json.dumps(param1)
    safe_param2 = json.dumps(param2)

    return f"""Task description here.

--------EXAMPLES-----------
Example 1
Input: {{
    "param1": "example value",
    "param2": "example value"
}}
Output: {{
    "result": "expected output format"
}}

Example 2
[Add 2-3 examples covering different scenarios]
-----------------------------

Now perform the same with the following input
Input: {{
    "param1": {safe_param1},
    "param2": {safe_param2}
}}
Output: """
```

**Key points**:
- Use `json.dumps()` for escaping user inputs
- Include 2-3 examples showing different cases
- Clear output format specification
- Match the logic from legacy PydanticPrompt

### 1.2 Define Output Models
**File**: `src/ragas/metrics/collections/_{metric_name}.py`

```python
from pydantic import BaseModel
import typing as t

class {MetricName}Item(BaseModel):
    """Single classification/item result."""
    field1: str
    field2: int
    # ... based on legacy output model

class {MetricName}Output(BaseModel):
    """Complete structured output."""
    items: t.List[{MetricName}Item]
    # or whatever structure the LLM returns
```

**Guidelines**:
- Match field names from legacy output models
- Use appropriate types (str, int, float, List, etc.)
- Add docstrings for clarity

### 1.3 Implement Metric Class
**File**: `src/ragas/metrics/collections/_{metric_name}.py`

```python
"""MetricName v2 - Modern implementation with instructor LLMs."""

import typing as t
import numpy as np
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.prompts.metrics.{metric_name} import {metric_name}_prompt

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.embeddings.base import BaseRagasEmbeddings

class {MetricName}(BaseMetric):
    """
    {Metric description - what it measures}.

    This implementation uses modern instructor LLMs with structured output.
    Only supports modern components - legacy wrappers rejected with clear errors.

    Usage:
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import instructor_llm_factory
        >>> from ragas.metrics.collections import {MetricName}
        >>>
        >>> client = AsyncOpenAI()
        >>> llm = instructor_llm_factory("openai", client=client, model="gpt-4o-mini")
        >>>
        >>> metric = {MetricName}(llm=llm)
        >>> result = await metric.ascore(param1="value1", param2="value2")
        >>> print(f"Score: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM (if needed)
        embeddings: Modern embeddings (if needed)
        name: Metric name
        allowed_values: Score range (0.0 to 1.0)
    """

    # Type hints for components
    llm: "InstructorBaseRagasLLM"  # If LLM-based
    embeddings: "BaseRagasEmbeddings"  # If embeddings-based

    def __init__(
        self,
        llm: t.Optional["InstructorBaseRagasLLM"] = None,
        embeddings: t.Optional["BaseRagasEmbeddings"] = None,
        name: str = "{metric_name}",
        **kwargs,
    ):
        """Initialize metric with required components."""
        # Set attributes before super() for validation
        if llm:
            self.llm = llm
        if embeddings:
            self.embeddings = embeddings

        # BaseMetric validates components are modern (not legacy wrappers)
        super().__init__(name=name, **kwargs)

    async def ascore(
        self,
        param1: str,
        param2: str,
        # ... other parameters based on metric needs
    ) -> MetricResult:
        """
        Calculate score asynchronously.

        Args:
            param1: Description
            param2: Description

        Returns:
            MetricResult with score (0.0-1.0)
        """
        # 1. Validate inputs (handle empty/None cases)
        if not param1 or not param2:
            return MetricResult(value=0.0)

        # 2. For LLM-based metrics: Generate prompt and get structured output
        prompt = {metric_name}_prompt(param1=param1, param2=param2)
        output = await self.llm.agenerate(prompt, {MetricName}Output)

        # 3. For embeddings-based metrics: Get embeddings and compute similarity
        # embedding1 = await self.embeddings.embed_text(param1)
        # embedding2 = await self.embeddings.embed_text(param2)
        # score = cosine_similarity(embedding1, embedding2)

        # 4. Calculate score from output (match legacy logic exactly)
        score = self._calculate_score(output)

        # 5. Return MetricResult
        return MetricResult(value=float(score))

    def _calculate_score(self, output: {MetricName}Output) -> float:
        """Calculate final score from LLM output."""
        # Implement exact logic from legacy _single_turn_ascore
        # This is where the core algorithm lives
        pass
```

**Key patterns**:
- `__init__` sets attributes before `super()` for validation
- `ascore()` is the main public method (not `_single_turn_ascore`)
- Return `MetricResult` not raw float
- Match legacy calculation logic exactly
- Handle edge cases (empty inputs, None values)
- Type hints use `TYPE_CHECKING` for circular imports

### 1.4 Update Exports
**File**: `src/ragas/metrics/collections/__init__.py`

```python
from ._metric_name import MetricName

__all__ = [
    # ... existing exports
    "MetricName",
]
```

---

## Phase 2: Manual Testing with General-Purpose Notebook

### 2.1 Use General-Purpose Testing Notebook

**File**: `tests/notebooks/metric_score_diff.ipynb` (already exists - reusable for all metrics)

**Purpose**: Validate migration on real-world datasets (PRIMARY) and test edge cases (SECONDARY)

**Testing Priority**:
1. **PRIMARY**: Large-scale dataset testing (amnesty_qa, fiqa) - proves migration quality
2. **SECONDARY**: Hand-crafted edge cases - validates specific behaviors

**Key Advantage**: This notebook is configuration-driven. You only need to edit ONE cell (Cell 2) with your metric configuration, then run all cells without any other modifications!

**What the notebook provides**:
- Automatic component creation (LLM/embeddings) based on your needs
- Dynamic metric loading from your configuration
- Dataset-based testing (Amnesty QA + FIQA)
- Comprehensive statistical analysis and visualizations
- Validation criteria checking
- Optional edge case testing

---

### 2.2 Generate Metric Configuration

Generate the `METRIC_CONFIG` dictionary for Cell 2 of the notebook. Print it to console for easy copy-pasting. Use the template below based on your metric type:

#### Configuration Template

```python
METRIC_CONFIG = {
    # ===== METRIC IMPORTS =====
    "legacy_import": {
        "module": "ragas.metrics._{legacy_module_name}",  # e.g., "ragas.metrics._answer_relevance"
        "class_name": "{LegacyMetricClassName}",  # e.g., "AnswerRelevancy"
    },
    "modern_import": {
        "module": "ragas.metrics.collections",
        "class_name": "{ModernMetricClassName}",  # e.g., "AnswerRelevancy"
    },
    # ===== COMPONENT REQUIREMENTS =====
    # Set to False if your metric doesn't need this component
    "needs_llm": True,  # Does your metric use an LLM?
    "needs_embeddings": True,  # Does your metric use embeddings?
    # ===== DATASET FIELD MAPPING =====
    # Choose ONE option based on your metric type (uncomment the appropriate one)
    # OPTION 1: Answer-based metrics (AnswerRelevancy, AnswerSimilarity, AnswerCorrectness, etc.)
    "dataset_fields": ["user_input", "response"],
    # OPTION 2: Context-based metrics (ContextRecall, ContextPrecision, Faithfulness, etc.)
    # "dataset_fields": ["user_input", "retrieved_contexts", "reference"],
    # OPTION 3: Deterministic/Non-LLM metrics (NonLLMContextRecall, etc.)
    # "dataset_fields": ["retrieved_contexts", "reference_contexts"],
}
```

#### Configuration Examples

**Example 1: AnswerRelevancy (LLM + Embeddings)**
```python
METRIC_CONFIG = {
    "legacy_import": {
        "module": "ragas.metrics._answer_relevance",
        "class_name": "AnswerRelevancy",
    },
    "modern_import": {
        "module": "ragas.metrics.collections",
        "class_name": "AnswerRelevancy",
    },
    "needs_llm": True,
    "needs_embeddings": True,
    "dataset_fields": ["user_input", "response"],
}
```

**Example 2: ContextRecall (LLM only)**
```python
METRIC_CONFIG = {
    "legacy_import": {
        "module": "ragas.metrics._context_recall",
        "class_name": "ContextRecall",
    },
    "modern_import": {
        "module": "ragas.metrics.collections",
        "class_name": "ContextRecall",
    },
    "needs_llm": True,
    "needs_embeddings": False,
    "dataset_fields": ["user_input", "retrieved_contexts", "reference"],
}
```

**Example 3: NonLLMContextRecall (No LLM/Embeddings)**
```python
METRIC_CONFIG = {
    "legacy_import": {
        "module": "ragas.metrics._context_recall",
        "class_name": "NonLLMContextRecall",
    },
    "modern_import": {
        "module": "ragas.metrics.collections",
        "class_name": "NonLLMContextRecall",
    },
    "needs_llm": False,
    "needs_embeddings": False,
    "dataset_fields": ["retrieved_contexts", "reference_contexts"],
}
```

**Example 4: ContextPrecision (LLM only)**
```python
METRIC_CONFIG = {
    "legacy_import": {
        "module": "ragas.metrics._context_precision",
        "class_name": "ContextPrecision",
    },
    "modern_import": {
        "module": "ragas.metrics.collections",
        "class_name": "ContextPrecision",
    },
    "needs_llm": True,
    "needs_embeddings": False,
    "dataset_fields": ["user_input", "retrieved_contexts", "reference"],
}
```

#### How to Choose `dataset_fields`

The `dataset_fields` list tells the notebook which fields to extract from the test datasets (Amnesty QA, FIQA) for your metric:

1. **Answer-based metrics**: Use `["user_input", "response"]`
   - Metrics that evaluate the quality of generated answers
   - Examples: AnswerRelevancy, AnswerSimilarity, AnswerCorrectness

2. **Context-based metrics**: Use `["user_input", "retrieved_contexts", "reference"]`
   - Metrics that evaluate retrieved context quality
   - Examples: ContextRecall, ContextPrecision, Faithfulness

3. **Deterministic metrics**: Use `["retrieved_contexts", "reference_contexts"]`
   - Metrics that don't use LLMs and compare contexts directly
   - Examples: NonLLMContextRecall
   - Note: The notebook will automatically split `retrieved_contexts` to create `reference_contexts` if needed

**Available dataset fields**:
- **Amnesty QA**: `user_input`, `response`, `retrieved_contexts`, `reference_contexts`
- **FIQA**: `user_input`, `response`, `retrieved_contexts`, `reference`

---

### 2.3 Run Notebook and Analyze Results

**Steps**:

1. **Open the notebook**: `tests/notebooks/metric_score_diff.ipynb`

2. **Edit Cell 2**: Replace the `METRIC_CONFIG` dictionary with your generated configuration from Section 2.2

3. **Run all cells**: The notebook handles everything automatically:
   - Loads your metric classes dynamically
   - Creates only the required components (LLM/embeddings)
   - Initializes both legacy and modern metrics
   - Loads and transforms datasets based on your `dataset_fields`
   - Runs concurrent comparisons on Amnesty QA and FIQA
   - Generates comprehensive statistical analysis
   - Creates 7-plot visualizations for each dataset
   - Validates results against migration criteria

4. **Review results**: The notebook displays inline:
   - Score comparison statistics (mean, std dev, differences)
   - Tolerance analysis (% of samples within various thresholds)
   - Top 10 largest differences with descriptions
   - Comprehensive visualizations (scatter, histograms, trends, distributions)
   - Validation criteria checkmarks (✅/❌)

5. **Iterate if needed**:
   - If scores don't match well, review the problematic cases
   - Adjust your metric implementation
   - Re-run the notebook to verify improvements

6. **Document findings**: Print a migration summary with the following information:
   - Mean absolute difference
   - Percentage of samples within tolerance
   - Recommended tolerance level
   - Any patterns or anomalies observed
   - Edge cases that need special handling
   - Key implementation details and algorithm differences

**Output approach**: Print the METRIC_CONFIG and migration summary directly to console/output instead of creating files. This allows for easy copy-pasting without cluttering the repository.

---

---

### 2.4 Migration Validation Criteria

After running the notebook, the migration is considered successful if:

**Amnesty QA Dataset** (PRIMARY criterion):
- ✅ Mean absolute difference < 0.15 (stricter than per-case tolerance)
- ✅ >90% of samples within 0.2 tolerance for LLM-based metrics
- ✅ >95% of samples within 1e-6 tolerance for deterministic metrics
- ✅ No systematic bias (mean diff close to 0, ideally < 0.05)
- ✅ Similar score distributions (check box plots and histograms)

**FIQA Dataset** (if available):
- ✅ Similar criteria as amnesty_qa
- ✅ Validates generalization across different domains

**Edge Cases** (SECONDARY criterion):
- ✅ All edge cases handle gracefully (no crashes)
- ✅ Empty inputs return 0.0 or handle appropriately
- ✅ Special characters don't break the metric

**Performance**:
- ✅ New implementation not significantly slower (< 2x)
- ✅ Concurrent processing works correctly

**Documentation**:
For the migration, review and document in the notebook:
- Dataset comparison statistics (displayed inline)
- Top 10 largest differences with analysis (displayed inline)
- Visual analysis with 7 comprehensive plots (displayed inline)
- Any patterns or anomalies observed
- Recommended tolerance for E2E tests

**This becomes the proof that migration works correctly!**

**Note**: All results are displayed inline in the notebook - no CSV or PNG files are saved.

---

## Phase 3: Write E2E Migration Tests

### 3.1 Create Test File
**File**: `tests/e2e/metrics_migration/test_{metric_name}_migration.py`

**Structure**:
```python
"""E2E tests for {MetricName} migration from v1 to v2."""

import pytest

from ragas.metrics import {LegacyMetricName}
from ragas.metrics.collections import {MetricName}

from .base_migration_test import BaseMigrationTest

class Test{MetricName}E2EMigration(BaseMigrationTest):
    """E2E compatibility tests between legacy and v2 implementations."""

    @pytest.fixture
    def sample_data(self):
        """Test cases for {metric_name} evaluation.

        Based on dataset testing in notebook: tests/notebooks/metric_score_diff.ipynb

        Dataset validation results:
        - Amnesty QA: Mean |diff|={mean_diff:.4f}, {pct_within_tolerance}% within tolerance
        - FIQA: Mean |diff|={mean_diff:.4f}, {pct_within_tolerance}% within tolerance (if tested)

        These test cases focus on edge cases and specific behaviors not fully covered by datasets.
        The primary validation comes from the dataset comparisons documented in the notebook.
        """
        return [
            # Edge cases from notebook testing
            # Cases with interesting/problematic behavior from dataset analysis
            # Specific scenarios requiring validation
            {
                "param1": "value1",
                "param2": "value2",
                "description": "Test case description",
            },
        ]

    @pytest.mark.asyncio
    async def test_legacy_vs_v2_e2e_compatibility(
        self,
        sample_data,
        legacy_llm,  # from conftest.py
        modern_llm,  # from conftest.py
        legacy_embeddings,  # if needed
        modern_embeddings,  # if needed
    ):
        """E2E test that legacy and v2 produce similar scores."""
        await self.run_e2e_compatibility_test(
            sample_data=sample_data,
            legacy_metric_factory={LegacyMetricName},
            v2_metric_factory={MetricName},
            legacy_components={"llm": legacy_llm, "embeddings": legacy_embeddings},
            v2_components={"llm": modern_llm, "embeddings": modern_embeddings},
            tolerance=0.2,  # Adjust based on notebook findings
            metric_name="{MetricName}",
            additional_info_keys=["param1", "param2"],  # For debug output
        )

    @pytest.mark.asyncio
    async def test_{metric_specific_behavior}(
        self,
        legacy_llm,
        modern_llm,
    ):
        """Test metric-specific behavior."""

        test_cases = [
            {
                "param1": "specific case",
                "param2": "for testing",
                "expected_high": True,  # or other expected behavior
                "description": "Specific behavior description",
            },
            # Add 2-3 cases testing specific behaviors
        ]

        def assertion_fn(case, legacy_score, v2_result):
            """Custom assertions for metric-specific behavior."""
            if case.get("expected_high"):
                assert legacy_score > 0.8
                assert v2_result.value > 0.8
                print("   ✅ High score as expected")
            # Add other assertions based on metric logic

        await self.run_metric_specific_test(
            test_cases=test_cases,
            legacy_metric_factory={LegacyMetricName},
            v2_metric_factory={MetricName},
            legacy_components={"llm": legacy_llm},
            v2_components={"llm": modern_llm},
            test_name="{specific behavior}",
            assertion_fn=assertion_fn,
        )

    def test_migration_requirements_documented(self):
        """Document requirements for running E2E tests."""
        requirements = {
            "llm": "OpenAI GPT or compatible LLM",
            "embeddings": "OpenAI embeddings (if needed)",
            "environment": "API keys configured",
            "purpose": "Verify v2 produces similar scores to legacy",
        }

        self.create_requirements_documentation(
            metric_name="{MetricName}",
            requirements=requirements,
            test_file_name="test_{metric_name}_migration.py",
        )

        assert True
```

**Key points**:
- Inherit from `BaseMigrationTest` for reusable test methods
- Use fixtures from `conftest.py` (no local fixture definitions)
- `sample_data` comes from notebook testing (working cases)
- Tolerance based on notebook findings
- Add metric-specific behavior tests
- Document requirements

### 3.2 Run Tests
```bash
# Run the new tests
uv run pytest tests/e2e/metrics_migration/test_{metric_name}_migration.py -v -s

# Check they collect properly
uv run pytest tests/e2e/metrics_migration/test_{metric_name}_migration.py --collect-only
```

---

## Phase 4: Code Quality & Finalization

### 4.1 Run Linting & Formatting
```bash
# Format code
make format

# Type check
make type

# Quick health check
make check
```

### 4.2 Run All Tests
```bash
# Unit tests
make test

# E2E tests
make test-e2e

# Or run specific test
uv run pytest tests/e2e/metrics_migration/ -v
```

### 4.3 Update Documentation
**File**: `docs/howtos/migrations/{metric_name}.md` (if needed)

Document:
- Migration rationale
- API changes
- Usage examples (before/after)
- Breaking changes (if any)

### 4.4 Create PR Checklist
- [ ] New metric implementation complete
- [ ] Prompt function with examples
- [ ] E2E migration tests passing
- [ ] Notebook testing completed
- [ ] Code formatted and linted
- [ ] Type checking passes
- [ ] Documentation updated
- [ ] Exports added to `__init__.py`

---

## Key Learnings & Best Practices

### From Context Recall Migration
1. **Components validation**: Base class rejects legacy wrappers automatically
2. **Structured output**: Use Pydantic models with instructor LLMs
3. **Prompt format**: Inline examples with json.dumps() escaping
4. **Score calculation**: Extract to separate method for clarity
5. **Edge cases**: Handle empty inputs gracefully

### From Test Infrastructure
1. **Use shared fixtures**: `conftest.py` provides llm/embeddings
2. **Base test class**: `BaseMigrationTest` eliminates boilerplate
3. **Test utilities**: `test_utils.py` for common operations
4. **Consistent patterns**: All tests follow same structure
5. **Proper skipping**: Tests skip gracefully without API keys

### From Notebook Testing
1. **Manual testing first**: Catches issues before E2E tests
2. **User modifications matter**: Inform final test design
3. **Performance tools**: Use optimized `compare_metrics` function
4. **Diverse test cases**: Cover normal, edge, high/low score scenarios
5. **Iteration speed**: Faster to debug in notebook than pytest

### Tolerance Guidelines
- **LLM-based metrics**: 0.2-0.3 (accounts for randomness)
- **Embeddings-based**: 1e-6 to 1e-10 (deterministic)
- **String/rule-based**: 1e-10 (exact match expected)
- **Adjust based on**: Notebook findings and metric nature

---

## Complete Checklist

### Pre-Migration
- [ ] Study legacy metric implementation thoroughly
- [ ] Identify required components (LLM/embeddings/neither)
- [ ] Document core algorithm logic
- [ ] Note edge cases and special handling
- [ ] Review existing migrated metrics for patterns

### Implementation
- [ ] Create prompt function with examples
- [ ] Define Pydantic output models
- [ ] Implement metric class inheriting from BaseMetric
- [ ] Match legacy calculation logic exactly
- [ ] Handle edge cases (empty, None, errors)
- [ ] Update `__init__.py` exports

### Manual Testing (Notebook)
- [ ] Open general-purpose notebook: `tests/notebooks/metric_score_diff.ipynb`
- [ ] Generate `METRIC_CONFIG` for your metric (Section 2.2)
- [ ] Edit Cell 2 with your configuration
- [ ] Run all cells (no other modifications needed)
- [ ] Review Amnesty QA and FIQA comparison results
- [ ] Iterate on implementation until scores match
- [ ] Document findings (mean |diff|, tolerance, patterns)

### E2E Testing
- [ ] Create test file inheriting from BaseMigrationTest
- [ ] Use fixtures from conftest.py
- [ ] Copy working test cases from notebook
- [ ] Set appropriate tolerance
- [ ] Add metric-specific behavior tests
- [ ] Document requirements
- [ ] Run tests and verify they pass

### Quality & Finalization
- [ ] Run `make format`
- [ ] Run `make type`
- [ ] Run `make check`
- [ ] Run `make test`
- [ ] Run `make test-e2e`
- [ ] Update documentation if needed
- [ ] Create PR with checklist

---

## File Structure Reference

```
ragas/
├── src/ragas/
│   ├── prompts/metrics/
│   │   └── {metric_name}.py          # NEW: Prompt function
│   └── metrics/
│       ├── collections/
│       │   ├── _{metric_name}.py     # NEW: V2 implementation
│       │   └── __init__.py           # MODIFIED: Add export
│       └── _{metric_name}.py         # EXISTING: Legacy implementation
├── tests/
│   ├── utils/                        # EXISTING: Shared utilities
│   │   ├── __init__.py
│   │   └── llm_setup.py
│   ├── notebooks/
│   │   └── metric_score_diff.ipynb  # EXISTING: General-purpose testing notebook
│   └── e2e/metrics_migration/
│       ├── conftest.py               # EXISTING: Shared fixtures
│       ├── test_utils.py             # EXISTING: Test utilities
│       ├── base_migration_test.py   # EXISTING: Base test class
│       └── test_{metric_name}_migration.py  # NEW: E2E tests
└── docs/
    └── howtos/migrations/
        └── {metric_name}.md          # OPTIONAL: Migration guide
```

---

## Success Criteria

✅ **Implementation**:
- New metric produces similar scores to legacy (within tolerance)
- Works only with modern components (rejects legacy wrappers)
- Handles all edge cases properly
- Code is clean, typed, and documented

✅ **Testing**:
- E2E tests pass
- Manual notebook testing completed
- User satisfied with score matching
- All code quality checks pass

✅ **Documentation**:
- Usage examples clear
- Requirements documented
- Migration path explained (if needed)

✅ **Integration**:
- Exports added
- No regressions in existing tests
- Ready for PR and review

---

This plan provides a complete, battle-tested workflow for migrating any metric from legacy to modern implementation, incorporating all learnings from previous migrations and leveraging the full testing infrastructure.
