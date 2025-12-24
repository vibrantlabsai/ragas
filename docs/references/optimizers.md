# Optimizers API Reference

Ragas provides optimizers to improve metric prompts through automated optimization. This page documents the available optimizer classes and their configuration.

## Overview

Optimizers use annotated datasets with ground truth scores to refine metric prompts, improving accuracy through:

- **Instruction optimization**: Finding better prompt wording
- **Demonstration optimization**: Selecting effective few-shot examples
- **Search strategies**: Exploring the prompt space efficiently

## Core Classes

::: ragas.optimizers
    options:
        members:
            - Optimizer
            - GeneticOptimizer
            - DSPyOptimizer

## GeneticOptimizer

Simple evolutionary optimizer for prompt instructions.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_steps` | `int` | 50 | Maximum evolution steps |
| `population_size` | `int` | 10 | Population size per generation |
| `mutation_rate` | `float` | 0.2 | Probability of mutation |

### Usage

```python
from ragas.optimizers import GeneticOptimizer
from ragas.config import InstructionConfig

optimizer = GeneticOptimizer(
    max_steps=50,
    population_size=10,
)

config = InstructionConfig(llm=llm, optimizer=optimizer)
metric.optimize_prompts(dataset, config)
```

### How it Works

1. Generates population of prompt variations
2. Evaluates each on annotated dataset
3. Selects best performers
4. Creates next generation via crossover and mutation
5. Repeats for max_steps iterations

**Pros**: Simple, works with limited data
**Cons**: Slower convergence, instruction-only

## DSPyOptimizer

Advanced optimizer using DSPy's [MIPROv2](https://dspy.ai/api/optimizers/MIPROv2/) algorithm.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_candidates` | `int` | 10 | Number of prompt variants to try |
| `max_bootstrapped_demos` | `int` | 5 | Max auto-generated examples |
| `max_labeled_demos` | `int` | 5 | Max human-annotated examples |
| `init_temperature` | `float` | 1.0 | Exploration temperature (0.0-2.0) |

### Usage

```python
from ragas.optimizers import DSPyOptimizer
from ragas.config import InstructionConfig

optimizer = DSPyOptimizer(
    num_candidates=10,
    max_bootstrapped_demos=5,
    max_labeled_demos=5,
)

config = InstructionConfig(llm=llm, optimizer=optimizer)
metric.optimize_prompts(dataset, config)
```

### How it Works

1. Generates candidate prompt instructions
2. Bootstraps few-shot demonstrations from data
3. Selects best human-annotated examples
4. Evaluates all combinations on dataset
5. Returns best-performing configuration

Learn more about DSPy concepts:
- [Signatures](https://dspy.ai/learn/programming/signatures/) - DSPy's approach to defining input/output specifications
- [Optimizers](https://dspy.ai/learn/optimization/optimizers/) - Algorithms for improving prompts and LM weights
- [Modules](https://dspy.ai/learn/programming/modules/) - Building blocks for LLM programs

**Pros**: Better results, combines instructions + demos
**Cons**: Requires DSPy installation, more LLM calls

### Installation

[DSPy](https://dspy.ai/) is an optional dependency:

```bash
# Using uv (recommended)
uv add "ragas[dspy]"

# Using pip
pip install "ragas[dspy]"
```

### Cost Estimation

Approximate LLM calls per optimization:

```
Total calls ≈ num_candidates × 30 + max_bootstrapped_demos × 7
```

Examples:

- Default config (10, 5, 5): ~335 calls
- Budget config (5, 2, 3): ~164 calls
- Aggressive config (20, 10, 10): ~670 calls

## Optimizer Base Class

::: ragas.optimizers.base.Optimizer
    options:
        show_source: false
        members:
            - optimize

## Configuration

Both optimizers are used with `InstructionConfig`:

```python
from ragas.config import InstructionConfig

config = InstructionConfig(
    llm=llm,                      # LLM for optimization
    optimizer=optimizer_instance, # Optimizer to use
)

# Use with metric
metric.optimize_prompts(dataset, config)
```

## Dataset Format

Optimizers require annotated datasets with ground truth scores:

```python
from ragas.dataset_schema import (
    PromptAnnotation,
    SampleAnnotation,
    SingleMetricAnnotation
)

# Create annotated sample
prompt_annotation = PromptAnnotation(
    prompt_input={"user_input": "...", "response": "..."},
    prompt_output={"score": 0.9},
    edited_output=None,  # Optional: corrected output
)

sample = SampleAnnotation(
    metric_input={"user_input": "...", "response": "..."},
    metric_output=0.9,  # Ground truth score
    prompts={"metric_prompt": prompt_annotation},
    is_accepted=True,  # Include in optimization
)

# Create dataset
dataset = SingleMetricAnnotation(
    name="metric_name",
    samples=[sample, ...]  # 20-50+ samples recommended
)
```

## Loss Functions

Optimizers use loss functions to evaluate prompt quality:

```python
from ragas.losses import MSELoss, HuberLoss

# Mean Squared Error (default)
loss = MSELoss()

# Huber Loss (robust to outliers)
loss = HuberLoss(delta=1.0)

# Use with config
config = InstructionConfig(llm=llm, optimizer=optimizer, loss=loss)
```

## Comparison

| Feature | GeneticOptimizer | DSPyOptimizer |
|---------|------------------|---------------|
| Installation | Built-in | Requires `ragas[dspy]` |
| Optimization Target | Instructions only | Instructions + Demos |
| Min Dataset Size | 10+ samples | 20+ samples |
| Typical LLM Calls | 100-500 | 200-700 |
| Accuracy Improvement | +5-8% | +8-12% |
| Best For | Quick optimization | Production metrics |

## See Also

- [DSPy Optimizer Guide](../howtos/customizations/optimizers/dspy-optimizer.md) - Detailed usage
- [Metric Customization](../howtos/customizations/metrics/custom-metrics.md) - Creating metrics
- [Prompt API Reference](./prompt.md) - Understanding prompts

## Additional Resources

**DSPy Documentation:**
- [DSPy Official Documentation](https://dspy.ai/) - Complete guide to DSPy
- [MIPROv2 API Reference](https://dspy.ai/api/optimizers/MIPROv2/) - Detailed MIPROv2 documentation
- [DSPy Optimizers Overview](https://dspy.ai/learn/optimization/optimizers/) - Guide to all DSPy optimizers
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy) - Source code and examples

**Research Papers:**
- [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://arxiv.org/abs/2406.11695) - MIPROv2 paper
