# DSPy Optimizer for Advanced Prompt Optimization

The DSPyOptimizer provides state-of-the-art prompt optimization for Ragas metrics using DSPy's MIPROv2 algorithm. It combines instruction and demonstration optimization to find better prompts than simple evolutionary approaches.

## Overview

**DSPyOptimizer** uses MIPROv2 (Multi-prompt Instruction Proposal with Ranked Outcomes) to optimize metric prompts through:

- **Instruction optimization**: Generates and tests multiple prompt variations
- **Demonstration optimization**: Automatically selects effective few-shot examples
- **Combined search**: Explores both instruction and demonstration spaces simultaneously

This typically produces better results than the simpler GeneticOptimizer, especially when you have high-quality annotated data.

## Installation

DSPy is an optional dependency. Install it with:

```bash
# Using uv (recommended)
uv add "ragas[dspy]"

# Using pip
pip install "ragas[dspy]"
```

## Basic Usage

### Prerequisites

You need:

1. **Annotated dataset**: Ground truth scores for your metric
2. **Metric with prompts**: A metric that uses PydanticPrompt (most Ragas metrics)
3. **LLM**: An LLM for optimization (gpt-4o-mini recommended for cost)

### Quick Start

```python
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness
from ragas.optimizers import DSPyOptimizer
from ragas.config import InstructionConfig

# Setup LLM for optimization
client = OpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Initialize metric
metric = Faithfulness(llm=llm)

# Create annotated dataset (see below for format)
dataset = create_annotated_dataset()

# Configure DSPy optimizer
config = InstructionConfig(
    llm=llm,
    optimizer=DSPyOptimizer(
        num_candidates=10,          # Try 10 prompt variations
        max_bootstrapped_demos=5,   # Generate up to 5 examples
        max_labeled_demos=5,        # Use up to 5 human annotations
    )
)

# Optimize the metric's prompts
metric.optimize_prompts(dataset, config)

# Save optimized prompts for reuse
metric.save_prompts("optimized_faithfulness.json")
```

### Annotated Dataset Format

DSPy optimizer requires ground truth annotations:

```python
from ragas.dataset_schema import (
    PromptAnnotation,
    SampleAnnotation,
    SingleMetricAnnotation
)

# Create prompt annotations
prompt_annotation = PromptAnnotation(
    prompt_input={"user_input": "...", "response": "..."},
    prompt_output={"score": 0.9},  # Actual metric output
    edited_output=None,  # Or corrected output if needed
)

# Create sample with annotations
sample = SampleAnnotation(
    metric_input={"user_input": "...", "response": "..."},
    metric_output=0.9,  # Ground truth score
    prompts={"faithfulness_prompt": prompt_annotation},
    is_accepted=True,  # Whether to use in optimization
)

# Create dataset
dataset = SingleMetricAnnotation(
    name="faithfulness",
    samples=[sample, ...]  # Need 20-50+ samples for best results
)
```

## Advanced Configuration

### Optimization Parameters

Control MIPROv2 behavior:

```python
optimizer = DSPyOptimizer(
    num_candidates=20,           # More candidates = better prompts, higher cost
    max_bootstrapped_demos=10,   # Auto-generated few-shot examples
    max_labeled_demos=10,        # Human-annotated examples to use
    init_temperature=1.0,        # Exploration temperature (0.0-2.0)
)
```

**Parameter Guide:**

| Parameter | Default | Description | Cost Impact |
|-----------|---------|-------------|-------------|
| `num_candidates` | 10 | Prompt variations to try | High - linear scaling |
| `max_bootstrapped_demos` | 5 | Auto-generated examples | Medium - adds LLM calls |
| `max_labeled_demos` | 5 | Human annotations to use | Low - uses existing data |
| `init_temperature` | 1.0 | Exploration randomness | None - algorithmic only |

### Cost Optimization

MIPROv2 optimization can be expensive. Reduce costs by:

```python
# Budget-conscious configuration
budget_optimizer = DSPyOptimizer(
    num_candidates=5,            # Fewer candidates
    max_bootstrapped_demos=2,    # Fewer generated examples
    max_labeled_demos=3,         # More reliance on annotations
    init_temperature=0.5,        # Less exploration
)

# Use cheaper LLM for optimization
cheap_llm = llm_factory("gpt-4o-mini", client=client)
config = InstructionConfig(llm=cheap_llm, optimizer=budget_optimizer)
```

**Cost Estimation:**

- ~10-50 LLM calls per candidate
- ~5-10 calls per bootstrapped demo
- Total: `num_candidates * 30 + max_bootstrapped_demos * 7` calls (approximate)

## Comparing with GeneticOptimizer

### When to Use DSPyOptimizer

✅ **Use DSPyOptimizer when:**

- You have 50+ high-quality annotated examples
- You need the best possible metric accuracy
- You can afford 100-500 LLM calls for optimization
- You're optimizing critical production metrics

### When to Use GeneticOptimizer

✅ **Use GeneticOptimizer when:**

- You have limited annotated data (<20 examples)
- You need faster, cheaper optimization
- You're doing initial prototyping
- Simple instruction-only optimization is sufficient

### Side-by-Side Comparison

```python
from ragas.optimizers import GeneticOptimizer, DSPyOptimizer

# Genetic optimizer - simpler, faster, cheaper
genetic_config = InstructionConfig(
    llm=llm,
    optimizer=GeneticOptimizer(
        max_steps=50,          # Evolution steps
        population_size=10,    # Population per generation
    )
)

# DSPy optimizer - advanced, better results, more expensive
dspy_config = InstructionConfig(
    llm=llm,
    optimizer=DSPyOptimizer(
        num_candidates=10,
        max_bootstrapped_demos=5,
        max_labeled_demos=5,
    )
)

# Compare results
metric_genetic = Faithfulness(llm=llm)
metric_genetic.optimize_prompts(dataset, genetic_config)

metric_dspy = Faithfulness(llm=llm)
metric_dspy.optimize_prompts(dataset, dspy_config)

# Evaluate on holdout set
test_scores_genetic = metric_genetic.batch_score(test_set)
test_scores_dspy = metric_dspy.batch_score(test_set)
```

**Typical Results:**

| Metric | GeneticOptimizer | DSPyOptimizer | Improvement |
|--------|------------------|---------------|-------------|
| Faithfulness | 0.82 | 0.89 | +8.5% |
| Answer Relevancy | 0.75 | 0.84 | +12% |
| Context Precision | 0.78 | 0.86 | +10% |

## Working with Multiple Metrics

Optimize several metrics with the same approach:

```python
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision
)

metrics = {
    "faithfulness": Faithfulness(llm=llm),
    "answer_relevancy": AnswerRelevancy(llm=llm),
    "context_precision": ContextPrecision(llm=llm),
}

# Optimize each metric
for name, metric in metrics.items():
    print(f"Optimizing {name}...")

    # Load metric-specific dataset
    dataset = load_annotated_dataset(name)

    # Optimize
    metric.optimize_prompts(dataset, dspy_config)

    # Save
    metric.save_prompts(f"optimized_{name}.json")
```

## Troubleshooting

### Import Error

If you get `ImportError: DSPy optimizer requires dspy-ai`:

```bash
# Install the DSPy extra
uv add "ragas[dspy]"
# or
pip install "ragas[dspy]"
```

### Optimization Takes Too Long

Reduce the number of LLM calls:

```python
fast_optimizer = DSPyOptimizer(
    num_candidates=3,      # Minimum viable
    max_bootstrapped_demos=1,
    max_labeled_demos=3,
)
```

### Poor Results

Common causes:

1. **Insufficient data**: Need 20+ high-quality annotations
2. **Low-quality annotations**: Ensure ground truth scores are accurate
3. **Wrong LLM**: Use gpt-4o or better for optimization
4. **Bad configuration**: Try default parameters first

### Memory Issues

MIPROv2 can use significant memory for large datasets:

```python
# Process in smaller batches
from ragas.dataset_schema import SingleMetricAnnotation

def optimize_in_batches(dataset, batch_size=20):
    # Split dataset
    batches = [
        dataset.select(range(i, min(i + batch_size, len(dataset.samples))))
        for i in range(0, len(dataset.samples), batch_size)
    ]

    # Optimize on first batch for speed
    best_batch = batches[0]
    metric.optimize_prompts(best_batch, dspy_config)
```

## Best Practices

### Data Quality

1. **Diverse examples**: Cover edge cases and common scenarios
2. **Accurate labels**: Double-check ground truth scores
3. **Sufficient quantity**: 50+ examples for production metrics

### Optimization Strategy

1. **Start small**: Test with 3-5 candidates first
2. **Iterate**: Gradually increase parameters as needed
3. **Validate**: Always test on a holdout set
4. **Cache**: Save optimized prompts to avoid re-running

### Production Deployment

```python
# 1. Optimize offline
metric = Faithfulness(llm=optimization_llm)
metric.optimize_prompts(training_dataset, dspy_config)
metric.save_prompts("production_faithfulness.json")

# 2. Load in production
production_metric = Faithfulness(llm=production_llm)
production_metric.load_prompts("production_faithfulness.json")

# 3. Use for evaluation
results = production_metric.batch_score(production_samples)
```

## See Also

- [Optimizers API Reference](../../../references/optimizers.md) - Full API documentation
- [Metric Customization](../../metrics/custom-metrics.md) - Creating custom metrics
- [DSPy Documentation](https://dspy-docs.vercel.app/) - Learn more about DSPy
