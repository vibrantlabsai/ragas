# Async Evaluation

## aevaluate()

::: ragas.evaluation.aevaluate

## Async Usage

Ragas provides both synchronous and asynchronous evaluation APIs to accommodate different use cases:

### Using aevaluate() (Recommended for Production)

For production async applications, use `aevaluate()` to avoid event loop conflicts:

```python
import asyncio
from ragas import aevaluate

async def evaluate_app():
    result = await aevaluate(dataset, metrics)
    return result

# In your async application
result = await evaluate_app()
```

### Using evaluate() in Different Environments

The `evaluate()` function automatically handles both Jupyter notebooks and standard Python environments:

```python
# Works in both Jupyter and standard Python
result = evaluate(dataset, metrics)
```

**How it works:**
- **In Jupyter notebooks:** Automatically schedules evaluation on the existing event loop
- **In standard Python:** Creates a new event loop with `asyncio.run()`

### When to Use aevaluate()

Use `aevaluate()` when you're already in an async context and want to avoid synchronous wrappers:

```python
async def evaluate_multiple_datasets():
    # More efficient in async code
    results1 = await aevaluate(dataset1, metrics)
    results2 = await aevaluate(dataset2, metrics)
    return results1, results2
```
