# Evaluation

## evaluate()

::: ragas.evaluation.evaluate

## Interpreting results with missing scores

With the default `raise_exceptions=False`, an exception while scoring a sample
produces `NaN` for that metric. The mean shown by `EvaluationResult` excludes
`NaN` values separately for each metric; it is not necessarily a mean over the
whole dataset. If every score for a metric is `NaN`, its mean is also `NaN`.

For example, scores of `0.8, NaN, NaN` display as `0.8000`, even though only one
of the three samples contributed to that mean. A `NaN` is not a score of zero:
replacing it with zero changes the aggregation rather than recovering the
missing evaluation.

Check the number of scored samples alongside each mean when comparing runs.
Use `result.scores` to select only metric columns, since `result.to_pandas()`
also includes the original dataset columns:

```python
import pandas as pd

# result is the EvaluationResult returned by evaluate(...).
scores = pd.DataFrame(result.scores)
summary = pd.DataFrame(
    {
        "mean": scores.mean(),
        "n_scored": scores.count(),
        "n_total": len(scores),
    }
)
summary["coverage"] = summary["n_scored"] / summary["n_total"]
print(summary)
```

`n_scored` counts non-missing scores, including valid zero scores; a metric
with no scored samples has `n_scored=0`, `coverage=0`, and a `NaN` mean.
Different metrics can have different coverage in the same run. Similar means
with different coverage are not sufficient evidence that two runs perform
equally well.

Inspect the affected rows with `result.to_pandas()` and the evaluation logs
before comparing those runs. For debugging, use `evaluate(...,
raise_exceptions=True)` to raise scoring exceptions instead of substituting
`NaN`. A metric can also return `NaN` without raising an exception, so this
option alone does not guarantee complete coverage.
