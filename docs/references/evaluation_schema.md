::: ragas.dataset_schema
    options:
        members_order: "source"

::: ragas.messages
    options:
        members_order: "source"

::: ragas.evaluation.EvaluationResult
    options:
        show_root_heading: True

## Missing scores and aggregate means

Evaluation failures return `NaN` when `raise_exceptions=False`. A metric may also
return `NaN` when its score is undefined, for example when faithfulness has no
statements to evaluate.

By default, `evaluate()` and `aevaluate()` exclude these scores from aggregate
means (`missing_score_policy="exclude"`). For example, `[1.0, 0.0]` has a mean of
`0.5`, but `[1.0, NaN]` reports `1.0` because only one row remains in the
denominator. Compare per-row scores as well as the mean when evaluating systems
with different numbers of missing scores.

To count every attempted row in the denominator, explicitly select zero for
missing scores:

```python
result = evaluate(dataset, metrics=metrics, missing_score_policy="zero")
# Scores [1.0, NaN] now report a mean of 0.5.
```

This policy replaces `NaN` with zero only during aggregation. It applies to
both failed and undefined scores, and reports `0.0` when every score is `NaN`.
Use it when zero is an appropriate penalty for your metrics; for metrics where
lower scores are better, zero may reward missing scores instead.

The original `NaN` values remain available in `result.scores`,
`result[metric_name]`, and `result.to_pandas()`. The policy can also be passed
directly to `EvaluationResult`. It does not suppress exceptions when
`raise_exceptions=True`, and it does not change the raw scores returned by
`return_executor=True`.

