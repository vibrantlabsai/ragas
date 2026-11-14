::: ragas.dataset_schema
    options:
        members_order: "source"

::: ragas.messages
    options:
        members_order: "source"

::: ragas.evaluation.EvaluationResult
    options:
        show_root_heading: True

## CI threshold checks

`EvaluationResult` exposes aggregate scores through `summary()` and can check
those scores against CI thresholds without requiring custom aggregation code.

```python
summary = result.summary()

report = result.check_thresholds(
    min_scores={"faithfulness": 0.8},
    max_scores={"error_rate": 0.2},
    ranges={"answer_relevancy": (0.75, 1.0)},
)

result.assert_thresholds(min_scores={"faithfulness": 0.8})
```

`check_thresholds()` returns a structured report and does not raise for normal
threshold failures. Use `assert_thresholds()` when a failing metric should fail
a CI job.

