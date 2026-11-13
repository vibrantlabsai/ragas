# Compare evaluation experiments

Independent means can change when the evaluated cohort changes. If difficult cases disappear after a timeout, an apparently higher score may reflect missing results. Paired comparison matches explicit case IDs, retains a fixed denominator, and gives each case equal weight across repeated runs.

## Compare loaded experiments

```python
from ragas import Experiment
from ragas.comparison import ComparisonPlan, MetricSpec

baseline = Experiment(
    "baseline",
    "inmemory",
    data=[
        {"case_id": "a", "quality": 0.5, "cost": 10.0},
        {"case_id": "b", "quality": 0.8, "cost": 12.0},
    ],
)
candidate = Experiment(
    "candidate",
    "inmemory",
    data=[
        {"case_id": "b", "quality": 0.9, "cost": 11.0},
        {"case_id": "a", "quality": 0.6, "cost": 9.0},
    ],
)
plan = ComparisonPlan(
    metrics=(
        MetricSpec("quality", minimum=0, maximum=1, margin=0.01),
        MetricSpec("cost", direction="lower", minimum=0, maximum=100),
    ),
    mode="observed",
)
report = candidate.compare(baseline, plan=plan)
print(report.verdict)
print(report.to_json())
```

`compare(baseline, candidate, plan=...)` also accepts iterables of dictionaries or Pydantic rows. It reads currently loaded data; it does not run experiments, save data or access a backend. Both displayed means use the same eligible cohort and average per-case means, rather than weighting cases by repeat count.

`mode="observed"` describes the finite benchmark only. The default `mode="bootstrap"` estimates uncertainty using a paired case-level percentile bootstrap. It requires at least 20 eligible cases, non-degenerate differences, and enough resamples in both interval tails. Use observed mode explicitly for a small deterministic acceptance suite.

## Identity and coverage

Case IDs must be nonempty strings. With `repeat_id="repeat"`, every observation must also have an explicit string repeat ID. Duplicate case/repeat keys are errors. The order of rows does not affect the result.

Use `case_fields=("input_version", "tenant")` for stable scalar attributes that must agree across repeats and versions. Without these fields, correct matching IDs remain your responsibility. Missing selected attributes or changed values are invalid inputs. Boolean, integer and floating-point identity values are distinct.

Each metric has its own eligibility mask. A case is eligible only when every expected repeat is present on both sides, successful under the configured status policy, and has a usable numeric score. Missing columns, null and NaN remain missing evidence. Booleans, numeric strings, infinity and values outside declared bounds are rejected. Integer scores that cannot be represented exactly as a float are rejected rather than rounded into ties. Metric magnitudes are limited to `1e150` to keep intermediate arithmetic finite.

By default, required coverage is 100%. An explicit `minimum_coverage=0.9` permits a passing conclusion conditional on the eligible cases. The report always includes exclusions and their reasons. A regression among eligible cases may conservatively block CI even when coverage is incomplete; it is not proof of regression among the omitted outcomes.

Without a manifest, the denominator is the **observed union**. Cases missing from both versions cannot be discovered from those results alone.

## Plan before running and retain failures

`plan_for_dataset` validates a dataset and constructs an expected manifest before evaluation. It copies only selected identifiers and attributes. `ComparisonAdapter` composes with the existing Ragas `Executor`:

```python
from ragas import Dataset
from ragas.comparison import ComparisonAdapter, plan_for_dataset
from ragas.executor import Executor
from ragas.run_config import RunConfig

dataset = Dataset("cases", "inmemory", data=[{"case_id": "a", "input": "hello"}])
prepared = plan_for_dataset(dataset, plan)


async def evaluate_case(row):
    return {"quality": 0.9, "cost": 2.0}


adapter = ComparisonAdapter(evaluate_case, plan=prepared, timeout=30)
executor = Executor(show_progress=False, run_config=RunConfig(max_workers=4))
for row in dataset:
    executor.submit(adapter, row)
rows = await executor.aresults()
```

Ordinary exceptions, cooperative timeouts and invalid outputs become unsuccessful observations. Error text, model reasoning and traces are not copied into these rows. Outputs cannot replace adapter-owned identity/status fields. Known Ragas `MetricResult` objects are unwrapped to their values. External cancellation propagates; the manifest still exposes absent jobs. There are no retries or exactly-once guarantees, and a timeout does not prove a remote side effect did not happen. Blocking or cancellation-suppressing adapters do not have a hard wall-clock deadline.

For repeats, configure the repeat column and supply all planned IDs to `plan_for_dataset(..., repeat_ids=("first", "second"))`. Pass `repeat_id=` when submitting each adapter invocation. Set `pass_repeat_id=True` only when the evaluator accepts that keyword, for example to choose a recorded seed. Execution inputs are owned by the caller; supply independent input objects if evaluators mutate them.

The runnable `examples/paired_comparison.py` uses only local functions. It demonstrates a passing improvement, retained errors, omitted jobs, repeated runs and all report formats. Its exported JSONL files can be compared in a fresh process.

## Results from `evaluate()`

Legacy `EvaluationResult` objects do not contain stable comparison IDs or authoritative execution statuses. Supply explicit keys in each result's original sample order:

```python
from ragas.comparison import EvaluationKey, compare_evaluations

report = compare_evaluations(
    baseline_result,
    candidate_result,
    baseline_keys=(EvaluationKey("a"), EvaluationKey("b")),
    candidate_keys=(EvaluationKey("b"), EvaluationKey("a")),
    plan=plan,
)
```

Selected case fields are read from the original evaluation samples. `plan_for_evaluation_dataset(dataset, keys, plan)` creates an expected manifest before scoring, including explicit repeated cases. Keys must travel with their samples; they are never inferred from positions or response text. Legacy adapters reject plans requiring an execution-status field rather than inventing successful statuses.

## Statistical interpretation

Within each case, corresponding repeat scores are paired and averaged. Bootstrap resampling draws **cases**, retaining each paired difference; repeated runs do not inflate the independent sample count. The interval describes uncertainty across independently sampled, representative cases, conditional on their recorded repeated outcomes. It does not estimate all unobserved within-case randomness. Correlated cases should be grouped into the independent unit before comparison.

Defaults are 5,000 resamples, 95% confidence and Bonferroni adjustment over every predeclared metric/group comparison, including optional and empty groups. The same local seed produces the same resamples independent of input order or global random state. Each tail must have at least 20 expected resamples: `resamples * (1 - confidence) / (2 * family_size) >= 20`. With `adjustment="none"`, the divisor uses a family size of one. Insufficient tail resolution is inconclusive; the system never silently increases compute.

Percentile intervals are approximate. Bonferroni adjusts their requested confidence levels; it does not turn approximate intervals into a distribution-free error guarantee. Small samples, poor sampling, correlated cases and repeated tuning on the same benchmark undermine generalization. See the [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) for paired resampling and percentile-interval background.

Positive oriented differences are better. A required comparison passes when its lower interval bound is at least `-margin` and coverage requirements hold; it regresses when its upper bound is below `-margin`; otherwise it is inconclusive. Observed mode uses the point difference at the same inclusive boundary. Required comparisons gate separately; improvements in one metric do not average away regressions in another. A `SliceSpec` predeclares scalar membership before outcomes are filtered; empty groups remain inconclusive.

## Diagnostics beyond the mean

Reports include distributions of per-case scores and changes, improvement/regression counts relative to the margin, and within-case repeat variability. A direction-flip count reveals cases whose repeat changes cross both sides of the tolerance margin. These are descriptive diagnostics, not additional hypothesis tests.

When a metric declares both `minimum` and `maximum`, missing-outcome sensitivity bounds describe every possible finite-cohort mean completion within that range. They use usable scores from partial cases, treat unsuccessful outcomes as unknown, and may show that omissions could reverse a conditional improvement. These are **completion bounds, not confidence intervals or imputed estimates**. They do not override the configured coverage gate. The manifest/observed-union distinction still applies.

## Files and CI

Use `write_observations(path, rows, plan=plan)` from `ragas.comparison` to export selected evidence from experiments, adapters or `evaluation_rows`. It validates identities, duplicates, score ranges and resource limits, then publishes a private temporary JSONL file atomically. It reads only configured columns. Missing columns and unsuccessful statuses are preserved. NaN scores become JSON null: eligibility is unchanged, but the missing-reason label and fingerprint change. For identical replay fingerprints, compare the exported files both times. Supply `input_paths=(...)` when exporting from existing files to reject source aliases.

```python
from ragas.comparison.io import write_plan, write_report

write_plan("plan.json", plan)
write_report("report.json", report)
write_report("report.md", report, output_format="markdown")
write_report("report.csv", report, output_format="csv")
write_report("report.xml", report, output_format="junit")
```

```sh
ragas compare --baseline baseline.jsonl --candidate candidate.jsonl \
  --plan plan.json --output report.xml --output-format junit

python -m ragas.comparison --baseline baseline.jsonl --candidate candidate.jsonl \
  --plan plan.json --format json
```

Exit codes are `0` pass, `1` regression, `2` inconclusive, `3` invalid input/resource/file error. JUnit exports required regressions as failures and required inconclusive results as errors; optional diagnostics are skipped. Summary CSV exports preserve numeric columns and neutralize formula-like text labels.

JSONL supports files written by `LocalJSONLBackend`. Select `--input-format csv` explicitly for CSV: only configured metric columns are parsed as numbers, empty metric cells become missing values, and IDs/statuses/attributes remain strings. For typed attributes, use JSONL or `Experiment.load(..., data_model=...)`; arbitrary numeric strings in direct Python inputs are not silently coerced.

Plans have a strict versioned schema. Unknown fields, duplicate JSON keys, unsupported versions, malformed CSV and nonfinite configuration values are errors. Report writes use private temporary files and atomic replacement, and output paths cannot alias CLI input paths. Reports omit unselected prompts and outputs; selected case IDs and numeric scores can still be sensitive. A saved manifest includes the selected case attributes you chose.

`ComparisonLimits` supplies caller-owned ceilings for input rows/bytes/strings, cases, repeats, alignment cells, group memberships, report details and total bootstrap work. Input plans cannot increase those ceilings. Detail truncation is explicit and does not skip decisions. Fingerprints identify the selected data and plan, not authenticity or unselected payloads.
