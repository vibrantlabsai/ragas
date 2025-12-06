# How to Evaluate Your Prompt and Improve It

In this guide, you'll learn how to evaluate and iteratively improve a prompt using Ragas.

## What you'll accomplish
- Iterate and improve a prompt based on error analysis of evals
- Establish clear decision criterias to choose between prompts
- Build a reusable evaluation pipeline for your dataset
- Learn how to leverage Ragas to build your evaluation pipeline

!!! note "Full code"
    - The dataset and scripts live under `examples/iterate_prompt/` in the repo
    - Full code is available on [GitHub](https://github.com/vibrantlabsai/ragas/tree/main/examples/iterate_prompt)

## Task definition
In this case, we are considering a customer support ticket classification task.

- Labels (multi-label): `Billing`, `Account`, `ProductIssue`, `HowTo`, `Feature`, `RefundCancel`
- Priority (exactly one): `P0`, `P1`, or `P2`


## Dataset 

We've created a synthetic dataset for our use case. Each row has `id, text, labels, priority`. Example rows from the dataset:

| id | text                                                                                                                | labels                 | priority |
|----|---------------------------------------------------------------------------------------------------------------------|------------------------|----------|
| 1  | Upgraded to Plus… bank shows two charges the same day; want the duplicate reversed.                                | Billing;RefundCancel   | P1       |
| 2  | SSO via Okta succeeds then bounces back to /login; colleagues can sign in; state mismatch; blocked from boards.    | Account;ProductIssue   | P0       |
| 3  | Need to export a board to PDF with comments and page numbers for audit; deadline next week.                         | HowTo                  | P2       |

To customize the dataset for your use case, create a `datasets/` directory and add your own CSV file. You can also connect to different backends. Refer to [Core Concepts - Evaluation Dataset](../../concepts/components/eval_dataset.md) for more information.

It is better to sample real data from your application to create the dataset. If that is not available, you can generate synthetic data using an LLM. We recommend using a reasoning model like gpt-5 high-reasoning which can generate more accurate and complex data. Always make sure to manually review and verify the data you use. 

## Evaluate your prompt on a dataset

### Prompt runner

First, we'll run the prompt on one case to test if everything works. 

??? example "See full prompt v1 here"
    ```text
    You categorize a short customer support ticket into (a) one or more labels and (b) a single priority.
    
    Allowed labels (multi-label):
    - Billing: charges, taxes (GST/VAT), invoices, plans, credits.
    - Account: login/SSO, password reset, identity/email/account merges.
    - ProductIssue: malfunction (crash, error code, won't load, data loss, loops, outages).
    - HowTo: usage questions ("where/how do I…", "where to find…").
    - Feature: new capability or improvement request.
    - RefundCancel: cancel/terminate and/or refund requests.
    - AbuseSpam: insults/profanity/spam (not mild frustration).
    
    Priority (exactly one):
    - P0 (High): blocked from core action or money/data at risk.
    - P1 (Normal): degraded/needs timely help, not fully blocked.
    - P2 (Low): minor/info/how-to/feature.
    
    Return exactly in JSON:
    {"labels":[<labels>], "priority":"P0"|"P1"|"P2"}
    ```

```bash
cd examples/iterate_prompt
export OPENAI_API_KEY=your_openai_api_key
uv run run_prompt.py
```

This will run the prompt on sample case and print the results.

??? example "Sample output"
    ```
    $ uv run run_prompt.py                      

    Test ticket:
    "SSO via Okta succeeds then bounces me back to /login with no session. Colleagues can sign in. I tried clearing cookies; same result. Error in devtools: state mismatch. I'm blocked from our boards."

    Response:
    {"labels":["Account","ProductIssue"], "priority":"P0"}
    ```


### Metrics for scoring

It is generally better to use a simpler metric instead of a complex one. You should use a metric relevant to your use case. More information on metrics can be found in [Core Concepts - Metrics](../../concepts/metrics/index.md). Here we use two discrete metrics: `labels_exact_match` and `priority_accuracy`. Keeping them separate helps analyze and fix different failure modes.

- `priority_accuracy`: Checks whether the predicted priority matches the expected priority; important for correct urgency triage.
- `labels_exact_match`: Checks whether the set of predicted labels exactly matches the expected labels; important to avoid over/under-tagging and helps us measure the accuracy of our system in labeling the cases.

```python
# examples/iterate_prompt/evals.py
import json
from ragas.metrics.discrete import discrete_metric
from ragas.metrics.result import MetricResult

@discrete_metric(name="labels_exact_match", allowed_values=["correct", "incorrect"])
def labels_exact_match(prediction: str, expected_labels: str):
    try:
        predicted = set(json.loads(prediction).get("labels", []))
        expected = set(expected_labels.split(";")) if expected_labels else set()
        return MetricResult(
            value="correct" if predicted == expected else "incorrect",
            reason=f"Expected={sorted(expected)}; Got={sorted(predicted)}",
        )
    except Exception as e:
        return MetricResult(value="incorrect", reason=f"Parse error: {e}")

@discrete_metric(name="priority_accuracy", allowed_values=["correct", "incorrect"])
def priority_accuracy(prediction: str, expected_priority: str):
    try:
        predicted = json.loads(prediction).get("priority")
        return MetricResult(
            value="correct" if predicted == expected_priority else "incorrect",
            reason=f"Expected={expected_priority}; Got={predicted}",
        )
    except Exception as e:
        return MetricResult(value="incorrect", reason=f"Parse error: {e}")
```

### The experiment function

The experiment function is used to run the prompt on a dataset. More information on experimentation can be found in [Core Concepts - Experimentation](../../concepts/experimentation.md).

Notice that we are passing `prompt_file` as a parameter so that we can run experiments with different prompts. You can also pass other parameters to the experiment function like model, temperature, etc. and experiment with different configurations. It is recommended to change only 1 parameter at a time while doing experimentation.

```python
# examples/iterate_prompt/evals.py
import asyncio, json
from ragas import experiment
from run_prompt import run_prompt

@experiment()
async def support_triage_experiment(row, prompt_file: str, experiment_name: str):
    response = await asyncio.to_thread(run_prompt, row["text"], prompt_file=prompt_file)
    try:
        parsed = json.loads(response)
        predicted_labels = ";".join(parsed.get("labels", [])) or ""
        predicted_priority = parsed.get("priority")
    except Exception:
        predicted_labels, predicted_priority = "", None

    return {
        "id": row["id"],
        "text": row["text"],
        "response": response,
        "experiment_name": experiment_name,
        "expected_labels": row["labels"],
        "predicted_labels": predicted_labels,
        "expected_priority": row["priority"],
        "predicted_priority": predicted_priority,
        "labels_score": labels_exact_match.score(prediction=response, expected_labels=row["labels"]).value,
        "priority_score": priority_accuracy.score(prediction=response, expected_priority=row["priority"]).value,
    }
```

### Dataset loader (CSV)

The dataset loader is used to load the dataset into a Ragas dataset object. More information on datasets can be found in [Core Concepts - Evaluation Dataset](../../concepts/components/eval_dataset.md).

```python
# examples/iterate_prompt/evals.py
import os, pandas as pd
from ragas import Dataset

def load_dataset():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(current_dir, "datasets", "support_triage.csv"))
    dataset = Dataset(name="support_triage", backend="local/csv", root_dir=".")
    for _, row in df.iterrows():
        dataset.append({
            "id": str(row["id"]),
            "text": row["text"],
            "labels": row["labels"],
            "priority": row["priority"],
        })
    return dataset
```

### Run the experiment using current prompt

```bash
uv run evals.py run --prompt_file promptv1.txt
```

This will run the given prompt on the dataset and save the results to `experiments/` directory.

??? example "Sample output"
    ```
    $ uv run evals.py run --prompt_file promptv1.txt        
    
    Loading dataset...
    Dataset loaded with 20 samples
    Running evaluation with prompt file: promptv1.txt
    Running experiment: 100%|██████████████████████████████████████████████████████████████████| 20/20 [00:11<00:00,  1.79it/s]
    ✅ promptv1: 20 cases evaluated
    Results saved to: experiments/20250826-041332-promptv1.csv
    promptv1 Labels Accuracy: 80.00%
    promptv1 Priority Accuracy: 75.00%
    ```

## Improve the prompt

### Analyze errors from the result

Open `experiments/{timestamp}-promptv1.csv` in your favorite spreadsheet editor and analyze the results. Look for cases where the labels_score or priority_score is incorrect.

From our promptv1 experiment, we can identify several error patterns:

#### Priority Errors: Over-prioritization (P1 → P0)
The model consistently assigns P0 (highest priority) to billing-related issues that should be P1:

| Case | Issue | Expected | Got | Pattern |
|------|-------|----------|-----|---------|
| ID 19 | Auto-charge after pausing workspace | P1 | P0 | Billing dispute treated as urgent |
| ID 1 | Duplicate charge on same day | P1 | P0 | Billing dispute treated as urgent |
| ID 5 | Cancellation with refund request | P1 | P0 | Routine cancellation treated as urgent |
| ID 13 | Follow-up on cancellation | P1 | P0 | Follow-up treated as urgent |

**Pattern**: The model treats any billing/refund/cancellation as urgent (P0) when most are routine business operations (P1).

#### Label Errors: Over-labeling and confusion

| Case | Issue | Expected | Got | Pattern |
|------|-------|----------|-----|---------|
| ID 9 | GST tax question from US user | `Billing;HowTo` | `Billing;Account` | Confuses informational questions with account actions |
| ID 10 | Account ownership transfer | `Account` | `Account;Billing` | Adds Billing when money/plans mentioned |
| ID 20 | API rate limit question | `ProductIssue;HowTo` | `ProductIssue;Billing;HowTo` | Adds Billing when plans mentioned |
| ID 16 | Feature request for offline mode | `Feature` | `Feature;HowTo` | Adds HowTo for feature requests |

**Patterns identified**:

1. **Over-labeling with Billing**: Adds "Billing" even when not primarily billing-related
2. **HowTo vs Account confusion**: Misclassifies informational questions as account management actions  
3. **Over-labeling with HowTo**: Adds "HowTo" to feature requests when users ask "how" but mean "can you build this"



### Improve the prompt

Based on our error analysis, we'll create `promptv2_fewshot.txt` with targeted improvements. You can use an LLM to generate the prompt or edit it manually. In this case, we passed the error patterns and the original prompt to an LLM to generate a revised prompt with few-shot examples.

#### Key additions in promptv2_fewshot:

**1. Enhanced Priority Guidelines with Business Impact Focus:**
```
- P0: Blocked from core functionality OR money/data at risk OR business operations halted
- P1: Degraded experience OR needs timely help BUT has workarounds OR not fully blocked  
- P2: Minor issues OR information requests OR feature requests OR non-urgent how-to
```

**2. Conservative Multi-labeling Rules to Prevent Over-tagging:**
```
## Multi-label Guidelines
Use single label for PRIMARY issue unless both aspects are equally important:
- Billing + RefundCancel: Always co-label. Cancellation/refund requests must include Billing.  
- Account + ProductIssue: For auth/login malfunctions (loops, "invalid_token", state mismatch, bounce-backs)
- Avoid adding Billing to account-only administration unless there is an explicit billing operation

Avoid over-tagging: Focus on which department should handle this ticket first.
```

**3. Detailed Priority Guidelines with Specific Scenarios:**
```
## Priority Guidelines  
- Ignore emotional tone - focus on business impact and available workarounds
- Billing disputes/adjustments (refunds, duplicate charges, incorrect taxes/pricing) = P1 unless causing an operational block
- Login workarounds: If Incognito/another account works, prefer P1; if cannot access at all, P0
- Core business functions failing (webhooks, API, sync) = P0
```

**4. Comprehensive Examples with Reasoning:**
Added 7 examples covering different scenarios with explicit reasoning to demonstrate proper classification. 

```md
## Examples with Reasoning

Input: "My colleague left and I need to change the team lead role to my email address."
Output: {"labels":["Account"], "priority":"P1"}
Reasoning: Administrative role change; avoid adding Billing unless a concrete billing action is requested.

Input: "Dashboard crashes when I click reports tab, but works fine in mobile app."
Output: {"labels":["ProductIssue"], "priority":"P1"}
Reasoning: Malfunction exists but workaround available (mobile app works); single label since primary issue is product malfunction.
```


!!! tip "Try to not directly add the examples from the dataset as that can lead to overfitting to dataset and your prompt might fail in other cases."


### Evaluate new prompt

After creating `promptv2_fewshot.txt` with the improvements, run the experiment with the new prompt:

```bash
uv run evals.py run --prompt_file promptv2_fewshot.txt
```

This will evaluate the improved prompt on the same dataset and save results to a new timestamped file.

??? example "Sample output"
    ```
    $ uv run evals.py run --prompt_file promptv2_fewshot.txt
    
    Loading dataset...
    Dataset loaded with 20 samples
    Running evaluation with prompt file: promptv2_fewshot.txt
    Running experiment: 100%|██████████████████████████████████████████████████████████████| 20/20 [00:11<00:00,  1.75it/s]
    ✅ promptv2_fewshot: 20 cases evaluated
    Results saved to: experiments/20250826-231414-promptv2_fewshot.csv
    promptv2_fewshot Labels Accuracy: 90.00%
    promptv2_fewshot Priority Accuracy: 95.00%
    ```

The experiment will create a new CSV file in the `experiments/` directory with the same structure as the first run, allowing for direct comparison.


### Analyze and compare results

We've created a simple utility function to take in multiple CSVs and combine it so that we can compare it easily:

```bash
uv run evals.py compare --inputs experiments/20250826-041332-promptv1.csv experiments/20250826-231414-promptv2_fewshot.csv 
```
This prints the accuracy for each experiment and saves a combined CSV file in `experiments/` directory.

??? Sample output
    ```bash
    $ uv run evals.py compare --inputs experiments/20250826-041332-promptv1.csv experiments/20250826-231414-promptv2_fewshot.csv 

    promptv1 Labels Accuracy: 80.00%
    promptv1 Priority Accuracy: 75.00%
    promptv2_fewshot Labels Accuracy: 90.00%
    promptv2_fewshot Priority Accuracy: 95.00%
    Combined comparison saved to: experiments/20250826-231545-comparison.csv
    ```

Here, we can see that promptv2_fewshot has improved the accuracy of both labels and priority. But we can also see that some cases still fail. We can analyze the errors and improve the prompt further.

Stop iterating when improvements plateau or accuracy meets business requirements.

!!! tip "If you hit a ceiling on improving accuracy with just the prompt improvements, you can try experiments with better models." 

## Apply this loop to your use case
- Create dataset, metrics, experiment for your use case
- Run evaluation and analyze errors
- Improve prompt based on the error analysis
- Re-run evaluation and compare results
- Stop when improvements plateau or accuracy meets business requirements

Once you have your dataset and evaluation loop setup, you can expand this to testing more parameters like model, etc. 

The Ragas framework handles the orchestration, parallel execution, and result aggregation automatically for you, helping you evaluate and focus on your use case!

!!! tip "Advanced: Aligning LLM judges"
    If you're using LLM-based metrics for evaluation, consider aligning your judge with human expert judgments first to ensure reliable evaluation. See [How to Align an LLM as a Judge](../applications/align-llm-as-judge.md).