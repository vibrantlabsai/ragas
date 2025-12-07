# How to Align an LLM as a Judge

In this guide, you'll learn how to systematically evaluate and align an LLM-as-judge metric with human expert judgments using Ragas.

- Build a reusable evaluation pipeline for judge alignment
- Analyze disagreement patterns between judge and human labels
- Iterate on judge prompts to improve alignment with expert decisions

## Why align your LLM judge first?

Before running evaluation experiments, it is important to align your LLM judge to your specific use case. A misaligned judge is like a compass pointing the wrong way - every improvement you make based on its guidance moves you further from your goal. Aligning the judge to match expert judgments ensures you're improving what actually matters. This alignment step is the foundation of reliable evaluation. 

!!! tip "The real value: Looking at your data"
    While building an aligned LLM judge is useful, the true business value comes from systematically analyzing your data and understanding failure patterns. The judge alignment process forces you to deeply examine edge cases, clarify evaluation criteria, and uncover insights about what makes responses good or bad. Think of the judge as a tool that scales your analysis, not a replacement for it.

## Setup your environment

We've created a simple module you can install and run so that you can focus on understanding the evaluation process instead of creating the application.

```bash
uv pip install "ragas[examples]"
export OPENAI_API_KEY="your-api-key-here"
```

!!! note "Full code"
    You can view the full code for the judge alignment evaluation pipeline [here](https://github.com/vibrantlabsai/ragas/tree/main/examples/ragas_examples/judge_alignment).

## Understand the dataset

We'll use the [EvalsBench dataset](https://github.com/vibrantlabsai/EvalsBench/blob/main/data/benchmark_df.csv) which contains expert-annotated examples of LLM responses to business questions. Each row includes:

- `question`: The original question asked
- `grading_notes`: Key points that should be covered in a good response
- `response`: The LLM's generated response
- `target`: Human expert's binary judgment (pass/fail)

**Download the dataset:**

```bash
# Create datasets folder and download the dataset
mkdir -p datasets
curl -o datasets/benchmark_df.csv https://raw.githubusercontent.com/vibrantlabsai/EvalsBench/main/data/benchmark_df.csv
```

**Load and examine the dataset:**

```python
import pandas as pd
from ragas import Dataset

def load_dataset(csv_path: str = None) -> Dataset:
    """Load annotated dataset with human judgments.
    
    Expected columns: question, grading_notes, response, target (pass/fail)
    """
    path = csv_path or "datasets/benchmark_df.csv"
    df = pd.read_csv(path)

    dataset = Dataset(name="llm_judge_alignment", backend="local/csv")
    
    for _, row in df.iterrows():
        dataset.append({
            "question": row["question"],
            "grading_notes": row["grading_notes"],
            "response": row["response"],
            "target": (row["target"]),
        })
    
    return dataset

# Load the dataset
dataset = load_dataset()
print(f"Dataset loaded with {len(dataset)} samples")
```

**Sample rows from the dataset:**

| question | grading_notes | response | target |
|----------|---------------|----------|---------|
| What are the key methods for determining the pre-money valuation of a tech startup before a Series A investment round, and how do they differ? | DCF method: !future cash flows!, requires projections; Comp. analysis: similar co. multiples; VC method: rev x multiple - post-$; *Founder's share matter*; strategic buyers pay more. | Determining the pre-money valuation of a tech startup before a Series A investment round is a critical step... (covers DCF, comparable analysis, VC method) | pass |
| What key metrics and strategies should a startup prioritize to effectively manage and reduce churn rate in a subscription-based business model? | Churn:! monitor monthly, <5% ideal. *Retention strategies*: engage users, improve onboarding. CAC & LTV: balance 3:1+. Feedback loops: implement early. *Customer support*: proactive & responsive, critical. | Managing and reducing churn rate in a subscription-based business model is crucial... (missing specific metrics and strategies) | fail |

The dataset includes multiple responses to the same questions - some pass and others fail. This helps the judge learn nuanced distinctions between acceptable and unacceptable responses.

!!! info "Understanding your ground truth"
    The quality of judge alignment depends entirely on the quality of your ground truth labels. In production scenarios, involve a **principal domain expert** - the person whose judgment is most critical for your use case (e.g., a psychologist for mental health AI, a lawyer for legal AI, or a customer service director for support chatbots). Their consistent judgment becomes the gold standard your judge aligns to. You don't need every example labeled - a representative sample (100-200 examples covering diverse scenarios) is sufficient for reliable alignment.

## Understand the evaluation approach

In this guide, we evaluate pre-existing responses from the dataset rather than generating new ones. This approach ensures reproducible results across evaluation runs, allows us to focus on judge alignment rather than response generation.

The evaluation workflow is: **Dataset row (question + response) → Judge → Compare with human target**

## Define evaluation metrics

For judge alignment, we need two metrics:

**Primary metric: `accuracy` (LLM judge)** - Evaluates responses and returns pass/fail decisions with reason.

**Alignment metric: `judge_alignment`** - Checks if the judge's decision matches the human expert's verdict.

### Setting up the judge metric

Define a simple baseline judge metric that evaluates responses against grading notes:

```python
from ragas.metrics import DiscreteMetric

# Define the judge metric with a simple baseline prompt
accuracy_metric = DiscreteMetric(
    name="accuracy",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\n\nResponse: {response}\nGrading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)
```

### The alignment metric

The alignment metric compares the judge's decision with the human verdict:

```python
from ragas.metrics.discrete import discrete_metric
from ragas.metrics.result import MetricResult

@discrete_metric(name="judge_alignment", allowed_values=["pass", "fail"])
def judge_alignment(judge_label: str, human_label: str) -> MetricResult:
    """Compare judge decision with human label."""
    judge = judge_label.strip().lower()
    human = human_label.strip().lower()
    
    if judge == human:
        return MetricResult(value="pass", reason=f"Judge={judge}; Human={human}")
    
    return MetricResult(value="fail", reason=f"Judge={judge}; Human={human}")
```

## The experiment function

The [experiment function](/concepts/experimentation) orchestrates the complete evaluation pipeline - evaluating responses with the judge and measuring alignment:

```python
from typing import Dict, Any
from ragas import experiment
from ragas.metrics import DiscreteMetric
from ragas_examples.judge_alignment import judge_alignment  # The metric we created above

@experiment()
async def judge_experiment(
    row: Dict[str, Any],
    accuracy_metric: DiscreteMetric,
    llm,
):
    """Run complete evaluation: Judge → Compare with human."""
    # Step 1: Get response (in production, this is where you'd call your LLM app)
    # For this evaluation, we use pre-existing responses from the dataset
    app_response = row["response"]
    
    # Step 2: Judge evaluates the response
    judge_score = await accuracy_metric.ascore(
        question=row["question"],
        grading_notes=row["grading_notes"],
        response=app_response,
        llm=llm,
    )

    # Step 3: Compare judge decision with human target
    alignment = judge_alignment.score(
        judge_label=judge_score.value,
        human_label=row["target"]
    )

    return {
        **row,
        "judge_label": judge_score.value,
        "judge_reason": judge_score.reason,
        "alignment": alignment.value,
        "alignment_reason": alignment.reason,
    }
```

## Run baseline evaluation

### Execute evaluation pipeline and collect results

```python
import os
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas_examples.judge_alignment import load_dataset

# Load dataset
dataset = load_dataset()
print(f"Dataset loaded with {len(dataset)} samples")

# Initialize LLM client
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
llm = llm_factory("gpt-4o-mini", client=openai_client)

# Run the experiment
results = await judge_experiment.arun(
    dataset,
    name="judge_baseline_v1_gpt-4o-mini",
    accuracy_metric=accuracy_metric,
    llm=llm,
)

# Calculate alignment rate
passed = sum(1 for r in results if r["alignment"] == "pass")
total = len(results)
print(f"✅ Baseline alignment: {passed}/{total} passed ({passed/total:.1%})")
```

??? "📋 Output (baseline v1)"

    ```text
    2025-10-08 22:40:00,334 - Loaded dataset with 160 samples
    2025-10-08 22:40:00,334 - Initializing LLM client with model: gpt-4o-mini
    2025-10-08 22:40:01,858 - Running baseline evaluation...
    Running experiment: 100%|████████████████████████| 160/160 [04:35<00:00,  1.72s/it]
    2025-10-08 22:44:37,149 - ✅ Baseline alignment: 121/160 passed (75.6%)
    ```

### Initial performance analysis

The evaluation generates comprehensive CSV results containing all inputs (question, grading_notes, response), human targets, judge decisions with reasoning, and alignment comparisons.

## Analyze errors and failure patterns

After running the baseline evaluation, we can analyze the misalignment patterns to understand where the judge disagrees with human experts.

**Baseline performance: 75.6% alignment (121/160 correct)**

Let's examine the error distribution

??? admonition "📋 Code"

    ```python
    import pandas as pd

    # Load results
    df = pd.read_csv('experiments/judge_baseline_v1_gpt-4o-mini.csv')

    # Analyze misalignments
    false_positives = len(df[(df['judge_label'] == 'pass') & (df['target'] == 'fail')])
    false_negatives = len(df[(df['judge_label'] == 'fail') & (df['target'] == 'pass')])

    print(f"False positives (judge too lenient): {false_positives}")
    print(f"False negatives (judge too strict): {false_negatives}")
    ```

    📋 Output

    ```text
    False positives (judge too lenient): 39
    False negatives (judge too strict): 0
    ```

**Key observation:** All 39 misalignments (24.4%) are false positives - cases where the judge said "pass" but human experts said "fail". The baseline judge is too lenient, missing responses that omit critical concepts from the grading notes.

### Sample failure cases

Here are examples where the judge incorrectly passed responses that were missing key concepts:

| Grading Notes | Human Label | Judge Label | What's Missing |
|---------------|-------------|-------------|----------------|
| `*Valuation caps*, $, post-$ val key. Liquidation prefs: 1x+ common. Anti-dilution: *full vs. weighted*. Board seats: 1-2 investor reps. ESOP: 10-20%.` | fail | pass | Response discusses all points comprehensively but human annotators marked it as fail for subtle omissions |
| `*Impact on valuation*: scalability potential, dev costs, integration ease. !Open-source vs proprietary issues. !Tech debt risks. Discuss AWS/GCP/Azure...` | fail | pass | Missing specific discussion of post-money valuation impact |
| `Historical vs. forecasted rev; top-down & bottom-up methods; *traction evidence*; !unbiased assumptions; 12-24mo project...` | fail | pass | Missing explicit mention of traction evidence |

**Common patterns in errors:**

1. **Missing 1-2 specific concepts** from grading notes while covering others
2. **Implicit vs explicit coverage** - judge accepts implied concepts, we want explicit mentions
3. **Abbreviated terms** not properly decoded (e.g., "mkt demand" = market demand, "post-$" = post-money valuation)
4. **Critical markers ignored** - points marked with `*` or `!` are often essential

## Improve the judge prompt

Based on error analysis, we need to create an improved prompt that:

1. **Understands abbreviations** used in grading notes
2. **Recognizes critical markers** (`*`, `!`, specific numbers)
3. **Requires all concepts** to be present, not just most
4. **Accepts semantic equivalents** (different wording for same concept)
5. **Balances strictness** - not too lenient or too strict

### Create the improved v2 prompt

Define the enhanced judge metric with comprehensive evaluation criteria:

```python
from ragas.metrics import DiscreteMetric

# Define improved judge metric with enhanced evaluation criteria
accuracy_metric_v2 = DiscreteMetric(
    name="accuracy",
    prompt="""Evaluate if the response covers ALL the key concepts from the grading notes. Accept semantic equivalents but carefully check for missing concepts.

ABBREVIATION GUIDE - decode these correctly:

• Financial: val=valuation, post-$=post-money, rev=revenue, ARR/MRR=Annual/Monthly Recurring Revenue, COGS=Cost of Goods Sold, Opex=Operating Expenses, LTV=Lifetime Value, CAC=Customer Acquisition Cost
• Business: mkt=market, reg/regs=regulation/regulatory, corp gov=corporate governance, integr=integration, S&M=Sales & Marketing, R&D=Research & Development, acq=acquisition
• Technical: sys=system, elim=elimination, IP=Intellectual Property, TAM=Total Addressable Market, diff=differentiation
• Metrics: NPS=Net Promoter Score, SROI=Social Return on Investment, proj=projection, cert=certification

EVALUATION APPROACH:

Step 1 - Parse grading notes into distinct concepts:

- Separate by commas, semicolons, or line breaks
- Each item is a concept that must be verified
- Example: "*Gross Margin* >40%, CAC, LTV:CAC >3:1" = 3 concepts

Step 2 - For each concept, check if it's addressed:

- Accept semantic equivalents (e.g., "customer acquisition cost" = "CAC")
- Accept implicit coverage when it's clear (e.g., "revenue forecasting" covers "historical vs forecasted rev")
- Be flexible on exact numbers (e.g., "around 40%" acceptable for ">40%")

Step 3 - Count missing concepts:

- Missing 0 concepts = PASS
- Missing 1+ concepts = FAIL (even one genuinely missing concept should fail)
- Exception: If a long list (10+ items) has 1 very minor detail missing but all major points covered, use judgment

CRITICAL RULES:

1. Do NOT require exact wording - "market demand" = "mkt demand" = "demand analysis"

2. Markers (* or !) mean important, not mandatory exact phrases:
   - "*traction evidence*" can be satisfied by discussing metrics, growth, or validation
   - "!unbiased assumptions" can be satisfied by discussing assumption methodology

3. Numbers should be mentioned but accept approximations:
   - "$47B to $10B" can be "$47 billion dropped to around $10 billion"
   - "LTV:CAC >3:1" can be "LTV to CAC ratio of at least 3 to 1" or "3x or higher"

4. FAIL only when concepts are genuinely absent:
   - If notes mention "liquidation prefs, anti-dilution, board seats" but response only has board seats → FAIL
   - If notes mention "scalability, tech debt, IP" but response never discusses technical risks → FAIL
   - If notes mention "GDPR compliance" and response never mentions GDPR or EU regulations → FAIL

5. PASS when ALL concepts present:
   - All concepts covered, even with different wording → PASS
   - Concepts addressed implicitly when clearly implied → PASS
   - Minor phrasing differences → PASS
   - One or more concepts genuinely absent → FAIL

Response: {response}

Grading Notes: {grading_notes}

Are ALL distinct concepts from the grading notes covered in the response (accepting semantic equivalents and implicit coverage)?""",
    allowed_values=["pass", "fail"],
)
```

!!! tip "Optimizing prompts using LLMs"
    You can use LLMs to optimize prompts after you identify error patterns clearly. You can use LLMs to identify errors too, but make sure to review them so they're aligned with the ground truth labels. You can also use coding agents like Cursor, Claude Code, or frameworks like [DSPy](https://github.com/stanfordnlp/dspy) to systematically optimize judge prompts.

## Re-run evaluation with improved prompt

Run the evaluation again with the enhanced v2 prompt (same setup as baseline, just swap the metric):

```python
# Use the same dataset and LLM setup from the baseline evaluation above
results = await judge_experiment.arun(
    dataset,
    name="judge_accuracy_v2_gpt-4o-mini",
    accuracy_metric=accuracy_metric_v2,  # ← Using improved v2 prompt
    llm=llm,
)

passed = sum(1 for r in results if r["alignment"] == "pass")
total = len(results)
print(f"✅ V2 alignment: {passed}/{total} passed ({passed/total:.1%})")
```

??? "📋 Output (improved v2)"

    ```text
    2025-10-08 23:42:11,650 - Loaded dataset with 160 samples
    2025-10-08 23:42:11,650 - Initializing LLM client with model: gpt-4o-mini
    2025-10-08 23:42:12,730 - Running v2 evaluation with improved prompt...
    Running experiment: 100%|██████████| 160/160 [04:39<00:00,  1.75s/it]
    2025-10-08 23:46:52,740 - ✅ V2 alignment: 139/160 passed (86.9%)
    ```

**Significant improvement!** The alignment increased from 75.6% to 86.9%.

If you need to iterate further:

- Analyze remaining errors to identify patterns (are they false positives or false negatives?)
- Annotate your reasoning along with label - this will help while improving the LLM Judge, you can add these as few shot examples as well.
- **Use smarter models** - More capable models like GPT-5 or Claude 4.5 Sonnet generally perform better as judges
- **Leverage AI assistants** - This guide was created using Cursor AI agents to analyze failures and iterate on prompts. You can use AI coding agents (Cursor, Claude, etc.) or frameworks like [DSPy](https://github.com/stanfordnlp/dspy) to systematically optimize judge prompts
- Stop when alignment plateaus across 2-3 iterations or meets your business threshold

## What you've accomplished

You've built a systematic evaluation pipeline using Ragas that:

- Measures judge alignment against expert judgments with clear metrics
- Identifies failure patterns through structured error analysis
- Tracks improvement across evaluation runs with reproducible experiments

This aligned judge becomes your foundation for reliable AI evaluation. With a judge you can trust, you can now confidently evaluate your RAG pipeline, agent workflows, or any LLM application—knowing that improvements in metrics translate to real improvements in quality.
