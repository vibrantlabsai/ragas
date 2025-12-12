# Overview of Metrics

## Why Metrics Matter

You can't improve what you don't measure. Metrics are the feedback loop that makes iteration possible.

In AI systems, progress depends on running many experiments—each a hypothesis about how to improve performance. But without a clear, reliable metric, you can't tell the difference between a successful experiment (a positive delta between the new score and the old one) and a failed one.

Metrics give you a compass. They let you quantify improvement, detect regressions, and align optimization efforts with user impact and business value.

A metric is a quantitative measure used to evaluate the performance of a AI application. Metrics help in assessing how well the application and individual components that makes up application is performing relative to the given test data. They provide a numerical basis for comparison, optimization, and decision-making throughout the application development and deployment process. Metrics are crucial for:

1. **Component Selection**: Metrics can be used to compare different components of the AI application like LLM, Retriever, Agent configuration, etc with your own data and select the best one from different options.
2. **Error Diagnosis and Debugging**: Metrics help identify which part of the application is causing errors or suboptimal performance, making it easier to debug and refine.
3. **Continuous Monitoring and Maintenance**: Metrics enable the tracking of an AI application's performance over time, helping to detect and respond to issues such as data drift, model degradation, or changing user requirements.

## Types of Metrics in AI Applications

### 1. End-to-End Metrics

End-to-end metrics evaluate the overall system performance from the user's perspective, treating the AI application as a black box. These metrics quantify key outcomes users care deeply about, based solely on the system's final outputs.

Examples:

- Answer correctness: Measures if the provided answers from a Retrieval-Augmented Generation (RAG) system are accurate.
- Citation accuracy: Evaluates whether the references cited by the RAG system are correctly identified and relevant.

Optimizing end-to-end metrics ensures tangible improvements aligned directly with user expectations.

### 2. Component-Level Metrics

Component-level metrics assess the individual parts of an AI system independently. These metrics are immediately actionable and facilitate targeted improvements but do not necessarily correlate directly with end-user satisfaction.

Example:

- Retrieval accuracy: Measures how effectively a RAG system retrieves relevant information. A low retrieval accuracy (e.g., 50%) signals that improving this component can enhance overall system performance. However, improving a component alone doesn't guarantee better end-to-end outcomes.

### 3. Business Metrics

Business metrics align AI system performance with organizational objectives and quantify tangible business outcomes. These metrics are typically lagging indicators, calculated after a deployment period (days/weeks/months).

Example:

- Ticket deflection rate: Measures the percentage reduction of support tickets due to the deployment of an AI assistant.

## Types of Metrics in Ragas

<figure markdown="span">
  ![Component-wise Evaluation](../../../_static/imgs/metrics_mindmap.png){width="600"}
  <figcaption>Metrics Mind map</figcaption>
</figure>

**Metrics can be classified into two categories based on the mechanism used underneath the hood**:

&nbsp;&nbsp;&nbsp;&nbsp; **LLM-based metrics**: These metrics use LLM underneath to do the evaluation. There might be one or more LLM calls that are performed to arrive at the score or result. These metrics can be somewhat non-deterministic as the LLM might not always return the same result for the same input. On the other hand, these metrics has shown to be more accurate and closer to human evaluation.

All LLM based metrics in ragas are inherited from `MetricWithLLM` class. These metrics expects a LLM object to be set before scoring.

```python
from ragas.metrics import FactualCorrectness
scorer = FactualCorrectness(llm=evaluation_llm)
```

Each LLM based metrics also will have prompts associated with it written using [Prompt Object](./../../components/prompt.md). You can customize these prompts to suit your domain and use-case. Learn more in the [Modifying Prompts in Metrics](../../../howtos/customizations/metrics/modifying-prompts-metrics.md) guide.


&nbsp;&nbsp;&nbsp;&nbsp; **Non-LLM-based metrics**: These metrics do not use LLM underneath to do the evaluation. These metrics are deterministic and can be used to evaluate the performance of the AI application without using LLM. These metrics rely on traditional methods to evaluate the performance of the AI application, such as string similarity, BLEU score, etc. Due to the same, these metrics are known to have a lower correlation with human evaluation.

All Non-LLM-based metrics in ragas are inherited from `Metric` class.

**Metrics can be broadly classified into two categories based on the type of data they evaluate**:

&nbsp;&nbsp;&nbsp;&nbsp; **Single turn metrics**: These metrics evaluate the performance of the AI application based on a single turn of interaction between the user and the AI. All metrics in ragas that supports single turn evaluation are inherited from [SingleTurnMetric][ragas.metrics.base.SingleTurnMetric] class and scored using `single_turn_ascore` method. It also expects a [Single Turn Sample][ragas.dataset_schema.SingleTurnSample] object as input.

```python
from ragas.metrics import FactualCorrectness

scorer = FactualCorrectness()
await scorer.single_turn_ascore(sample)
```

&nbsp;&nbsp;&nbsp;&nbsp; **Multi-turn metrics**: These metrics evaluate the performance of the AI application based on multiple turns of interaction between the user and the AI. All metrics in ragas that supports multi turn evaluation are inherited from [MultiTurnMetric][ragas.metrics.base.MultiTurnMetric] class and scored using `multi_turn_ascore` method. It also expects a [Multi Turn Sample][ragas.dataset_schema.MultiTurnSample] object as input.

```python
from ragas.metrics import AgentGoalAccuracy
from ragas import MultiTurnSample

scorer = AgentGoalAccuracy()
await scorer.multi_turn_ascore(sample)
```

### Output Types

In Ragas, we categorize metrics based on the type of output they produce. This classification helps clarify how each metric behaves and how its results can be interpreted or aggregated. The three types are:

#### 1. Discrete Metrics

These return a single value from a predefined list of categorical classes. There is no implicit ordering among the classes. Common use cases include classifying outputs into categories such as pass/fail or good/okay/bad. Discrete metrics accept custom prompts directly, making them ideal for quick custom evaluations.

Example:
```python
from ragas.metrics import discrete_metric

@discrete_metric(name="response_quality", allowed_values=["pass", "fail"])
def my_metric(predicted: str, expected: str) -> str:
    return "pass" if predicted.lower() == expected.lower() else "fail"
```

For modifying prompts in existing collection metrics (like Faithfulness, FactualCorrectness), see [Modifying prompts in metrics](../../../howtos/customizations/metrics/modifying-prompts-metrics.md).

#### 2. Numeric Metrics

These return an integer or float value within a specified range. Numeric metrics support aggregation functions such as mean, sum, or mode, making them useful for statistical analysis.

```python
from ragas.metrics import numeric_metric

@numeric_metric(name="response_accuracy", allowed_values=(0, 1))
def my_metric(predicted: float, expected: float) -> float:
    return abs(predicted - expected) / max(expected, 1e-5)

my_metric.score(predicted=0.8, expected=1.0)  # Returns a float value
```

#### 3. Ranking Metrics

These evaluate multiple outputs at once and return a ranked list based on a defined criterion. They are useful when the goal is to compare multiple outputs from the same pipeline relative to one another.

```python
from ragas.metrics import ranking_metric
@ranking_metric(name="response_ranking", allowed_values=[0,1])
def my_metric(responses: list) -> list:
    response_lengths = [len(response) for response in responses]
    sorted_indices = sorted(range(len(response_lengths)), key=lambda i: response_lengths[i])
    return sorted_indices

my_metric.score(responses=["short", "a bit longer", "the longest response"])  # Returns a ranked list of indices
```

## Metric Design Principles

Designing effective metrics for AI applications requires following to a set of core principles to ensure their reliability, interpretability, and relevance. Here are five key principles we follow in ragas when designing metrics:

**1. Single-Aspect Focus**
A single metric should target only one specific aspect of the AI application's performance. This ensures that the metric is both interpretable and actionable, providing clear insights into what is being measured.

**2. Intuitive and Interpretable**
Metrics should be designed to be easy to understand and interpret. Clear and intuitive metrics make it simpler to communicate results and draw meaningful conclusions.

**3. Effective Prompt Flows**
When developing metrics using large language models (LLMs), use intelligent prompt flows that align closely with human evaluation. Decomposing complex tasks into smaller sub-tasks with specific prompts can improve the accuracy and relevance of the metric.

**4. Robustness**
Ensure that LLM-based metrics include sufficient few-shot examples that reflect the desired outcomes. This enhances the robustness of the metric by providing context and guidance for the LLM to follow.

**5.Consistent Scoring Ranges**
It is crucial to normalize metric score values or ensure they fall within a specific range, such as 0 to 1. This facilitates comparison between different metrics and helps maintain consistency and interpretability across the evaluation framework.

These principles serve as a foundation for creating metrics that are not only effective but also practical and meaningful in evaluating AI applications.

## Choosing the Right Metrics for Your Application

### 1. Prioritize End-to-End Metrics

Focus first on metrics reflecting overall user satisfaction. While many aspects influence user satisfaction—such as factual correctness, response tone, and explanation depth—concentrate initially on the few dimensions delivering maximum user value (e.g., answer and citation accuracy in a RAG-based assistant).

### 2. Ensure Interpretability

Design metrics clear enough for the entire team to interpret and reason about. For example:

- Execution accuracy in a text-to-SQL system: Does the SQL query generated return precisely the same dataset as the ground truth query crafted by domain experts?

### 3. Emphasize Objective Over Subjective Metrics

Prioritize metrics with objective criteria, minimizing subjective judgment. Assess objectivity by independently labeling samples across team members and measuring agreement levels. A high inter-rater agreement (≥80%) indicates greater objectivity.

### 4. Few Strong Signals over Many Weak Signals

Avoid a proliferation of metrics that provide weak signals and impede clear decision-making. Instead, select fewer metrics offering strong, reliable signals. For instance:

- In a conversational AI, using a single metric such as goal accuracy (whether the user's objective for interacting with the AI was met) provides strong proxy for the performance of the system than multiple weak proxies like coherence or helpfulness.