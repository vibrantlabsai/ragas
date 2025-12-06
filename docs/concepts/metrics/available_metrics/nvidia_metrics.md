# Nvidia Metrics

## Answer Accuracy

**Answer Accuracy** measures the agreement between a model’s response and a reference ground truth for a given question. This is done via two distinct "LLM-as-a-Judge" prompts that each return a rating (0, 2, or 4). The metric converts these ratings into a [0,1] scale and then takes the average of the two scores from the judges. Higher scores indicate that the model’s answer closely matches the reference.

- **0** → The **response** is inaccurate or does not address the same question as the **reference**.
- **2** → The **response** partially align with the **reference**.
- **4** → The **response** exactly aligns with the **reference**.


### Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerAccuracy

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = AnswerAccuracy(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="When was Einstein born?",
    response="Albert Einstein was born in 1879.",
    reference="Albert Einstein was born in 1879."
)
print(f"Answer Accuracy Score: {result.value}")
```

Output:

```
Answer Accuracy Score: 1.0
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        user_input="When was Einstein born?",
        response="Albert Einstein was born in 1879.",
        reference="Albert Einstein was born in 1879."
    )
    ```

### How It’s Calculated

**Step 1:** The LLM generates ratings using two distinct templates to ensure robustness:

- **Template 1:** The LLM compares the **response** with the **reference** and rates it on a scale of **0, 2, or 4**.
- **Template 2:** The LLM evaluates the same question again, but this time the roles of the **response** and the **reference** are swapped.

This dual-perspective approach guarantees a fair assessment of the answer's accuracy.

**Step 2:** If both ratings are valid, the final score is average of score1 and score2; otherwise, it takes the valid one.

**Example Calculation:**

- **User Input:** "When was Einstein born?"
- **Response:** "Albert Einstein was born in 1879."
- **Reference:** "Albert Einstein was born in 1879."

Assuming both templates return a rating of **4** (indicating an exact match), the conversion is as follows:

- A rating of **4** corresponds to **1** on the [0,1] scale.
- Averaging the two scores: (1 + 1) / 2 = **1**.

Thus, the final **Answer Accuracy** score is **1**.

### Similar Ragas Metrics

1. [Answer Correctness](answer_correctness.md): This metric gauges the accuracy of the generated answer compared to the ground truth by considering both semantic and factual similarity.

2. [Rubric Score](general_purpose.md#rubrics-based-criteria-scoring): The Rubric-Based Criteria Scoring Metric allows evaluations based on user-defined rubrics, where each rubric outlines specific scoring criteria. The LLM assesses responses according to these customized descriptions, ensuring a consistent and objective evaluation process.

### Comparison of Metrics

#### Answer Correctness vs. Answer Accuracy

- **LLM Calls:** Answer Correctness requires three LLM calls (two for decomposing the response and reference into standalone statements and one for classifying them), while Answer Accuracy uses two independent LLM judgments.
- **Token Usage:** Answer Correctness consumes lot more tokens due to its detailed breakdown and classification process.
- **Explainability:** Answer Correctness offers high explainability by providing detailed insights into factual correctness and semantic similarity, whereas Answer Accuracy provides a straightforward raw score.
- **Robust Evaluation:** Answer Accuracy ensures consistency through dual LLM evaluations, while Answer Correctness offers a holistic view by deeply assessing the quality of the response.

#### Answer Accuracy vs. Rubric Score

- **LLM Calls**: Answer Accuracy makes two calls (one per LLM judge), while Rubric Score requires only one.
- **Token Usage**: Answer Accuracy is minimal since it outputs just a score, whereas Rubric Score generates reasoning, increasing token consumption.
- **Explainability**: Answer Accuracy provides a raw score without justification, while Rubric Score offers reasoning with verdict.
- **Efficiency**: Answer Accuracy is lightweight and works very well with smaller models.

### Legacy Metrics API

The following examples use the legacy metrics API pattern. For new projects, we recommend using the collections-based API shown above.

!!! warning "Deprecation Timeline"
    This API will be deprecated in version 0.4 and removed in version 1.0. Please migrate to the collections-based API shown above.

#### Example with SingleTurnSample

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AnswerAccuracy

sample = SingleTurnSample(
    user_input="When was Einstein born?",
    response="Albert Einstein was born in 1879.",
    reference="Albert Einstein was born in 1879."
)

scorer = AnswerAccuracy(llm=evaluator_llm) # evaluator_llm wrapped with ragas LLM Wrapper
score = await scorer.single_turn_ascore(sample)
print(score)
```

Output:

```
1.0
```

## Context Relevance

**Context Relevance** evaluates whether the **retrieved_contexts** (chunks or passages) are pertinent to the **user_input**. This is done via two independent "LLM-as-a-Judge" prompt calls that each rate the relevance on a scale of **0, 1, or 2**. The ratings are then converted to a [0,1] scale and averaged to produce the final score. Higher scores indicate that the contexts are more closely aligned with the user's query.

- **0** → The retrieved contexts are not relevant to the user's query at all.
- **1** → The contexts are partially relevant.
- **2** → The contexts are completely relevant.

### Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRelevance

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = ContextRelevance(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="When and Where Albert Einstein was born?",
    retrieved_contexts=[
        "Albert Einstein was born March 14, 1879.",
        "Albert Einstein was born at Ulm, in Württemberg, Germany.",
    ]
)
print(f"Context Relevance Score: {result.value}")
```

Output:

```
Context Relevance Score: 1.0
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        user_input="When and Where Albert Einstein was born?",
        retrieved_contexts=[...]
    )
    ```

### Implementation Note

**Difference from Original Paper:** The original Ragas paper defines Context Relevance using sentence-level extraction (CR = number of relevant sentences / total sentences), but the current implementation uses a more robust discrete judgment approach. Each LLM is asked to rate overall context relevance on a 0-2 scale, which is more efficient and less prone to sentence boundary errors. This was an intentional design decision to improve reliability and reduce computational overhead while maintaining the core evaluation objective.

### How It's Calculated

**Step 1:** The LLM is prompted with two distinct templates (template_relevance1 and template_relevance2) to evaluate the relevance of the retrieved contexts concerning the user's query. Each prompt returns a relevance rating of **0**, **1**, or **2**. Using two independent evaluations provides robustness and helps mitigate individual LLM biases.

**Step 2:** Each rating is normalized to a [0,1] scale by dividing by 2. If both ratings are valid, the final score is the average of these normalized values; if only one is valid, that score is used.

**Example Calculation:**

- **User Input:** "When and Where Albert Einstein was born?"
- **Retrieved Contexts:**
  - "Albert Einstein was born March 14, 1879."
  - "Albert Einstein was born at Ulm, in Württemberg, Germany."

In this example, the two retrieved contexts together fully address the user's query by providing both the birthdate and location of Albert Einstein. Consequently, both prompts would rate the combined contexts as **2** (fully relevant). Normalizing each score yields **1.0** (2/2), and averaging the two results maintains the final Context Relevance score at **1**.

### Similar Ragas Metrics

1. [Context Precision](context_precision.md): It measures the proportion of retrieved contexts that are relevant to answering a user's query. It is computed as the mean precision@k across all retrieved chunks, indicating how accurately the retrieval system ranks relevant information.

2. [Context Recall](context_recall.md): It quantifies the extent to which the relevant information is successfully retrieved. It is calculated as the ratio of the number of relevant claims (or contexts) found in the retrieved results to the total number of relevant claims in the reference, ensuring that important information is not missed.

3. [Rubric Score](general_purpose.md#rubrics-based-criteria-scoring): The Rubric-Based Criteria Scoring Metric evaluates responses based on user-defined rubrics with customizable scoring criteria, ensuring consistent and objective assessments. The scoring scale is flexible to suit user needs.

#### Context Precision and Context Recall vs. Context Relevance

- **LLM Calls:** Context Precision and Context Recall each require one LLM call each, one verifies context usefulness to get reference (verdict "1" or "0") and one classifies each answer sentence as attributable (binary 'Yes' (1) or 'No' (0)) while Context Relevance uses two LLM calls for increased robustness.
- **Token Usage:** Context Precision and Context Recall consume lot more tokens, whereas Context Relevance is more token-efficient.
- **Explainability:** Context Precision and Context Recall offer high explainability with detailed reasoning, while Context Relevance provides a raw score without explanations.
- **Robust Evaluation:** Context Relevance delivers a more robust evaluation through dual LLM judgments compared to the single-call approach of Context Precision and Context Recall.

### Legacy Metrics API

The following examples use the legacy metrics API pattern. For new projects, we recommend using the collections-based API shown above.

!!! warning "Deprecation Timeline"
    This API will be deprecated in version 0.4 and removed in version 1.0. Please migrate to the collections-based API shown above.

#### Example with SingleTurnSample

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ContextRelevance

sample = SingleTurnSample(
    user_input="When and Where Albert Einstein was born?",
    retrieved_contexts=[
        "Albert Einstein was born March 14, 1879.",
        "Albert Einstein was born at Ulm, in Württemberg, Germany.",
    ]
)

scorer = ContextRelevance(llm=evaluator_llm)
score = await scorer.single_turn_ascore(sample)
print(score)
```

Output:

```
1.0
```

## Response Groundedness

**Response Groundedness** measures how well a response is supported or "grounded" by the retrieved contexts. It assesses whether each claim in the response can be found, either wholly or partially, in the provided contexts.

- **0** → The response is **not** grounded in the context at all.
- **1** → The response is partially grounded.
- **2** → The response is fully grounded (every statement can be found or inferred from the retrieved context).

### Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ResponseGroundedness

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = ResponseGroundedness(llm=llm)

# Evaluate
result = await scorer.ascore(
    response="Albert Einstein was born in 1879.",
    retrieved_contexts=[
        "Albert Einstein was born March 14, 1879.",
        "Albert Einstein was born at Ulm, in Württemberg, Germany.",
    ]
)
print(f"Response Groundedness Score: {result.value}")
```

Output:

```
Response Groundedness Score: 1.0
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        response="Albert Einstein was born in 1879.",
        retrieved_contexts=[...]
    )
    ```

### How It’s Calculated

**Step 1:** The LLM is prompted with two distinct templates to evaluate the grounding of the response with respect to the retrieved contexts. Each prompt returns a grounding rating of **0**, **1**, or **2**.

**Step 2:** Each rating is normalized to a [0,1] scale by dividing by 2 (i.e., 0 becomes 0.0, 1 becomes 0.5, and 2 becomes 1.0). If both ratings are valid, the final score is computed as the average of these normalized values; if only one is valid, that score is used.

**Example Calculation:**

- **Response:** "Albert Einstein was born in 1879."
- **Retrieved Contexts:**
  - "Albert Einstein was born March 14, 1879."
  - "Albert Einstein was born at Ulm, in Württemberg, Germany."

In this example, the retrieved contexts provide both the birthdate and location of Albert Einstein. Since the response's claim is supported by the context (even though the date is partially provided), both prompts would likely rate the grounding as **2** (fully grounded). Normalizing a score of 2 gives **1.0** (2/2), and averaging the two normalized ratings maintains the final Response Groundedness score at **1**.

### Similar Ragas Metrics

1. [Faithfulness](faithfulness.md): This metric measures how factually consistent a response is with the retrieved context, ensuring that every claim in the response is supported by the provided information. The Faithfulness score ranges from 0 to 1, with higher scores indicating better consistency.

2. [Rubric Score](general_purpose.md#rubrics-based-criteria-scoring): This is a general-purpose metric that evaluates responses based on user-defined criteria and can be adapted to assess Answer Accuracy, Context Relevance or Response Groundedness by aligning the rubric with the requirements.

### Comparison of Metrics

#### Faithfulness vs. Response Groundedness

- **LLM Calls:** Faithfulness requires two calls for detailed claim breakdown and verdict, while Response Groundedness uses two independent LLM judgments.
- **Token Usage:** Faithfulness consumes more tokens, whereas Response Groundedness is more token-efficient.
- **Explainability:** Faithfulness provides transparent, reasoning for each claim, while Response Groundedness provides a raw score.
- **Robust Evaluation:** Faithfulness incorporates user input for a comprehensive assessment, whereas Response Groundedness ensures consistency through dual LLM evaluations.

### Legacy Metrics API

The following examples use the legacy metrics API pattern. For new projects, we recommend using the collections-based API shown above.

!!! warning "Deprecation Timeline"
    This API will be deprecated in version 0.4 and removed in version 1.0. Please migrate to the collections-based API shown above.

#### Example with SingleTurnSample

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ResponseGroundedness

sample = SingleTurnSample(
    response="Albert Einstein was born in 1879.",
    retrieved_contexts=[
        "Albert Einstein was born March 14, 1879.",
        "Albert Einstein was born at Ulm, in Württemberg, Germany.",
    ]
)

scorer = ResponseGroundedness(llm=evaluator_llm)
score = await scorer.single_turn_ascore(sample)
print(score)
```

Output:

```
1.0
```
