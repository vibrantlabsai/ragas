# Rubric-Based Evaluation

Rubric-based evaluation metrics allow you to evaluate LLM responses using custom scoring criteria. Ragas provides two types of rubric metrics:

1. **DomainSpecificRubrics**: Uses the same rubric for all samples in a dataset (set at initialization)
2. **InstanceSpecificRubrics**: Each sample can have its own unique rubric (passed per evaluation)

The rubric consists of descriptions for each score, typically ranging from 1 to 5. The response is evaluated and scored using an LLM based on the descriptions specified in the rubric.

## Domain-Specific Rubrics

Use `DomainSpecificRubrics` when you want to apply the same evaluation criteria across all samples. This is useful for domain-wide evaluations where the scoring criteria remain constant.

### Example

```python
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import DomainSpecificRubrics

# Setup
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Reference-free evaluation (default)
metric = DomainSpecificRubrics(llm=llm)
result = await metric.ascore(
    user_input="What's the longest river in the world?",
    response="The longest river in the world is the Nile, stretching approximately 6,650 kilometers through northeastern Africa.",
)
print(f"Score: {result.value}, Feedback: {result.reason}")

# Reference-based evaluation
metric_with_ref = DomainSpecificRubrics(llm=llm, with_reference=True)
result = await metric_with_ref.ascore(
    user_input="What's the longest river in the world?",
    response="The longest river in the world is the Nile.",
    reference="The Nile is a major north-flowing river in northeastern Africa.",
)
```

### Custom Rubrics

You can define your own rubrics to customize the scoring criteria:

```python
from ragas.metrics.collections import DomainSpecificRubrics

my_custom_rubrics = {
    "score1_description": "Answer and ground truth are completely different",
    "score2_description": "Answer and ground truth are somewhat different",
    "score3_description": "Answer and ground truth are somewhat similar",
    "score4_description": "Answer and ground truth are similar",
    "score5_description": "Answer and ground truth are exactly the same",
}

metric = DomainSpecificRubrics(llm=llm, rubrics=my_custom_rubrics, with_reference=True)
```

### With Retrieved Contexts

The metric also supports evaluation with retrieved contexts:

```python
result = await metric.ascore(
    user_input="What's the longest river in the world?",
    response="Based on the context, the Nile is the longest river.",
    retrieved_contexts=[
        "Scientists debate whether the Amazon or the Nile is the longest river.",
        "The Nile River was central to Ancient Egyptians' wealth and power.",
    ],
)
```

### Convenience Classes

For clearer intent, use the convenience classes:

```python
from ragas.metrics.collections import (
    RubricsScoreWithoutReference,
    RubricsScoreWithReference,
)

# Reference-free
metric_no_ref = RubricsScoreWithoutReference(llm=llm)

# Reference-based
metric_with_ref = RubricsScoreWithReference(llm=llm)
```

## Default Rubrics

### Reference-Free Rubrics (Default)

| Score | Description |
|-------|-------------|
| 1 | The response is entirely incorrect and fails to address any aspect of the user input. |
| 2 | The response contains partial accuracy but includes major errors or significant omissions. |
| 3 | The response is mostly accurate but lacks clarity, thoroughness, or minor details. |
| 4 | The response is accurate and clear, with only minor omissions or slight inaccuracies. |
| 5 | The response is completely accurate, clear, and thoroughly addresses the user input. |

### Reference-Based Rubrics

| Score | Description |
|-------|-------------|
| 1 | The response is entirely incorrect, irrelevant, or does not align with the reference. |
| 2 | The response partially matches the reference but contains major errors or omissions. |
| 3 | The response aligns with the reference overall but lacks sufficient detail or clarity. |
| 4 | The response is mostly accurate, aligns closely with the reference with minor issues. |
| 5 | The response is fully accurate, completely aligns with the reference, clear and detailed. |

---

## Instance-Specific Rubrics

Use `InstanceSpecificRubrics` when different samples require different evaluation criteria. This is useful when:
- Different questions require different evaluation standards
- You want to customize scoring based on specific task requirements
- Evaluation criteria vary across your dataset

### Example

```python
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import InstanceSpecificRubrics

# Setup
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

metric = InstanceSpecificRubrics(llm=llm)

# Each sample can have its own rubrics
email_rubrics = {
    "score1_description": "The email is unprofessional or inappropriate",
    "score2_description": "The email lacks proper formatting or tone",
    "score3_description": "The email is acceptable but could be improved",
    "score4_description": "The email is professional with minor issues",
    "score5_description": "The email is highly professional and well-written",
}

result = await metric.ascore(
    user_input="Write a professional email declining a meeting invitation",
    response="Dear John, Thank you for the invitation...",
    rubrics=email_rubrics,
)
print(f"Score: {result.value}, Feedback: {result.reason}")

# Different rubrics for a different type of task
code_rubrics = {
    "score1_description": "The code doesn't work or has critical bugs",
    "score2_description": "The code has significant issues or is poorly structured",
    "score3_description": "The code works but lacks optimization or best practices",
    "score4_description": "The code is good with minor improvements possible",
    "score5_description": "The code is excellent, efficient, and follows best practices",
}

result = await metric.ascore(
    user_input="Write a function to sort a list",
    response="def sort_list(arr): return sorted(arr)",
    rubrics=code_rubrics,
)
```

### With Reference and Contexts

```python
result = await metric.ascore(
    user_input="Explain the water cycle",
    response="The water cycle involves evaporation, condensation, and precipitation.",
    reference="The water cycle describes how water evaporates from surfaces, rises into the atmosphere, condenses into clouds, and falls as precipitation.",
    retrieved_contexts=["Water cycle information from encyclopedia..."],
    rubrics={
        "score1_description": "Explanation is completely wrong",
        "score2_description": "Explanation has major inaccuracies",
        "score3_description": "Explanation is partially correct",
        "score4_description": "Explanation is mostly correct",
        "score5_description": "Explanation is comprehensive and accurate",
    },
)
```

---

## Legacy API

!!! warning "Deprecated"
    The legacy API below is deprecated. Please use `ragas.metrics.collections.DomainSpecificRubrics` or `ragas.metrics.collections.InstanceSpecificRubrics` instead.

```python
from ragas import evaluate
from datasets import Dataset

from ragas.metrics import rubrics_score_without_reference, rubrics_score_with_reference

rows = {
    "question": [
        "What's the longest river in the world?",
    ],
    "ground_truth": [
        "The Nile is a major north-flowing river in northeastern Africa.",
    ],
    "answer": [
        "The longest river in the world is the Nile, stretching approximately 6,650 kilometers (4,130 miles) through northeastern Africa.",
    ],
    "contexts": [
        [
            "Scientists debate whether the Amazon or the Nile is the longest river in the world.",
            "The Nile River was central to the Ancient Egyptians' rise to wealth and power.",
        ],
    ]
}

dataset = Dataset.from_dict(rows)

result = evaluate(
    dataset,
    metrics=[
        rubrics_score_without_reference,
        rubrics_score_with_reference
    ],
)
```

Custom rubrics with legacy API:

```python
from ragas.metrics._domain_specific_rubrics import RubricsScore

my_custom_rubrics = {
    "score1_description": "answer and ground truth are completely different",
    "score2_description": "answer and ground truth are somewhat different",
    "score3_description": "answer and ground truth are somewhat similar",
    "score4_description": "answer and ground truth are similar",
    "score5_description": "answer and ground truth are exactly the same",
}

rubrics_score = RubricsScore(rubrics=my_custom_rubrics)
```


