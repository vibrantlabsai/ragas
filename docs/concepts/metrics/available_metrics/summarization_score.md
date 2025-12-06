# Tasks Metrics

## Summarization Score

The **Summarization Score** metric measures how well a summary (`response`) captures the important information from the `reference_contexts`. The intuition behind this metric is that a good summary should contain all the important information present in the context.

We first extract a set of important keyphrases from the context. These keyphrases are then used to generate a set of questions. The answers to these questions are always `yes(1)` for the context. We then ask these questions to the summary and calculate the summarization score as the ratio of correctly answered questions to the total number of questions. 

We compute the question-answer score using the answers, which is a list of `1`s and `0`s. The question-answer score is then calculated as the ratio of correctly answered questions(answer = `1`) to the total number of questions.

$$
\text{QA score} = \frac{|\text{correctly answered questions}|}{|\text{total questions}|}
$$

We also introduce an option to penalize larger summaries by proving a conciseness score. If this option is enabled, the final score is calculated as the weighted average of the summarization score and the conciseness score. This conciseness scores ensures that summaries that are just copies of the text do not get a high score, because they will obviously answer all questions correctly.

$$
\text{conciseness score} = 1 - \frac{\min(\text{length of summary}, \text{length of context})}{\text{length of context} + \text{1e-10}}
$$

We also provide a coefficient `coeff`(default value 0.5) to control the weightage of the scores. 

The final summarization score is then calculated as:

$$
\text{Summarization Score} = \text{QA score}*\text{(1-coeff)} + \\
\text{conciseness score}*\text{coeff}
$$

### Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import SummaryScore

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = SummaryScore(llm=llm)

# Evaluate
result = await scorer.ascore(
    reference_contexts=[
        "A company is launching a new product, a smartphone app designed to help users track their fitness goals. The app allows users to set daily exercise targets, log their meals, and track their water intake. It also provides personalized workout recommendations and sends motivational reminders throughout the day."
    ],
    response="A company is launching a fitness tracking app that helps users set exercise goals, log meals, and track water intake, with personalized workout suggestions and motivational reminders."
)
print(f"Summary Score: {result.value}")
```

Output:

```
Summary Score: 0.6423387096775146
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        reference_contexts=[...],
        response="..."
    )
    ```


## Legacy Metrics API

The following examples use the legacy metrics API pattern. For new projects, we recommend using the collections-based API shown above.

!!! warning "Deprecation Timeline"
    This API will be deprecated in version 0.4 and removed in version 1.0. Please migrate to the collections-based API shown above.

### Example with SingleTurnSample

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import SummarizationScore


sample = SingleTurnSample(
    response="A company is launching a fitness tracking app that helps users set exercise goals, log meals, and track water intake, with personalized workout suggestions and motivational reminders.",
    reference_contexts=[
        "A company is launching a new product, a smartphone app designed to help users track their fitness goals. The app allows users to set daily exercise targets, log their meals, and track their water intake. It also provides personalized workout recommendations and sends motivational reminders throughout the day."
    ]
)

scorer = SummarizationScore(llm=evaluator_llm)
await scorer.single_turn_ascore(sample)
```

Output:

```
0.6423387096775146
```