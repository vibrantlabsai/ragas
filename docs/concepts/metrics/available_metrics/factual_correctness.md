## Factual Correctness

`FactualCorrectness` is a metric that compares and evaluates the factual accuracy of the generated `response` with the `reference`. This metric is used to determine the extent to which the generated response aligns with the reference. The factual correctness score ranges from 0 to 1, with higher values indicating better performance. To measure the alignment between the response and the reference, the metric uses the LLM to first break down the response and reference into claims and then uses natural language inference to determine the factual overlap between the response and the reference. Factual overlap is quantified using precision, recall, and F1 score, which can be controlled using the `mode` parameter.

### Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import FactualCorrectness

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = FactualCorrectness(llm=llm)

# Evaluate
result = await scorer.ascore(
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris. It has a height of 1000ft."
)
print(f"Factual Correctness Score: {result.value}")
```

Output:

```
Factual Correctness Score: 0.67
```

By default, the mode is set to `f1`. You can change the mode to `precision` or `recall` by setting the `mode` parameter:

```python
# Precision mode - measures what fraction of response claims are supported by reference
scorer = FactualCorrectness(llm=llm, mode="precision")
result = await scorer.ascore(
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris. It has a height of 1000ft."
)
print(f"Precision Score: {result.value}")
```

Output:

```
Precision Score: 1.0
```

You can also configure the claim decomposition granularity using `atomicity` and `coverage` parameters:

```python
# High granularity - more detailed claim decomposition
scorer = FactualCorrectness(
    llm=llm,
    mode="f1",
    atomicity="high",  # More atomic claims
    coverage="high"    # Comprehensive coverage
)
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        response="The Eiffel Tower is located in Paris.",
        reference="The Eiffel Tower is located in Paris. It has a height of 1000ft."
    )
    ```

### How It's Calculated

The formula for calculating True Positive (TP), False Positive (FP), and False Negative (FN) is as follows:

$$
\text{True Positive (TP)} = \text{Number of claims in response that are present in reference}
$$

$$
\text{False Positive (FP)} = \text{Number of claims in response that are not present in reference}
$$

$$
\text{False Negative (FN)} = \text{Number of claims in reference that are not present in response}
$$

The formula for calculating precision, recall, and F1 score is as follows:

$$
\text{Precision} = {TP \over (TP + FP)}
$$

$$
\text{Recall} = {TP \over (TP + FN)}
$$

$$
\text{F1 Score} = {2 \times \text{Precision} \times \text{Recall} \over (\text{Precision} + \text{Recall})}
$$

### Controlling the Number of Claims

Each sentence in the response and reference can be broken down into one or more claims. The number of claims that are generated from a single sentence is determined by the level of `atomicity` and `coverage` required for your application.


#### Example

```python
scorer = FactualCorrectness(mode="precision",atomicity="low")
```
Output
```
1.0
```


#### Understanding Atomicity and Coverage

In claim decomposition, two important parameters influence the output:

1. **Atomicity**
2. **Coverage**

These parameters help control the granularity and completeness of the generated claims.

#### Atomicity

**Atomicity** refers to how much a sentence is broken down into its smallest, meaningful components. It can be adjusted based on whether you need highly detailed claims or a more consolidated view.

- **High Atomicity**: The sentence is broken down into its fundamental, indivisible claims. This results in multiple, smaller claims, each representing a distinct piece of information.
  
  **Example:**
  - Original Sentence: 
    - "Albert Einstein was a German theoretical physicist who developed the theory of relativity and contributed to quantum mechanics."
  - Decomposed Claims:
    - "Albert Einstein was a German theoretical physicist."
    - "Albert Einstein developed the theory of relativity."
    - "Albert Einstein contributed to quantum mechanics."

- **Low Atomicity**: The sentence is kept more intact, resulting in fewer claims that may contain multiple pieces of information.
  
  **Example:**
  - Original Sentence:
    - "Albert Einstein was a German theoretical physicist who developed the theory of relativity and contributed to quantum mechanics."
  - Decomposed Claims:
    - "Albert Einstein was a German theoretical physicist who developed the theory of relativity and contributed to quantum mechanics."

#### Coverage

**Coverage** refers to how comprehensively the claims represent the information in the original sentence. It can be adjusted to either include all details or to generalize the content.

- **High Coverage**: The decomposed claims capture all the information present in the original sentence, preserving every detail.
  
  **Example:**
  - Original Sentence: 
    - "Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity."
  - Decomposed Claims:
    - "Marie Curie was a Polish physicist."
    - "Marie Curie was a naturalized-French physicist."
    - "Marie Curie was a chemist."
    - "Marie Curie conducted pioneering research on radioactivity."

- **Low Coverage**: The decomposed claims cover only the main points, omitting some details to provide a more generalized view.
  
  **Example:**
  - Original Sentence:
    - "Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity."
  - Decomposed Claims:
    - "Marie Curie was a physicist."
    - "Marie Curie conducted research on radioactivity."

#### Combining Atomicity and Coverage

By adjusting both atomicity and coverage, you can customize the level of detail and completeness to meet the needs of your specific use case.

- **High Atomicity & High Coverage**: Produces highly detailed and comprehensive claims that cover all aspects of the original sentence.

  **Example:**
  - Original Sentence:
    - "Charles Babbage was an English mathematician, philosopher, inventor, and mechanical engineer."
  - Decomposed Claims:
    - "Charles Babbage was an English mathematician."
    - "Charles Babbage was a philosopher."
    - "Charles Babbage was an inventor."
    - "Charles Babbage was a mechanical engineer."

- **Low Atomicity & Low Coverage**: Produces fewer claims with less detail, summarizing the main idea without going into specifics.

  **Example:**
  - Original Sentence:
    - "Charles Babbage was an English mathematician, philosopher, inventor, and mechanical engineer."
  - Decomposed Claims:
    - "Charles Babbage was an English mathematician."
    - "Charles Babbage was an inventor."

#### Practical Application

- Use **High Atomicity and High Coverage** when you need a detailed and comprehensive breakdown for in-depth analysis or information extraction.
- Use **Low Atomicity and Low Coverage** when only the key information is necessary, such as for summarization.

This flexibility in controlling the number of claims helps ensure that the information is presented at the right level of granularity for your application's requirements.

## Legacy Metrics API

The following examples use the legacy metrics API pattern. For new projects, we recommend using the collections-based API shown above.

!!! warning "Deprecation Timeline"
    This API will be deprecated in version 0.4 and removed in version 1.0. Please migrate to the collections-based API shown above.

### Example with SingleTurnSample

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._factual_correctness import FactualCorrectness


sample = SingleTurnSample(
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris. I has a height of 1000ft."
)

scorer = FactualCorrectness(llm = evaluator_llm)
await scorer.single_turn_ascore(sample)
```

Output:

```
0.67
```

### Changing the Mode

By default, the mode is set to `F1`, you can change the mode to `precision` or `recall` by setting the `mode` parameter.

```python
scorer = FactualCorrectness(llm = evaluator_llm, mode="precision")
```

Output:

```
1.0
```

### Controlling Atomicity

```python
scorer = FactualCorrectness(mode="precision", atomicity="low")
```

Output:

```
1.0
```
