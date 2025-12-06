## Faithfulness

The **Faithfulness** metric measures how factually consistent a `response` is with the `retrieved context`. It ranges from 0 to 1, with higher scores indicating better consistency.

A response is considered **faithful** if all its claims can be supported by the retrieved context.

To calculate this:
1. Identify all the claims in the response.
2. Check each claim to see if it can be inferred from the retrieved context.
3. Compute the faithfulness score using the formula:

$$
\text{Faithfulness Score} = \frac{\text{Number of claims in the response supported by the retrieved context}}{\text{Total number of claims in the response}}
$$


### Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = Faithfulness(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=[
        "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
    ]
)
print(f"Faithfulness Score: {result.value}")
```

Output:

```
Faithfulness Score: 1.0
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[...]
    )
    ```


### How It’s Calculated

!!! example
    **Question**: Where and when was Einstein born?

    **Context**: Albert Einstein (born 14 March 1879) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time

    **High faithfulness answer**: Einstein was born in Germany on 14th March 1879.

    **Low faithfulness answer**:  Einstein was born in Germany on 20th March 1879.

Let's examine how faithfulness was calculated using the low faithfulness answer:

- **Step 1:** Break the generated answer into individual statements.
    - Statements:
        - Statement 1: "Einstein was born in Germany."
        - Statement 2: "Einstein was born on 20th March 1879."

- **Step 2:** For each of the generated statements, verify if it can be inferred from the given context.
    - Statement 1: Yes
    - Statement 2: No

- **Step 3:** Use the formula depicted above to calculate faithfulness.

    $$
    \text{Faithfulness} = { \text{1} \over \text{2} } = 0.5
    $$


## Legacy Metrics API

The following examples use the legacy metrics API pattern. For new projects, we recommend using the collections-based API shown above.

!!! warning "Deprecation Timeline"
    This API will be deprecated in version 0.4 and removed in version 1.0. Please migrate to the collections-based API shown above.

### Example with SingleTurnSample

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness

sample = SingleTurnSample(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ]
    )
scorer = Faithfulness(llm=evaluator_llm)
await scorer.single_turn_ascore(sample)
```

Output:

```
1.0
```

### Faithfulness with HHEM-2.1-Open

[Vectara's HHEM-2.1-Open](https://vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model/) is a classifier model (T5) that is trained to detect hallucinations from LLM generated text. This model can be used in the second step of calculating faithfulness, i.e. when claims are cross-checked with the given context to determine if it can be inferred from the context. The model is free, small, and open-source, making it very efficient in production use cases.

To use the model to calculate faithfulness:

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import FaithfulnesswithHHEM


sample = SingleTurnSample(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ]
    )
scorer = FaithfulnesswithHHEM(llm=evaluator_llm)
await scorer.single_turn_ascore(sample)
```

You can load the model onto a specified device by setting the `device` argument and adjust the batch size for inference using the `batch_size` parameter. By default, the model is loaded on the CPU with a batch size of 10:

```python
my_device = "cuda:0"
my_batch_size = 10

scorer = FaithfulnesswithHHEM(device=my_device, batch_size=my_batch_size)
await scorer.single_turn_ascore(sample)
```
