# Noise Sensitivity

`NoiseSensitivity` measures how often a system makes errors by providing incorrect responses when utilizing either relevant or irrelevant retrieved documents. The score ranges from 0 to 1, with lower values indicating better performance. Noise sensitivity is computed using the `user_input`, `reference`, `response`, and the `retrieved_contexts`.

To estimate noise sensitivity, each claim in the generated response is examined to determine whether it is correct based on the ground truth and whether it can be attributed to the relevant (or irrelevant) retrieved context. Ideally, all claims in the answer should be supported by the relevant retrieved context.


$$
\text{noise sensitivity (relevant)} = {|\text{Total number of incorrect claims in response}| \over |\text{Total number of claims in the response}|}
$$


### Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import NoiseSensitivity

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = NoiseSensitivity(llm=llm)

# Evaluate
result = await scorer.ascore(
    user_input="What is the Life Insurance Corporation of India (LIC) known for?",
    response="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.",
    reference="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.",
    retrieved_contexts=[
        "The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.",
        "LIC is the largest insurance company in India, with a vast network of policyholders and huge investments.",
        "As the largest institutional investor in India, LIC manages substantial funds, contributing to the financial stability of the country.",
        "The Indian economy is one of the fastest-growing major economies in the world, thanks to sectors like finance, technology, manufacturing etc."
    ]
)
print(f"Noise Sensitivity Score: {result.value}")
```

Output:

```
Noise Sensitivity Score: 0.3333333333333333
```

To calculate noise sensitivity of irrelevant context, you can set the `mode` parameter to `irrelevant`:

```python
scorer = NoiseSensitivity(llm=llm, mode="irrelevant")
result = await scorer.ascore(
    user_input="What is the Life Insurance Corporation of India (LIC) known for?",
    response="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.",
    reference="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.",
    retrieved_contexts=[
        "The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.",
        "LIC is the largest insurance company in India, with a vast network of policyholders and huge investments.",
        "As the largest institutional investor in India, LIC manages substantial funds, contributing to the financial stability of the country.",
        "The Indian economy is one of the fastest-growing major economies in the world, thanks to sectors like finance, technology, manufacturing etc."
    ]
)
print(f"Noise Sensitivity (Irrelevant) Score: {result.value}")
```

Output:

```
Noise Sensitivity (Irrelevant) Score: 0.0
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        user_input="What is the Life Insurance Corporation of India (LIC) known for?",
        response="The Life Insurance Corporation of India (LIC) is the largest insurance company in India...",
        reference="The Life Insurance Corporation of India (LIC) is the largest insurance company...",
        retrieved_contexts=[...]
    )
    ```

## How It’s Calculated

!!! example
    Question: What is the Life Insurance Corporation of India (LIC) known for?

    Ground truth: The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.

    Relevant Retrieval:
        - The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.
        - LIC is the largest insurance company in India, with a vast network of policyholders and a significant role in the financial sector.
        - As the largest institutional investor in India, LIC manages a substantial life fund, contributing to the financial stability of the country.

    Irrelevant Retrieval:
        - The Indian economy is one of the fastest-growing major economies in the world, thanks to the sectors like finance, technology, manufacturing etc.

Let's examine how noise sensitivity in relevant context was calculated:

- **Step 1:** Identify the relevant contexts from which the ground truth can be inferred.

    - Ground Truth:
    The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.

    - Contexts:
        - Context 1: The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.
        - Context 2: LIC is the largest insurance company in India, with a vast network of policyholders and a significant role in the financial sector.
        - Context 3: As the largest institutional investor in India, LIC manages a substantial funds`, contributing to the financial stability of the country.

- **Step 2:** Verify if the claims in the generated answer can be inferred from the relevant context.

    - Answer:
    The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.

    - Contexts:
        - Context 1: The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.
        - Context 2: LIC is the largest insurance company in India, with a vast network of policyholders and a significant role in the financial sector.
        - Context 3: As the largest institutional investor in India, LIC manages a substantial funds, contributing to the financial stability of the country.


- **Step 3:** Identify any incorrect claims in the answer (i.e., answer statements that are not supported by the ground truth).

    - Ground Truth:
    The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.

    - Answer:
    The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.

    Explanation: The ground truth does not mention anything about LIC contributing to the financial stability of the country. Therefore, this statement in the answer is incorrect.

    Incorrect Statement: 1
    Total claims: 3

- **Step 4:** Calculate noise sensitivity using the formula:

    $$
    \text{noise sensitivity} = { \text{1} \over \text{3} } = 0.333
    $$

This results in a noise sensitivity score of 0.333, indicating that one out of three claims in the answer was incorrect.


## Legacy Metrics API

The following examples use the legacy metrics API pattern. For new projects, we recommend using the collections-based API shown above.

!!! warning "Deprecation Timeline"
    This API will be deprecated in version 0.4 and removed in version 1.0. Please migrate to the collections-based API shown above.

### Example with SingleTurnSample

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NoiseSensitivity

sample = SingleTurnSample(
    user_input="What is the Life Insurance Corporation of India (LIC) known for?",
    response="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.",
    reference="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.",
    retrieved_contexts=[
        "The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.",
        "LIC is the largest insurance company in India, with a vast network of policyholders and huge investments.",
        "As the largest institutional investor in India, LIC manages substantial funds, contributing to the financial stability of the country.",
        "The Indian economy is one of the fastest-growing major economies in the world, thanks to sectors like finance, technology, manufacturing etc."
    ]
)

scorer = NoiseSensitivity(llm=evaluator_llm)
await scorer.single_turn_ascore(sample)
```

Output:

```
0.3333333333333333
```

To calculate noise sensitivity of irrelevant context, you can set the `mode` parameter to `irrelevant`:

```python
scorer = NoiseSensitivity(mode="irrelevant")
await scorer.single_turn_ascore(sample)
```

Credits: Noise sensitivity was introduced in [RAGChecker](https://github.com/amazon-science/RAGChecker/tree/main/ragchecker)