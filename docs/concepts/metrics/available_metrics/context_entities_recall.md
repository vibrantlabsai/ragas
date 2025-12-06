## Context Entities Recall

`ContextEntityRecall` metric gives the measure of recall of the retrieved context, based on the number of entities present in both `reference` and `retrieved_contexts` relative to the number of entities present in the `reference` alone. Simply put, it is a measure of what fraction of entities is recalled from `reference`. This metric is useful in fact-based use cases like tourism help desk, historical QA, etc. This metric can help evaluate the retrieval mechanism for entities, based on comparison with entities present in `reference`, because in cases where entities matter, we need the `retrieved_contexts` which cover them.

To compute this metric, we use two sets:

- **$RE$**: The set of entities in the reference.
- **$RCE$**: The set of entities in the retrieved contexts.

We calculate the number of entities common to both sets ($RCE \cap RE$) and divide it by the total number of entities in the reference ($RE$). The formula is:

$$
\text{Context Entity Recall} = \frac{\text{Number of common entities between $RCE$ and $RE$}}{\text{Total number of entities in $RE$}}
$$


### Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextEntityRecall

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = ContextEntityRecall(llm=llm)

# Evaluate
result = await scorer.ascore(
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Eiffel Tower is located in Paris."]
)
print(f"Context Entity Recall Score: {result.value}")
```

Output:
```
Context Entity Recall Score: 0.999999995
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=["The Eiffel Tower is located in Paris."]
    )
    ```

### How It’s Calculated



!!! example
    **reference**: The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in the Indian city of Agra. It was commissioned in 1631 by the Mughal emperor Shah Jahan to house the tomb of his favorite wife, Mumtaz Mahal.
    **High entity recall context**: The Taj Mahal is a symbol of love and architectural marvel located in Agra, India. It was built by the Mughal emperor Shah Jahan in memory of his beloved wife, Mumtaz Mahal. The structure is renowned for its intricate marble work and beautiful gardens surrounding it.
    **Low entity recall context**: The Taj Mahal is an iconic monument in India. It is a UNESCO World Heritage Site and attracts millions of visitors annually. The intricate carvings and stunning architecture make it a must-visit destination.

Let us consider the reference and the retrieved contexts given above.

- **Step-1**: Find entities present in the reference.
    - Entities in ground truth (RE) - ['Taj Mahal', 'Yamuna', 'Agra', '1631', 'Shah Jahan', 'Mumtaz Mahal']
- **Step-2**: Find entities present in the retrieved contexts.
    - Entities in context (RCE1) - ['Taj Mahal', 'Agra', 'Shah Jahan', 'Mumtaz Mahal', 'India']
    - Entities in context (RCE2) - ['Taj Mahal', 'UNESCO', 'India']
- **Step-3**: Use the formula given above to calculate entity-recall

    $$
    \text{context entity recall 1} = \frac{| RCE1 \cap RE |}{| RE |}
                                 = 4/6
                                 = 0.666
    $$

    $$
    \text{context entity recall 2} = \frac{| RCE2 \cap RE |}{| RE |}
                                 = 1/6
    $$

    We can see that the first context had a high entity recall, because it has a better entity coverage given the reference. If these two retrieved contexts were fetched by two retrieval mechanisms on same set of documents, we could say that the first mechanism was better than the other in use-cases where entities are of importance.

## Legacy Metrics API

The following examples use the legacy metrics API pattern. For new projects, we recommend using the collections-based API shown above.

!!! warning "Deprecation Timeline"
    This API will be deprecated in version 0.4 and removed in version 1.0. Please migrate to the collections-based API shown above.

### Example with SingleTurnSample

```python
from ragas import SingleTurnSample
from ragas.metrics import ContextEntityRecall

sample = SingleTurnSample(
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["The Eiffel Tower is located in Paris."],
)

scorer = ContextEntityRecall(llm=evaluator_llm)

await scorer.single_turn_ascore(sample)
```
Output:
```
0.999999995
```
