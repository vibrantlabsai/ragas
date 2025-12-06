## Semantic Similarity

The **Semantic Similarity** metric evaluates the semantic resemblance between a generated response and a reference (ground truth) answer. It ranges from 0 to 1, with higher scores indicating better alignment between the generated answer and the ground truth.

This metric uses embeddings and cosine similarity to measure how semantically similar two answers are, which can offer valuable insights into the quality of the generated response.


### Example

```python
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import SemanticSimilarity

# Setup embeddings
client = AsyncOpenAI()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", client=client)

# Create metric
scorer = SemanticSimilarity(embeddings=embeddings)

# Evaluate
result = await scorer.ascore(
    reference="The Eiffel Tower is located in Paris. It has a height of 1000ft.",
    response="The Eiffel Tower is located in Paris."
)
print(f"Semantic Similarity Score: {result.value}")
```

Output:

```
Semantic Similarity Score: 0.8151
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        reference="The Eiffel Tower is located in Paris. It has a height of 1000ft.",
        response="The Eiffel Tower is located in Paris."
    )
    ```


### How It's Calculated 

!!! example

    **Reference**: Albert Einstein's theory of relativity revolutionized our understanding of the universe.

    **High similarity response**: Einstein's groundbreaking theory of relativity transformed our comprehension of the cosmos.

    **Low similarity response**: Isaac Newton's laws of motion greatly influenced classical physics.

Let's examine how semantic similarity was calculated for the high similarity response:

- **Step 1:** Vectorize the reference answer using the specified embedding model.
- **Step 2:** Vectorize the generated response using the same embedding model.
- **Step 3:** Compute the cosine similarity between the two vectors.
- **Step 4:** The cosine similarity value (0-1) is the final score.


## Legacy Metrics API

The following examples use the legacy metrics API pattern. For new projects, we recommend using the collections-based API shown above.

!!! warning "Deprecation Timeline"
    This API will be deprecated in version 0.4 and removed in version 1.0. Please migrate to the collections-based API shown above.

### Example with SingleTurnSample

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import SemanticSimilarity
from ragas.embeddings import LangchainEmbeddingsWrapper

sample = SingleTurnSample(
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris. It has a height of 1000ft."
)

scorer = SemanticSimilarity(embeddings=LangchainEmbeddingsWrapper(evaluator_embedding))
await scorer.single_turn_ascore(sample)
```

Output:

```
0.8151371879226978
```
