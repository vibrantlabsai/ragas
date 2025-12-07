## Answer Correctness

The assessment of Answer Correctness involves gauging the accuracy of the generated answer when compared to the ground truth. This evaluation relies on the `ground truth` and the `answer`, with scores ranging from 0 to 1. A higher score indicates a closer alignment between the generated answer and the ground truth, signifying better correctness.

Answer correctness encompasses two critical aspects: semantic similarity between the generated answer and the ground truth, as well as factual similarity. These aspects are combined using a weighted scheme to formulate the answer correctness score. Users also have the option to employ a 'threshold' value to round the resulting score to binary, if desired.

!!! note "Embedding Requirement"
    AnswerCorrectness requires embeddings for semantic similarity calculation. When using `evaluate()` without explicitly providing embeddings, Ragas will automatically match the embedding provider to your LLM provider. For example, if you use Gemini as your LLM, Google embeddings will be used automatically (no OpenAI API key needed). You can also provide embeddings explicitly for full control.


!!! example
    **Ground truth**: Einstein was born in 1879 in Germany.

    **High answer correctness**: In 1879, Einstein was born in Germany.

    **Low answer correctness**: Einstein was born in Spain in 1879.


### Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerCorrectness

# Setup LLM and embeddings
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

# Create metric
scorer = AnswerCorrectness(llm=llm, embeddings=embeddings)

# Evaluate
result = await scorer.ascore(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    reference="The first superbowl was held on January 15, 1967"
)
print(f"Answer Correctness Score: {result.value}")
```

Output:

```
Answer Correctness Score: 0.95
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        reference="The first superbowl was held on January 15, 1967"
    )
    ```

### Calculation

Let's calculate the answer correctness for the answer with low answer correctness. It is computed as the sum of factual correctness and the semantic similarity between the given answer and the ground truth.

Factual correctness quantifies the factual overlap between the generated answer and the ground truth answer. This is done using the concepts of:
- TP (True Positive): Facts or statements that are present in both the ground truth and the generated answer.
- FP (False Positive): Facts or statements that are present in the generated answer but not in the ground truth.
- FN (False Negative): Facts or statements that are present in the ground truth but not in the generated answer.

In the second example:
- TP: `[Einstein was born in 1879]`
- FP: `[Einstein was born in Spain]`
- FN: `[Einstein was born in Germany]`

Now, we can use the formula for the F1 score to quantify correctness based on the number of statements in each of these lists:


$$
\text{F1 Score} = {|\text{TP} \over {(|\text{TP}| + 0.5 \times (|\text{FP}| + |\text{FN}|))}}
$$

Next, we calculate the semantic similarity between the generated answer and the ground truth. Read more about it [here](./semantic_similarity.md).


Once we have the semantic similarity, we take a weighted average of the semantic similarity and the factual similarity calculated above to arrive at the final score. You can adjust this weightage by modifying the `weights` parameter.

## Legacy Metrics API

The following examples use the legacy metrics API pattern. For new projects, we recommend using the collections-based API shown above.

!!! warning "Deprecation Timeline"
    This API will be deprecated in version 0.4 and removed in version 1.0. Please migrate to the collections-based API shown above.

### Example with Dataset

```python
from datasets import Dataset 
from ragas.metrics import answer_correctness
from ragas import evaluate

data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
}
dataset = Dataset.from_dict(data_samples)
score = evaluate(dataset,metrics=[answer_correctness])
score.to_pandas()
```
