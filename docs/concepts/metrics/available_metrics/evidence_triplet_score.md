# Evidence Triplet Score

The **Evidence Triplet Score** evaluates a generated RAG response against retrieved contexts using three independent evidence dimensions:

- `truth`: retrieved evidence supports the response.
- `indeterminacy`: retrieved evidence is missing, incomplete, ambiguous, pending, stale, or insufficient.
- `falsity`: retrieved evidence contradicts the response or exposes unsupported claims.

The metric is inspired by neutrosophic logic, introduced by Florentin Smarandache, where truth, indeterminacy, and falsity are treated as independent dimensions instead of a single confidence value.

## Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import EvidenceTripletScore

client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

metric = EvidenceTripletScore(llm=llm)

result = await metric.ascore(
    user_input="What caused the system outage?",
    response="The outage was caused by a database failover.",
    retrieved_contexts=[
        "The database failover completed successfully.",
        "Network routing errors were detected during the outage window.",
        "Root cause analysis is still pending.",
    ],
)

print(result.value)
print(result.reason)
```

Output:

```python
{
    "truth": 0.2,
    "indeterminacy": 0.7,
    "falsity": 0.6,
}
```

## How To Interpret It

Unlike a scalar metric, the triplet does not force all evidence states into one value.

```text
truth=0.20, indeterminacy=0.80, falsity=0.10
```

This means the system does not have enough evidence.

```text
truth=0.20, indeterminacy=0.20, falsity=0.80
```

This means the retrieved evidence contradicts the response.

Both cases may be risky, but they require different remediation.

## Similar Ragas Metrics

- [Faithfulness](faithfulness.md): evaluates whether generated claims are supported by retrieved context.
- [Context Relevance](nvidia_metrics.md#context-relevance): evaluates whether retrieved contexts are relevant to the user input.
- [Response Groundedness](nvidia_metrics.md#response-groundedness): evaluates whether a response is grounded in retrieved contexts.
