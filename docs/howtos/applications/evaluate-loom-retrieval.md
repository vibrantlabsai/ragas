# Evaluate document retrieval from Loom

Use Ragas ID-based context precision and recall to measure whether [ClickHouse Loom](https://github.com/ClickHouse/loom) retrieves the documents you expect. The test compares source URLs with a labeled set, so it can run without an LLM judge.

## Label a query

Use a Loom namespace with indexed test documents. For each query, record the `source_uri` of every relevant document. The example uses two references for a deployment question. Replace the reference URLs with the values you indexed.

Install the version used in this example with `pip install ragas==0.4.4`. Set `LOOM_BASE_URL` to your server's HTTPS origin, `LOOM_ORG` and `LOOM_NAMESPACE` to the test namespace, and `LOOM_VIEWER_TOKEN` to a viewer token scoped to that namespace.

## Score the retrieved sources

```python
import os

import requests

from ragas import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall

query = "What checks should I run before a deployment?"
reference_urls = [
    "https://example.com/runbooks/deploy",
    "https://example.com/runbooks/preflight",
]

origin = os.environ["LOOM_BASE_URL"].rstrip("/")
response = requests.post(
    f"{origin}/v1/knowledge.search",
    headers={"Authorization": f"Bearer {os.environ['LOOM_VIEWER_TOKEN']}"},
    json={
        "org": os.environ["LOOM_ORG"],
        "namespace": os.environ["LOOM_NAMESPACE"],
        "namespace_scope": "this",
        "query": query,
        "top_k": 5,
        "search_mode": "lexical",
        "rerank": "none",
    },
    timeout=30,
    allow_redirects=False,
)
if response.status_code != 200:
    raise RuntimeError(f"Loom returned HTTP {response.status_code}")

sample = SingleTurnSample(
    retrieved_context_ids=[hit["source_uri"] for hit in response.json()["results"]],
    reference_context_ids=reference_urls,
)

precision = IDBasedContextPrecision().single_turn_score(sample)
recall = IDBasedContextRecall().single_turn_score(sample)
print(f"Document precision: {precision:.2f}")
print(f"Document recall: {recall:.2f}")
```

Loom returns chunks. The ID-based metrics count each source URL once even if several chunks come from the same document. If the search returns one of the two references and one unrelated source, both scores are `0.50`. For an empty result, recall is `0.00` and precision is undefined (`nan`). Use the same labeled queries and retrieval limit when comparing runs.
