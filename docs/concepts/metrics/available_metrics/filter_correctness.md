# Filter Correctness

`FilterCorrectness` is a **non-LLM** collections metric that checks whether retrieved
chunks respect a metadata filter that was supposed to constrain retrieval
(tenant isolation, jurisdiction scoping, access-tier / RBAC, category filters, etc.).

It answers a structural question: *among the chunks that came back, how many violate
the where-clause?* It does **not** prove that a filter could never leak under every
possible query — it only inspects the returned set. Report `checked` alongside
`violations` in `traces["output"]`; empty retrieval and missing metadata are
distinct statuses that score `0.0`, not a perfect filter.

## Score

| Situation | `result.value` | `traces["output"]["status"]` |
|-----------|----------------|------------------------------|
| All checked chunks respect the filter | `1.0` | `scored` |
| Some checked chunks leak | `1 - violations/checked` | `scored` |
| Empty retrieval | `0.0` | `empty_retrieval` |
| Chunks exist but none expose `filter_field` | `0.0` | `missing_metadata` |

Chunks **missing** `filter_field` are counted under `missing_metadata` and are **not**
counted as violations. Empty retrieval and all-missing metadata intentionally score
`0.0` so they are not mistaken for a perfect filter.

## Example

```python
from ragas.metrics.collections import FilterCorrectness

scorer = FilterCorrectness()

# Equality filter (tenant isolation)
result = await scorer.ascore(
    retrieved_metadatas=[
        {"tenant_id": "A", "text": "..."},
        {"tenant_id": "B", "text": "..."},  # leak
        {"tenant_id": "A", "text": "..."},
    ],
    filter_field="tenant_id",
    allowed_values="A",
)
print(result.value)   # 0.6667  (1 of 3 checked chunks violated)
print(result.reason)  # includes offending values
print(result.traces["output"])
# {'status': 'scored', 'checked': 3, 'violations': 1, ...}

# Membership / $in-style filter
result = await scorer.ascore(
    retrieved_metadatas=[
        {"category": "public"},
        {"category": "partner"},
    ],
    filter_field="category",
    allowed_values=["public", "partner"],
)
print(result.value)  # 1.0
```

!!! note "Synchronous usage"
    Use `.score()` instead of `.ascore()` from a sync context.

## When to use it

- Multi-tenant SaaS RAG regression suites
- Jurisdiction-isolated legal / compliance retrieval
- Access-tier / RBAC-scoped retrieval
- Catching refactors that silently drop a `where`-clause

Pair it with quality metrics such as `Faithfulness` / `ContextPrecision`: those score
answer quality; `FilterCorrectness` scores filter mechanism integrity.
