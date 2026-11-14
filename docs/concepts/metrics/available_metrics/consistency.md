# Consistency

LLM applications are non-deterministic: the same question can get a different answer, or different retrieved documents, each time it is asked. Consistency metrics measure this directly by running the **same input several times** and comparing the runs **with each other**. They need no reference answer and no relevance labels, so they also work on data nobody has labelled, and they are a quick way to check whether a model, prompt or index change altered behaviour.

!!! note
    Consistency is not correctness. A system that gives the same wrong answer every time is perfectly consistent. Use these metrics alongside correctness metrics such as [Factual Correctness](factual_correctness.md) or [Context Precision](context_precision.md).

Comparing every pair of `N` runs takes `N(N-1)/2` comparisons. Both metrics instead compare a random cycle of `N` pairs by default, in which every run is compared at least twice; set `max_comparisons` to change this.

## Answer Consistency

`AnswerConsistency` uses an LLM judge to compare pairs of responses to the same `user_input`. For each pair the judge rates agreement from 1 (completely different) to 5 (identical) on five dimensions:

| Dimension | Weight | Question |
|---|---|---|
| `core_facts` | 0.40 | Do both give the same central answer? |
| `specifics` | 0.25 | Do names, numbers, dates and identifiers match? |
| `supporting_detail` | 0.15 | Do explanations, examples and caveats agree? |
| `stance` | 0.10 | Same stance and certainty (answer, hedge or refuse)? |
| `format` | 0.10 | Same structure and length? |

The pair score is the weighted mean, rescaled to 0-1, and the metric is the mean over compared pairs:

$$
\text{Answer Consistency} = \frac{1}{|P|} \sum_{(i, j) \in P} \sum_{d} w_d \cdot \frac{s_d(r_i, r_j) - 1}{4}
$$

The judge only compares the responses with each other; it does not decide which is more correct. Identical responses (ignoring case and whitespace) are scored 1.0 without an LLM call. The `reason` lists the mean agreement per dimension, so you can see *what* varies between runs (for example, stable facts but drifting format), and names the most different pair of runs.

### Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerConsistency

llm = llm_factory("gpt-4o-mini", client=AsyncOpenAI())
scorer = AnswerConsistency(llm=llm)

# Responses from running your application three times on the same input
result = await scorer.ascore(
    user_input="How long are backups kept on the gold policy?",
    responses=[
        "Backups on the gold policy are kept for 30 days.",
        "The gold policy retains backups for 30 days.",
        "Gold policy backups are kept for 90 days.",
    ],
)
print(result.value)
print(result.reason)
```

## Retrieval Consistency

`RetrievalConsistency` measures whether a retriever returns the same evidence for the same input across runs. It uses no LLM. Pass the retrieved chunks of each run, either as IDs or as texts:

- `retrieved_context_ids_runs`: runs are compared by the Jaccard similarity of their chunk-ID sets. Exact and cheap, but re-chunking or re-indexing changes the IDs.
- `retrieved_contexts_runs`: runs are compared by the Jaccard similarity of the word n-grams (`ngram_size`, default 3) of their pooled chunk text, so the same evidence split into different chunks still matches.

$$
\text{Retrieval Consistency} = \frac{1}{|P|} \sum_{(i, j) \in P} \frac{|C_i \cap C_j|}{|C_i \cup C_j|}
$$

where $C_i$ is the set of chunk IDs (or word n-grams) retrieved in run $i$. Order within a run is ignored.

### Example

```python
from ragas.metrics.collections import RetrievalConsistency

scorer = RetrievalConsistency()

result = await scorer.ascore(
    retrieved_context_ids_runs=[
        ["doc-12", "doc-7", "doc-3"],
        ["doc-7", "doc-12", "doc-3"],
        ["doc-12", "doc-7", "doc-41"],
    ]
)
print(result.value)
print(result.reason)
```

Output:

```
0.6666666666666666
Mean Jaccard similarity of chunk IDs over 3 pairs of 3 runs; most different: runs 1 and 3 (0.50).
```

!!! note "Synchronous Usage"
    Both metrics also offer `.score()`, and `.abatch_score()` to score many inputs at once:

    ```python
    results = scorer.batch_score(
        [
            {"retrieved_context_ids_runs": [["a", "b"], ["a", "b"]]},
            {"retrieved_context_ids_runs": [["a", "b"], ["a", "c"]]},
        ]
    )
    ```

### Comparing two versions of a system

To check whether an upgrade changed behaviour, score the runs of each version, or compare the versions directly while accounting for each one's own run-to-run noise with [repeatability-evals](https://github.com/krishna-gogineni-765/repeatability-evals).
