# Example Usage for RAGAS Contribution

This example shows how the checked-in English-Uzbek evaluation rows can be adapted for RAGAS-style evaluation without changing the repository schema.

## Local Data

Use the public sample first:

```bash
python ragas_contribution/minimal_example.py --input data/eval/sample/manual_eval_v5_sample.jsonl --limit 5
```

Use the full checked-in evaluation set for inspection:

```bash
python ragas_contribution/minimal_example.py --input data/eval/manual_eval_v5.jsonl --limit 5
```

The script validates required fields and prints preview records with the shape:

```json
{
  "user_input": "question text",
  "reference": "gold answer text",
  "retrieved_contexts": [],
  "response": "",
  "metadata": {
    "id": "en_00",
    "language": "en",
    "domain": "culture",
    "source_doc_ids": ["Art Deco"],
    "source_title": "Art Deco",
    "difficulty": "medium",
    "quality_flag": null
  }
}
```

## Mapping to RAGAS Concepts

| Repository field | RAGAS-style use |
| --- | --- |
| `question` | `user_input` |
| `gold_answer` | `reference` |
| retrieved passage text from a RAG system | `retrieved_contexts` |
| generated answer from a RAG system | `response` |
| `source_doc_ids` | retrieval target in metadata |
| `language`, `domain`, `difficulty`, `quality_flag` | analysis slices in metadata |

The committed dataset contains questions, references, and source identifiers. It does not contain retrieved contexts or generated answers for arbitrary user systems. Those fields must be supplied by the user's RAG pipeline before running context or answer metrics.

## Suggested Evaluation Flow

1. Load `manual_eval_v5.jsonl`.
2. Run each `question` through a RAG pipeline.
3. Store retrieved contexts and document IDs.
4. Compute retrieval recall by checking whether any retrieved document ID matches `source_doc_ids`.
5. Pass `user_input`, `retrieved_contexts`, `response`, and `reference` to RAGAS metrics that match the available evidence.
6. Report results by language and domain.

## Metrics Fit

Appropriate when retrieved contexts and answers are available:

- context relevance or context precision-style checks
- faithfulness checks against retrieved contexts
- answer correctness or semantic similarity checks against `gold_answer`

Appropriate from repository fields alone:

- schema validation
- language/domain slicing
- retrieval-target preparation
- source coverage analysis through `source_doc_ids`

Not supported by repository fields alone:

- final RAGAS scoring for a live model
- claims about generated answer quality
- cross-lingual retrieval results

## Claim Wording

Use careful wording:

- In this benchmark setting, targeted Uzbek source curation improved retrieval recall from 39% to 98%.
- On this English-Uzbek evaluation set, corpus coverage and curation had a larger observed effect than embedding-model variation.
- The result suggests source coverage can be a first-order bottleneck in culturally grounded low-resource RAG evaluation.

Avoid wording:

- Broad claims about corpus coverage versus model choice outside this evaluation setting.
- Proof-style claims about all low-resource RAG settings.
- Claims about best-in-class answer generation.

## TODOs

TODO: Add final Hugging Face dataset URL after upload.
TODO: Add exact RAGAS PR path after maintainer guidance.
TODO: Add an end-to-end RAGAS metric example only after retrieved contexts and generated answers are included.
