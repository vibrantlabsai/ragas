# English-Uzbek Retrieval Evaluation Example for RAGAS

## Summary

This example contributes a small, inspectable English-Uzbek retrieval evaluation sample for culturally grounded RAG systems. The benchmark focuses on whether a retriever returns the intended supporting source document for questions about governance, history, institutions, and culture.

In this benchmark setting, targeted Uzbek source curation improved Uzbek retrieval recall from 39% to 98% without changing the underlying model. The repository results suggest that corpus coverage and source quality had a larger observed effect than embedding-model variation for this evaluation set.

## Motivation

Most RAG evaluation examples are centered on English or high-resource settings. This makes it difficult to test whether evaluation workflows expose failures caused by missing local source material, uneven language coverage, or culturally specific evidence requirements.

English-Uzbek is a useful case study because the benchmark pairs a high-resource language with a lower-resource Central Asian language and keeps the evaluation target concrete: can the system retrieve the relevant source document for a culturally grounded question?

## Contribution Scope

This RAGAS contribution adds release-ready benchmark material rather than expanding RAGAS into a dataset-hosting project. Its scope is:

- a compact English-Uzbek evaluation sample
- documentation for the full benchmark schema
- an example showing how to adapt the rows into RAGAS-style evaluation inputs
- guidance on which RAGAS metrics are appropriate once retrieved contexts and generated answers are available

The full project repository remains the source of methodology, result tables, and reproducibility notes.

## Benchmark Structure

The current `manual_eval_v5` data contains 400 evaluation items balanced across English and Uzbek, with four domains per language: governance, history, institutions, and culture.

Current fields are:

- `id`
- `language`
- `domain`
- `question`
- `gold_answer`
- `source_doc_ids`
- `answerable`
- `cultural_specificity`
- `source_title`
- `difficulty`
- `quality_flag`

The benchmark uses `source_doc_ids` as the auditable retrieval target. It does not require renaming fields for RAGAS; an adapter can map `question` to `user_input`, `gold_answer` to `reference`, and externally supplied retrieved passages to `retrieved_contexts`.

## How This Supports RAGAS

The benchmark can support several RAG evaluation tasks:

- Retrieval: check whether retrieved document IDs match `source_doc_ids`.
- Grounding: inspect whether generated answers are supported by retrieved contexts.
- Faithfulness: evaluate answers against retrieved contexts once real generated answers are supplied.
- Answer quality: compare generated answers against `gold_answer` with appropriate caution.

The benchmark is most directly useful for retrieval and context-grounding evaluation. It should not be presented as a complete generative answer-quality benchmark because the checked-in public workflow uses lightweight generation placeholders.

## Reproducibility

Users can inspect the public dataset files and run the local sample adapter:

```bash
python ragas_contribution/minimal_example.py --limit 5
```

The public repository includes:

- `data/eval/manual_eval_v5.jsonl`
- `data/eval/sample/manual_eval_v5_sample.jsonl`
- `docs/dataset_card_v5.md`
- `docs/results.md`
- `results/reports/statistical_analysis.md`
- `results/reports/manual_eval_v5_validation.md`

Full reruns require the excluded raw corpora, processed indexes, run outputs, and Isambard-AI environment described in the repository documentation.

## Results Summary

Validated retrieval-side results in the repository include:

- English baseline recall: 63%.
- Uzbek baseline recall: 39%.
- Uzbek recall after targeted corpus supplementation: 98%.
- English supplementation: retracted due to synthetic leakage and not reportable.

These results should be described as specific to this English-Uzbek benchmark and its experimental setup. They do not justify a broader conclusion about corpus coverage versus model choice outside this setting.

## Limitations

- The benchmark has 400 items and should be treated as moderate in size.
- It covers only English and Uzbek.
- Results are retrieval-side results, not full end-to-end answer-quality results.
- Public reruns are limited by excluded raw corpora, indexes, and HPC artifacts.
- Some `source_title` values could not be resolved from the supplied corpus.
- The English supplementation attempt was retracted and should not be used as evidence.
- Quality flags remain in the dataset for transparency rather than being removed.

## Affiliations

- SOAS University of London
- American University of Technology, Uzbekistan

## Source Repository

The full benchmark, methodology, result tables, and reproducibility notes live in the source repository:

https://github.com/rajantripathi/soas-rag-evaluation

This directory includes only a small public sample and adapter example so the documentation remains lightweight.
