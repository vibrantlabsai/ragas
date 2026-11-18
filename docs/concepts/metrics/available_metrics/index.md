# List of available metrics

Ragas provides a set of evaluation metrics that can be used to measure the performance of your LLM application. These metrics are designed to help you objectively measure the performance of your application. Metrics are available for different applications and tasks, such as RAG and Agentic workflows. 

Each metric are essentially paradigms that are designed to evaluate a particular aspect of the application. LLM Based metrics might use one or more LLM calls to arrive at the score or result. One can also modify or write your own metrics using ragas.

## Retrieval Augmented Generation
- [Context Precision](context_precision.md)
- [Context Recall](context_recall.md)
- [Context Entities Recall](context_entities_recall.md)
- [Noise Sensitivity](noise_sensitivity.md)
- [Evidence Triplet Score](evidence_triplet_score.md)
- [Response Relevancy](answer_relevance.md)
- [Faithfulness](faithfulness.md)
- [Multimodal Faithfulness](multi_modal_faithfulness.md)
- [Multimodal Relevance](multi_modal_relevance.md)

## Nvidia Metrics
- [Answer Accuracy](nvidia_metrics.md#answer-accuracy)
- [Context Relevance](nvidia_metrics.md#context-relevance)
- [Response Groundedness](nvidia_metrics.md#response-groundedness)

## Agents or Tool use cases

- [Topic adherence](agents.md#topic-adherence)
- [Tool call Accuracy](agents.md#tool-call-accuracy)
- [Tool Call F1](agents.md#tool-call-f1)
- [Agent Goal Accuracy](agents.md#agent-goal-accuracy)

## Natural Language Comparison

- [Factual Correctness](factual_correctness.md)
- [Semantic Similarity](semantic_similarity.md)
- [Non LLM String Similarity](traditional.md#non-llm-string-similarity)
- [BLEU Score](traditional.md#bleu-score)
- [CHRF Score](traditional.md#chrf-score)
- [ROUGE Score](traditional.md#rouge-score)
- [String Presence](traditional.md#string-presence)
- [Exact Match](traditional.md#exact-match)


## SQL

- [Execution based Datacompy Score](sql.md#execution-based-metrics)
- [SQL query Equivalence](sql.md#sql-query-semantic-equivalence)

## General purpose

- [Aspect critic](general_purpose.md#aspect-critic)
- [Simple Criteria Scoring](general_purpose.md#simple-criteria-scoring)
- [Rubrics based scoring](general_purpose.md#rubrics-based-scoring)
- [Instance specific rubrics scoring](general_purpose.md#instance-specific-rubrics-scoring)

## Other tasks

- [Summarization](summarization_score.md)
