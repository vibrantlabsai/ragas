# Migration from v0.1 to v0.2

v0.2 is the start of the transition for Ragas from an evaluation library for RAG pipelines to a more general library that you can use to evaluate any LLM applications you build. The meant we had to make some fundamental changes to the library that will break your workflow. Hopeful this guide will make that transition as easy as possible.

## Outline

1. Evaluation Dataset
2. Metrics
3. Testset Generation
4. Prompt Object

## Evaluation Dataset

We have moved from using HuggingFace [`Datasets`](https://huggingface.co/docs/datasets/v3.0.1/en/package_reference/main_classes#datasets.Dataset) to our own [`EvaluationDataset`][ragas.dataset_schema.EvaluationDataset]. You can read more about it from the core concepts section for [EvaluationDataset](../../concepts/components/eval_dataset.md) and [EvaluationSample](../../concepts/components/eval_sample.md)

You can easily translate

```python
from ragas import EvaluationDataset, SingleTurnSample

hf_dataset = ... # your huggingface evaluation dataset
eval_dataset = EvaluationDataset.from_hf_dataset(hf_dataset)

# save eval dataset
eval_dataset.to_csv("path/to/save/dataset.csv")

# load eva dataset
eval_dataset = EvaluationDataset.from_csv("path/to/save/dataset.csv")
```

## Metrics

All the default metrics are still supported, and many new metrics have been added. Take a look at the [documentation page](../../concepts/metrics/available_metrics/index.md) for the entire list.

However, there are a couple of changes in how you use metrics

Firstly it is now preferred to initialize metrics with the evaluator LLM of your choice as opposed to using the initialized version of the metrics into [`evaluate()`][ragas.evaluation.evaluate]. This avoids a lot of confusion regarding which LLMs are used where.

```python
from ragas.metrics import faithfullness # old way, not recommended but still supported till v0.3
from ragas.metrics import Faithfulness

# preffered way
faithfulness_metric = Faithfulness(llm=your_evaluator_llm)
```
Second is that [`metrics.ascore`][ragas.metrics.base.Metric.ascore] is now being deprecated in favor of [`metrics.single_score`][ragas.metrics.base.SingleTurnMetric.single_turn_ascore] . You can make the transition as such

```python
# create a Single Turn Sample
from ragas import SingleTurnSample

sample = SingleTurnSample(
    user_input="user query",
    response="response from your pipeline",
    retrieved_contexts=["retrieved", "contexts", "from your pipeline" ]
)

# Init the metric
from ragas.metrics import Faithfulness
faithfulness_metric = Faithfulness(llm=your_evaluator_llm)
await faithfulness_metric.single_turn_ascore(sample)
```
Output
```
1
```

## Testset Generation

[Testset Generation](../../concepts/test_data_generation/rag.md) has been redesigned to be much more cost-efficient. If you were using the end-to-end workflow checkout the [getting started](../../getstarted/rag_testset_generation.md).

**Notable Changes**

- Removed `Docstore` in favor of a new `Knowledge Graph`
- Added `Transforms` which will convert the documents passed into a rich knowledge graph
- More customizable with `Synthesizer` objects. Also refer to the documentation.
- New workflow makes it much cheaper and intermediate states can be saved easily

This might be a bit rough but if you do need help here, feel free to chat or mention it here and we would love to help you out 🙂

## Prompt Object

All the prompts have been rewritten to use [`PydanticPrompts`][ragas.prompt.pydantic_prompt.PydanticPrompt] which is based on [`BasePrompt`][ragas.prompt.base.BasePrompt] object. If you are using the old `Prompt` object you will have to upgrade it to the new one, check the docs to learn more on how to do it

- [How to Guide on how to create new prompts](./../customizations/metrics/_modifying-prompts-metrics.md)
- [GitHub PR for the changes](https://github.com/vibrantlabsai/ragas/pull/1462)

!!! note "Need Further Assistance?"

    If you have any further questions feel free to post them in this [github issue](https://github.com/vibrantlabsai/ragas/issues/1486) or reach out to us on [cal.com](https://cal.com/shahul-ragas/30min)

