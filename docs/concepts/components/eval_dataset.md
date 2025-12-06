# Evaluation Dataset

An evaluation dataset is a homogeneous collection of [data samples](eval_sample.md) designed to assess the performance and capabilities of an AI application. In Ragas, evaluation datasets are represented using the `EvaluationDataset` class, which provides a structured way to organize and manage data samples for evaluation purposes. 

- [Overview](#overview)
- [Creating an Evaluation Dataset from SingleTurnSamples](#creating-an-evaluation-dataset-from-singleturnsamples)
- [Loading an Evaluation Dataset from Hugging Face Datasets](#loading-an-evaluation-dataset-from-hugging-face-datasets)

## Overview

### Structure of an Evaluation Dataset

An evaluation dataset consists of:

- **Samples**: A collection of [SingleTurnSample](eval_sample.md#singleturnsample) or [MultiTurnSample](eval_sample.md#multiturnsample) instances. Each sample represents a unique interaction or scenario.
- **Consistency**: All samples within the dataset should be of the same type (either all single-turn or all multi-turn samples) to maintain consistency in evaluation.


### Guidelines for Curating an Effective Evaluation Dataset

- **Define Clear Objectives**: Identify the specific aspects of the AI application that you want to evaluate and the scenarios you want to test. Collect data samples that reflect these objectives.

- **Collect Representative Data**: Ensure that the dataset covers a diverse range of scenarios, user inputs, and expected responses to provide a comprehensive evaluation of the AI application. This can be achieved by collecting data from various sources or [generating synthetic data](./../../howtos/customizations/index.md#testset-generation).

- **Quality and Size**: Aim for a dataset that is large enough to provide meaningful insights but not so large that it becomes unwieldy. Ensure that the data is of high quality and accurately reflects the real-world scenarios you want to evaluate.


## Creating an Evaluation Dataset from SingleTurnSamples

In this example, we’ll demonstrate how to create an EvaluationDataset using multiple `SingleTurnSample` instances. We’ll walk through the process step by step, including creating individual samples, assembling them into a dataset, and performing basic operations on the dataset.


**Step 1:** Import Necessary Classes

First, import the SingleTurnSample and EvaluationDataset classes from your module.
```python
from ragas import SingleTurnSample, EvaluationDataset
```

**Step 2:** Create Individual Samples

Create several SingleTurnSample instances that represent individual evaluation samples.

```python
# Sample 1
sample1 = SingleTurnSample(
    user_input="What is the capital of Germany?",
    retrieved_contexts=["Berlin is the capital and largest city of Germany."],
    response="The capital of Germany is Berlin.",
    reference="Berlin",
)

# Sample 2
sample2 = SingleTurnSample(
    user_input="Who wrote 'Pride and Prejudice'?",
    retrieved_contexts=["'Pride and Prejudice' is a novel by Jane Austen."],
    response="'Pride and Prejudice' was written by Jane Austen.",
    reference="Jane Austen",
)

# Sample 3
sample3 = SingleTurnSample(
    user_input="What's the chemical formula for water?",
    retrieved_contexts=["Water has the chemical formula H2O."],
    response="The chemical formula for water is H2O.",
    reference="H2O",
)
```

**Step 3:** Create the EvaluationDataset

Create an EvaluationDataset by passing a list of SingleTurnSample instances.

```python
dataset = EvaluationDataset(samples=[sample1, sample2, sample3])
``` 

## Loading an Evaluation Dataset from Hugging Face Datasets

In practice, you may want to load an evaluation dataset from an existing dataset source, such as the Hugging Face Datasets library. The following example demonstrates how to load an evaluation dataset from a Hugging Face dataset and convert it into an EvaluationDataset instance.

Ensure that the dataset contains the necessary fields for evaluation, such as user inputs, retrieved contexts, responses, and references.

```python
from datasets import load_dataset
dataset = load_dataset("vibrantlabsai/amnesty_qa","english_v3")
```

Load the dataset into a Ragas EvaluationDataset object.

```python
from ragas import EvaluationDataset

eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])
```
