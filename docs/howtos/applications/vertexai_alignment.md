# Aligning LLM Evaluators with Human Judgment

This tutorial is part of a three-part series on how to use Vertex AI models with Ragas. It is recommended that you have gone through [Getting Started: Ragas with Vertex AI](./vertexai_x_ragas.md), even if you have not, you can easily follow this. You can navigate to the Model Comparison tutorial using the [link](./vertexai_model_comparision.md).

## Overview

In this tutorial, you will learn how to train and align your own custom LLM-based metric using Ragas. While LLM-based evaluators offer a powerful means of scoring AI applications, they can sometimes produce judgments that diverge from human expectations due to differences in style, context, or subtle nuances. By following this guide, you will refine your metric so that it more accurately mirrors human judgment.

In this tutorial, you will:

1. Define a model-based metric using Ragas.
2. Construct an EvaluationDataset from the "helpful" subset of the HHH dataset.
3. Run an initial evaluation to benchmark the metric’s performance.
4. Review and annotate 15–20 evaluation examples.
5. Train the metric using your annotated data.
6. Reevaluate the metric to observe improvements in alignment with human judgments.

## Getting Started

### Install Dependencies


```python
%pip install --upgrade --user --quiet langchain-core langchain-google-vertexai langchain ragas
```

### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.


```python
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.


```python
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### Set Google Cloud project information and initialize Vertex AI SDK


```python
PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    raise ValueError("Please set your PROJECT_ID")


import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

## Set up eval metrics

LLM-based metrics have tremendous potential but can sometimes misjudge responses compared to human evaluators. To bridge this gap, we align our model-based metric with human judgment using a feedback loop.

### Define evaluator_llm

Import the required wrappers and define your evaluator LLM and embedder.


```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings


evaluator_llm = LangchainLLMWrapper(VertexAI(model_name="gemini-2.0-flash-001"))
evaluator_embeddings = LangchainEmbeddingsWrapper(
    VertexAIEmbeddings(model_name="text-embedding-004")
)
```

### Ragas metrics

Ragas offers various model-based metrics that can be fine-tuned to align with human evaluators. For demonstration, we will use the **Aspect Critic** metric—a user-defined, binary metric. For further details, please refer to the [Aspect Critic documentation](../../concepts/metrics/available_metrics/general_purpose.md/#aspect-critic).


```python
from ragas.metrics import AspectCritic

helpfulness_critic = AspectCritic(
    name="helpfulness",
    definition="Evaluate how helpful the assistant's response is to the user's query.",
    llm=evaluator_llm,
)
```

You can preview the prompt that will be passed to the LLM (before alignment) by running:


```python
print(helpfulness_critic.get_prompts()["single_turn_aspect_critic_prompt"].instruction)
```
Output
```
Evaluate the Input based on the criterial defined. Use only 'Yes' (1) and 'No' (0) as verdict.
Criteria Definition: Evaluate how helpful the assistant's response is to the user's query.
```

### Defining Alignment Score

Since we are using a binary metric, we will measure the alignment using the F1-score. However, depending on the metric you are aligning, you can modify this function accordingly to use other methods to measure the alignment.


```python
from typing import List
from sklearn.metrics import f1_score


def alignment_score(human_score: List[float], llm_score: List[float]) -> float:
    """
    Computes the alignment between human-annotated binary scores and LLM-generated binary scores
    using the F1-score metric.

    Args:
        human_score (List[int]): Binary labels from human evaluation (0 or 1).
        llm_score (List[int]): Binary labels from LLM predictions (0 or 1).

    Returns:
        float: The F1-score measuring alignment.
    """
    return f1_score(human_score, llm_score)
```

## Prepare your dataset

The `process_hhh_dataset` function prepares data from the [HHH dataset](https://paperswithcode.com/dataset/hhh?utm_source=chatgpt.com) for use in training and aligning of the LLM evaluator. Alternate 0 and 1 scores (1 for helpful, 0 for non-helpful) are assigned to each example, indicating which response is preferred.


```python
import numpy as np
from datasets import load_dataset
from ragas import EvaluationDataset


def process_hhh_dataset(split: str = "helpful", total_count: int = 50):
    dataset = load_dataset(
        "HuggingFaceH4/hhh_alignment", split, split=f"test[:{total_count}]"
    )
    data = []
    expert_scores = []

    for idx, entry in enumerate(dataset):
        # Extract input and target details
        user_input = entry["input"]
        choices = entry["targets"]["choices"]
        labels = entry["targets"]["labels"]

        # Choose target based on whether the index is even or odd
        if idx % 2 == 0:
            target_label = 1
            score = 1
        else:
            target_label = 0
            score = 0

        label_index = labels.index(target_label)

        response = choices[label_index]

        data.append(
            {
                "user_input": user_input,
                "response": response,
            }
        )
        expert_scores.append(score)

    return EvaluationDataset.from_list(data), expert_scores


eval_dataset, expert_scores = process_hhh_dataset()
```

## Run evaluation

With the evaluation dataset and the helpfulness metric defined, you can now run the evaluation:


```python
from ragas import evaluate

results = evaluate(eval_dataset, metrics=[helpfulness_critic])
```
```
Evaluating: 100%|██████████| 50/50 [00:00<?, ?it/s]
```

This initial run highlights the level of misalignment present in LLM-based evaluators, which the subsequent training will address.

Next, benchmark the metric's performance against the expert scores:


```python
human_score = expert_scores
llm_score = results.to_pandas()["helpfulness"].values

initial_score = alignment_score(human_score, llm_score)
initial_score
```
Output
```
0.8076923076923077
```


## Review and Annotate

Now that you have obtained the evaluation results, it’s time to review and annotate them. As discussed in blog [Aligning LLM as judge with human evaluators](https://blog.ragas.io/aligning-llm-as-judge-with-human-evaluators), collecting detailed feedback is essential for bridging the gap between LLM-based and human evaluations. Annotate at least 15–20 examples to capture diverse scenarios where the metric might be misaligned.

Here is a sample annotation for the above example. You can [download](../../_static/annotated_data.json) and use it.

## Training and Alignment

The next step is to train your metric using the annotated examples. This training process leverages a gradient-free prompt optimization approach that adjusts both instructions and few-shot demonstrations based on the annotated feedback.


```python
from ragas.config import InstructionConfig, DemonstrationConfig

demo_config = DemonstrationConfig(embedding=evaluator_embeddings)
inst_config = InstructionConfig(llm=evaluator_llm)

helpfulness_critic.train(
    path="annotated_data.json",
    instruction_config=inst_config,
    demonstration_config=demo_config,
)
```
```
Overall Progress: 100%|██████████| 170/170 [00:00<?, ?it/s]

Few-shot examples [single_turn_aspect_critic_prompt]: 100%|██████████| 16/16 [00:00<?, ?it/s]
```

After training, review the updated instructions that have been optimized for the metric:


```python
print(helpfulness_critic.get_prompts()["single_turn_aspect_critic_prompt"].instruction)
```
Output
```
You are provided with a user input and an assistant/model response. Your task is to evaluate the quality of the response based on how well it addresses the user input, considering all requests and constraints. Assign a score/verdict of 1 if the response is helpful, appropriate, and effective, and 0 if it is not. A good response should be accurate, complete, relevant, and provide a tangible improvement or solution, without omitting key information. Provide a brief explanation for your score/verdict.
```

## Re-evaluate

Now that your metric has been aligned with human feedback, re-run the evaluation on your dataset. This step allows you to benchmark the improvements and quantify how well the alignment process has enhanced the metric’s reliability.


```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings


evaluator_llm = LangchainLLMWrapper(VertexAI(model_name="gemini-pro"))
evaluator_embeddings = LangchainEmbeddingsWrapper(
    VertexAIEmbeddings(model_name="text-embedding-004")
)
```


```python
from ragas import evaluate

results2 = evaluate(eval_dataset, metrics=[helpfulness_critic])
```
```
Evaluating: 100%|██████████| 50/50 [00:00<?, ?it/s]
```

Benchmark the updated results against the expert scores:


```python
human_score = expert_scores
llm_score = results2.to_pandas()["helpfulness"].values

new_score = alignment_score(human_score, llm_score)
new_score
```
Output
```
0.8444444444444444
```

Checkout other tutorials of this series:

- [Ragas with Vertex AI](./vertexai_x_ragas.md): Learn how to use Vertex AI models with Ragas to evaluate your LLM workflows.
- [Model Comparison](./vertexai_model_comparision.md): Compare models provided by VertexAI on RAG-based Q&A task using Ragas metrics.