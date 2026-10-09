# Getting Started: Ragas with Vertex AI

This tutorial is part of a three-part series on how to use Vertex AI models with Ragas. This first tutorial is intended to set up the groundwork; the remaining two can be followed in any order. You can navigate to the other tutorials using the links below:

- [Align LLM Metrics](./vertexai_alignment.md): Train and align your LLM evaluators to better match human judgment.
- [Model Comparison](./vertexai_model_comparision.md): Compare models provided by VertexAI on RAG-based Q&A task using Ragas metrics.

Let’s get started!

## Overview

This notebook demonstrates how to get started with Ragas for Gen AI Evaluation using the generative models in Vertex AI Studio.

**Ragas** is a comprehensive evaluation library designed to enhance the assessment of your LLM applications. It offers a suite of tools and metrics that enable developers to systematically evaluate and optimize AI applications.

In this tutorial, we’ll explore:

1. Preparing data for Ragas evaluation
2. An overview of the various types of metrics provided by Ragas

For additional use cases and advanced features, refer to the documentation and How-To's section for evaluation use cases:

- [Ragas Concepts](../../concepts/index.md)
- [Ragas How-Tos](../../howtos/index.md)

## Getting Started

## Install Dependencies


```python
!pip install --upgrade --user --quiet langchain-core langchain-google-vertexai langchain ragas rouge_score
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

## Set up eval using Ragas metrics

**Ragas** is an open‐source library that equips you with a comprehensive suite of tools to evaluate AI applications. Whether you need computation-based metrics, model-based metrics, or even the flexibility to define your own custom metrics, Ragas has you covered. For further details and to explore the full range of supported metrics, visit the [Ragas documentation](../../concepts/metrics/available_metrics/index.md).

In the sections below, you will learn how to leverage the various types of metrics available in Ragas:

- **Custom Metrics:** Define and integrate your own metrics best tailored for your application evaluations.
- **Model-based Metrics:** Evaluations that analyse model outputs against specific criteria using LLM calls, either with or without references.
- **Computation-based Metrics:** Quantitative measures based on mathematical formulas that do not require LLM calls.


### Define evaluator_llm

To use and define model-based metrics, you need to configure both an evaluator LLM and evaluator embeddings.


```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings


evaluator_llm = LangchainLLMWrapper(VertexAI(model_name="gemini-pro"))
evaluator_embeddings = LangchainEmbeddingsWrapper(
    VertexAIEmbeddings(model_name="text-embedding-004")
)
```

### Build your own model-based point-wise metrics using Ragas

Custom or user-defined metrics empower you to tailor evaluation criteria using scoring rules, rubrics, or any custom logic executed through LLM calls to assess responses based on your unique task requirements. For more details, please refer to the [general-purpose metrics documentation](../../concepts/metrics/available_metrics/general_purpose.md).


```python
from ragas import evaluate
from ragas.metrics import AspectCritic, RubricsScore

maliciousness_definition = (
    "Is the response intended to harm, deceive, or exploit users?"
)

aspect_critic = AspectCritic(
    name="maliciousness",
    definition=maliciousness_definition,
    llm=evaluator_llm,
)

# adapeted google's helpfulness_prompt_template
helpfulness_rubrics = {
    "score1_description": "Response is useless/irrelevant, contains inaccurate/deceptive/misleading information, and/or contains harmful/offensive content. The user would feel not at all satisfied with the content in the response.",
    "score2_description": "Response is minimally relevant to the instruction and may provide some vaguely useful information, but it lacks clarity and detail. It might contain minor inaccuracies. The user would feel only slightly satisfied with the content in the response.",
    "score3_description": "Response is relevant to the instruction and provides some useful content, but could be more relevant, well-defined, comprehensive, and/or detailed. The user would feel somewhat satisfied with the content in the response.",
    "score4_description": "Response is very relevant to the instruction, providing clearly defined information that addresses the instruction's core needs.  It may include additional insights that go slightly beyond the immediate instruction.  The user would feel quite satisfied with the content in the response.",
    "score5_description": "Response is useful and very comprehensive with well-defined key details to address the needs in the instruction and usually beyond what explicitly asked. The user would feel very satisfied with the content in the response.",
}

rubrics_score = RubricsScore(
    name="helpfulness", rubrics=helpfulness_rubrics, llm=evaluator_llm
)
```

### Ragas model-based metrics

Model-based metrics leverage pre-trained language models to assess generated text by comparing responses against specific criteria, offering nuanced, context-aware evaluations that emulate human judgment. These metrics are computed via LLM calls. For more details, please see the [model-based metrics documentation](../../concepts/metrics/available_metrics/index.md).


```python
from ragas import evaluate
from ragas.metrics import ContextPrecision, Faithfulness

context_precision = ContextPrecision(llm=evaluator_llm)
faithfulness = Faithfulness(llm=evaluator_llm)
```

### Ragas computation-based metrics

These metrics employ established string matching, n-gram, and statistical methods to quantify text similarity and quality computed entirely mathematically without LLM calls. For more details, please visit the [computation-based metrics documentation](../../concepts/metrics/available_metrics/traditional.md).


```python
from ragas.metrics import RougeScore

rouge_score = RougeScore()
```

## Prepare your dataset

To perform evaluations using Ragas metrics, you need to convert your data into an `EvaluationDataset`, a data type in Ragas. You can read more about it [here](../../concepts/components/eval_dataset.md).

For example, consider the following sample data:


```python
# questions or query from user
user_inputs = [
    "Which part of the brain does short-term memory seem to rely on?",
    "What provided the Roman senate with exuberance?",
    "What area did the Hasan-jalalians command?",
]

# retrieved data used in answer generation
retrieved_contexts = [
    [
        "Short-term memory is supported by transient patterns of neuronal communication, dependent on regions of the frontal lobe (especially dorsolateral prefrontal cortex) and the parietal lobe. Long-term memory, on the other hand, is maintained by more stable and permanent changes in neural connections widely spread throughout the brain. The hippocampus is essential (for learning new information) to the consolidation of information from short-term to long-term memory, although it does not seem to store information itself. Without the hippocampus, new memories are unable to be stored into long-term memory, as learned from patient Henry Molaison after removal of both his hippocampi, and there will be a very short attention span. Furthermore, it may be involved in changing neural connections for a period of three months or more after the initial learning."
    ],
    [
        "In 62 BC, Pompey returned victorious from Asia. The Senate, elated by its successes against Catiline, refused to ratify the arrangements that Pompey had made. Pompey, in effect, became powerless. Thus, when Julius Caesar returned from a governorship in Spain in 61 BC, he found it easy to make an arrangement with Pompey. Caesar and Pompey, along with Crassus, established a private agreement, now known as the First Triumvirate. Under the agreement, Pompey's arrangements would be ratified. Caesar would be elected consul in 59 BC, and would then serve as governor of Gaul for five years. Crassus was promised a future consulship."
    ],
    [
        "The Seljuk Empire soon started to collapse. In the early 12th century, Armenian princes of the Zakarid noble family drove out the Seljuk Turks and established a semi-independent Armenian principality in Northern and Eastern Armenia, known as Zakarid Armenia, which lasted under the patronage of the Georgian Kingdom. The noble family of Orbelians shared control with the Zakarids in various parts of the country, especially in Syunik and Vayots Dzor, while the Armenian family of Hasan-Jalalians controlled provinces of Artsakh and Utik as the Kingdom of Artsakh."
    ],
]

# answers generated by the rag
responses = [
    "frontal lobe and the parietal lobe",
    "The Roman Senate was filled with exuberance due to successes against Catiline.",
    "The Hasan-Jalalians commanded the area of Syunik and Vayots Dzor.",
]

# expected responses or ground truth
references = [
    "frontal lobe and the parietal lobe",
    "Due to successes against Catiline.",
    "The Hasan-Jalalians commanded the area of Artsakh and Utik.",
]
```

Convert these into Ragas' EvaluationDataset:


```python
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

n = len(user_inputs)
samples = []


for i in range(n):
    sample = SingleTurnSample(
        user_input=user_inputs[i],
        retrieved_contexts=retrieved_contexts[i],
        response=responses[i],
        reference=references[i],
    )
    samples.append(sample)


ragas_eval_dataset = EvaluationDataset(samples=samples)
ragas_eval_dataset.to_pandas()
```
Output
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border=1>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>response</th>
      <th>reference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Which part of the brain does short-term memory...</td>
      <td>[Short-term memory is supported by transient p...</td>
      <td>frontal lobe and the parietal lobe</td>
      <td>frontal lobe and the parietal lobe</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What provided the Roman senate with exuberance?</td>
      <td>[In 62 BC, Pompey returned victorious from Asi...</td>
      <td>The Roman Senate was filled with exuberance du...</td>
      <td>Due to successes against Catiline.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What area did the Hasan-jalalians command?</td>
      <td>[The Seljuk Empire soon started to collapse. I...</td>
      <td>The Hasan-Jalalians commanded the area of Syun...</td>
      <td>The Hasan-Jalalians commanded the area of Arts...</td>
    </tr>
  </tbody>
</table>
</div>

## Run evaluation

With the evaluation dataset and desired metrics defined, you can run evaluations by passing them into Ragas' `evaluate` function:

```python
from ragas import evaluate

ragas_metrics = [
    aspect_critic,
    context_precision,
    faithfulness,
    rouge_score,
    rubrics_score,
]

result = evaluate(metrics=ragas_metrics, dataset=ragas_eval_dataset)
result
```
```
Evaluating: 100%|██████████| 15/15 [00:00<?, ?it/s]
```

View the detailed scores for each row in your dataset:


```python
result.to_pandas()
```


Output
<div>
<table border=1>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>response</th>
      <th>reference</th>
      <th>maliciousness</th>
      <th>context_precision</th>
      <th>faithfulness</th>
      <th>rouge_score(mode=fmeasure)</th>
      <th>helpfulness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Which part of the brain does short-term memory...</td>
      <td>[Short-term memory is supported by transient p...</td>
      <td>frontal lobe and the parietal lobe</td>
      <td>frontal lobe and the parietal lobe</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What provided the Roman senate with exuberance?</td>
      <td>[In 62 BC, Pompey returned victorious from Asi...</td>
      <td>The Roman Senate was filled with exuberance du...</td>
      <td>Due to successes against Catiline.</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.588235</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What area did the Hasan-jalalians command?</td>
      <td>[The Seljuk Empire soon started to collapse. I...</td>
      <td>The Hasan-Jalalians commanded the area of Syun...</td>
      <td>The Hasan-Jalalians commanded the area of Arts...</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.761905</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

Checkout other tutorials of this series:

- [Align LLM Metrics](./vertexai_alignment.md): Train and align your LLM evaluators to better match human judgment.
- [Model Comparison](./vertexai_model_comparision.md): Compare models provided by VertexAI on RAG-based Q&A task using Ragas metrics.