# Zeno
## Visualizing Ragas Results with Zeno

You can use the [Zeno](https://zenoml.com) evaluation platform to easily visualize and explore the results of your Ragas evaluation.

> Check out what the result of this tutorial looks like [here](https://hub.zenoml.com/project/b35c83b8-0b22-4b9c-aedb-80964011d7a7/ragas%20FICA%20eval)

First, install the `zeno-client` package:

```bash
pip install zeno-client
```

Next, create an account at [hub.zenoml.com](https://hub.zenoml.com) and generate an API key on your [account page](https://hub.zenoml.com/account).

We can now pick up the evaluation where we left off at the [Getting Started](../../getstarted/evaluation.md) guide:


```python
import os

import pandas as pd
from datasets import load_dataset
from zeno_client import ZenoClient, ZenoMetric

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
```


```python
# Set API keys
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["ZENO_API_KEY"] = "your-zeno-api-key"
```


```python
fiqa_eval = load_dataset("vibrantlabsai/fiqa", "ragas_eval")
result = evaluate(
    fiqa_eval["baseline"],
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)

df = result.to_pandas()
df.head()
```

We can now take the `df` with our data and results and upload it to Zeno.

We first create a project with a custom RAG view specification and the metric columns we want to do evaluation across:


```python
client = ZenoClient(os.environ["ZENO_API_KEY"])

project = client.create_project(
    name="Ragas FICA eval",
    description="Evaluation of RAG model using Ragas on the FICA dataset",
    view={
        "data": {
            "type": "vstack",
            "keys": {
                "question": {"type": "markdown"},
                "texts": {
                    "type": "list",
                    "elements": {"type": "markdown"},
                    "border": True,
                    "pad": True,
                },
            },
        },
        "label": {
            "type": "markdown",
        },
        "output": {
            "type": "vstack",
            "keys": {
                "answer": {"type": "markdown"},
                "ground_truth": {
                    "type": "list",
                    "elements": {"type": "markdown"},
                    "border": True,
                    "pad": True,
                },
            },
        },
        "size": "large",
    },
    metrics=[
        ZenoMetric(
            name="context_precision", type="mean", columns=["context_precision"]
        ),
        ZenoMetric(name="faithfulness", type="mean", columns=["faithfulness"]),
        ZenoMetric(name="answer_relevancy", type="mean", columns=["answer_relevancy"]),
        ZenoMetric(name="context_recall", type="mean", columns=["context_recall"]),
    ],
)
```

Next, we upload the base dataset with the questions and ground truths:


```python
data_df = pd.DataFrame(
    {
        "data": df.apply(
            lambda x: {"question": x["question"], "texts": list(x["contexts"])}, axis=1
        ),
        "label": df["ground_truth"].apply(lambda x: "\n".join(x)),
    }
)
data_df["id"] = data_df.index

project.upload_dataset(
    data_df, id_column="id", data_column="data", label_column="label"
)
```

Lastly, we upload the RAG outputs and Ragas metrics. 

You can run this for any number of models when doing comparison and iteration:


```python
output_df = df[
    [
        "context_precision",
        "faithfulness",
        "answer_relevancy",
        "context_recall",
    ]
].copy()

output_df["output"] = df.apply(
    lambda x: {"answer": x["answer"], "ground_truth": list(x["ground_truth"])}, axis=1
)
output_df["id"] = output_df.index

project.upload_system(
    output_df, name="Base System", id_column="id", output_column="output"
)
```

Reach out to the Zeno team on [Discord](https://discord.gg/km62pDKAkE) or at [hello@zenoml.com](mailto:hello@zenoml.com) if you have any questions!
