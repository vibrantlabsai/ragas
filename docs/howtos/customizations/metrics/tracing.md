# Tracing and logging evaluations with Observability tools

Logging and tracing results from LLM are important for any language model-based application. This is a tutorial on how to do tracing with Ragas. Ragas provides `callbacks` functionality which allows you to hook various tracers like LangSmith, wandb, Opik, etc easily.  In this notebook, I will be using LangSmith for tracing.

To set up LangSmith, we need to set some environment variables that it needs. For more information, you can refer to the [docs](https://docs.smith.langchain.com/)

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

Now we have to import the required tracer from LangChain, here we are using `LangChainTracer`, but you can similarly use any tracer supported by LangChain like [WandbTracer](https://python.langchain.com/docs/integrations/providers/wandb_tracing) or [OpikTracer](https://comet.com/docs/opik/tracing/integrations/ragas?utm_source=ragas&utm_medium=docs&utm_campaign=opik&utm_content=tracing_how_to)

```python
# LangSmith
from langchain.callbacks.tracers import LangChainTracer

tracer = LangChainTracer(project_name="callback-experiments")
```

We now pass the tracer to the `callbacks` parameter when calling `evaluate`

```python
from ragas import EvaluationDataset
from datasets import load_dataset
from ragas.metrics import LLMContextRecall

dataset = load_dataset("vibrantlabsai/amnesty_qa", "english_v3")

dataset = EvaluationDataset.load_from_hf(dataset["eval"])
evaluate(dataset, metrics=[LLMContextRecall()],callbacks=[tracer])
```

```text
{'context_precision': 1.0000}
```
<figure markdown="span">
  ![Tracing with LangSmith](../../../_static/imgs/trace-langsmith.png)
  <figcaption>Tracing with LangSmith</figcaption>
</figure>


You can also write your own custom callbacks using LangChain’s `BaseCallbackHandler`, refer [here](https://www.notion.so/Docs-logging-and-tracing-6f21cde9b3cb4d499526f48fd615585d?pvs=21) to read more about it.
