# Langsmith
## Dataset and Tracing Visualisation

[Langsmith](https://docs.smith.langchain.com/) in a platform for building production-grade LLM applications from the langchain team. It helps you with tracing, debugging and evaluting LLM applications.

The langsmith + ragas integrations offer 2 features
1. View the traces of ragas `evaluator` 
2. Use ragas metrics in langchain evaluation - (soon)


## Tracing ragas metrics

since ragas uses langchain under the hood all you have to do is setup langsmith and your traces will be logged.

to setup langsmith make sure the following env-vars are set (you can read more in the [langsmith docs](https://docs.smith.langchain.com/#quick-start)

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

Once langsmith is setup, just run the evaluations as your normally would


```python
from datasets import load_dataset

from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

fiqa_eval = load_dataset("vibrantlabsai/fiqa", "ragas_eval")

result = evaluate(
    fiqa_eval["baseline"].select(range(3)),
    metrics=[context_precision, faithfulness, answer_relevancy],
)

result
```

    Found cached dataset fiqa (/home/jjmachan/.cache/huggingface/datasets/vibrantlabs___fiqa/ragas_eval/1.0.0/3dc7b639f5b4b16509a3299a2ceb78bf5fe98ee6b5fee25e7d5e4d290c88efb8)



      0%|          | 0/1 [00:00<?, ?it/s]


    evaluating with [context_precision]


    100%|█████████████████████████████████████████████████████████████| 1/1 [00:23<00:00, 23.21s/it]


    evaluating with [faithfulness]


    100%|█████████████████████████████████████████████████████████████| 1/1 [00:36<00:00, 36.94s/it]


    evaluating with [answer_relevancy]


    100%|█████████████████████████████████████████████████████████████| 1/1 [00:10<00:00, 10.58s/it]





    {'context_precision': 0.5976, 'faithfulness': 0.8889, 'answer_relevancy': 0.9300}



Voila! Now you can head over to your project and see the traces
