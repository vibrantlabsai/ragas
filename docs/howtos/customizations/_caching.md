# Caching in Ragas

You can use caching to speed up your evaluations and testset generation by avoiding redundant computations. We use Exact Match Caching to cache the responses from the LLM and Embedding models.

You can use the [DiskCacheBackend][ragas.cache.DiskCacheBackend] which uses a local disk cache to store the cached responses. You can also implement your own custom cacher by implementing the [CacheInterface][ragas.cache.CacheInterface].


## Using DefaultCacher

Let's see how you can use the [DiskCacheBackend][ragas.cache.DiskCacheBackend] LLM and Embedding models.



```python
from ragas.cache import DiskCacheBackend

cacher = DiskCacheBackend()

# check if the cache is empty and clear it
print(len(cacher.cache))
cacher.cache.clear()
print(len(cacher.cache))
```




    DiskCacheBackend(cache_dir=.cache)



Create an LLM and Embedding model with the cacher, here I'm using the `ChatOpenAI` from [langchain-openai](https://github.com/langchain-ai/langchain-openai) as an example.



```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

cached_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"), cache=cacher)
```


```python
# if you want to see the cache in action, set the logging level to debug
import logging
from ragas.utils import set_logging_level

set_logging_level("ragas.cache", logging.DEBUG)
```

Now let's run a simple evaluation.


```python
from ragas import evaluate
from ragas import EvaluationDataset

from ragas.metrics import FactualCorrectness, AspectCritic
from datasets import load_dataset

# Define Answer Correctness with AspectCritic
answer_correctness = AspectCritic(
    name="answer_correctness",
    definition="Is the answer correct? Does it match the reference answer?",
    llm=cached_llm,
)

metrics = [answer_correctness, FactualCorrectness(llm=cached_llm)]

# load the dataset
dataset = load_dataset(
    "vibrantlabsai/amnesty_qa", "english_v3", trust_remote_code=True
)
eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])

# evaluate the dataset
results = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
)

results
```

This took almost 2mins to run in our local machine. Now let's run it again to see the cache in action.


```python
results = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
)

results
```

Runs almost instantaneously.

You can also use this with testset generation also by replacing the `generator_llm` with a cached version of it. Refer to the [testset generation](../../getstarted/rag_testset_generation.md) section for more details.
