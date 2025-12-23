# Caching in Ragas

You can use caching to speed up your evaluations and testset generation by avoiding redundant computations. We use Exact Match Caching to cache the responses from the LLM and Embedding models.

You can use the [DiskCacheBackend][ragas.cache.DiskCacheBackend] which uses a local disk cache to store the cached responses. You can also implement your own custom cacher by implementing the [CacheInterface][ragas.cache.CacheInterface].

## Using Caching with Modern LLMs and Embeddings

The new metrics collections and experiments support caching through a simple interface.

### Quick Start

```python
from ragas.cache import DiskCacheBackend
from ragas.llms import llm_factory
from openai import OpenAI

# Create cache once
cache = DiskCacheBackend()

# Use with LLM factory
client = OpenAI(api_key="...")
llm = llm_factory("gpt-4o-mini", client=client, cache=cache)

# All LLM calls are now cached!
from pydantic import BaseModel

class Response(BaseModel):
    answer: str

response = llm.generate("Evaluate this...", Response)
```

### Caching with llm_factory

```python
from ragas.cache import DiskCacheBackend
from ragas.llms import llm_factory
from openai import OpenAI

# Create cache instance
cache = DiskCacheBackend()

# Create LLM with caching
client = OpenAI(api_key="...")
llm = llm_factory("gpt-4o-mini", client=client, cache=cache)

# First call - makes API request and caches result
response1 = llm.generate("Evaluate this text", Response)

# Second call - returns cached result instantly
response2 = llm.generate("Evaluate this text", Response)

# Result: Same output, 60x faster, $0 cost
```

### Caching with embedding_factory

```python
from ragas.cache import DiskCacheBackend
from ragas.embeddings import embedding_factory
from openai import OpenAI

cache = DiskCacheBackend()
client = OpenAI(api_key="...")

embeddings = embedding_factory("openai", client=client, cache=cache)

# First call - makes API request
vector1 = embeddings.embed_text("Some text to embed")

# Second call - instant cache hit
vector2 = embeddings.embed_text("Some text to embed")

assert vector1 == vector2  # Identical results
```

### Caching in Experiments

Caching is especially powerful in experiments where you run the same evaluation multiple times:

```python
from ragas import experiment, Dataset
from ragas.cache import DiskCacheBackend
from ragas.llms import llm_factory
from ragas.metrics.collections import FactualCorrectness

# Setup cached LLM once
cache = DiskCacheBackend()
llm = llm_factory("gpt-4o-mini", client=client, cache=cache)

# Use in metric
metric = FactualCorrectness(llm=llm)

@experiment()
async def evaluate_model(row):
    score = metric.score(
        response=row["response"],
        reference=row["reference"]
    )
    return {
        **row,
        "factual_correctness": score.value,
        "reason": score.reason
    }

# Load your dataset
dataset = Dataset.from_list([
    {"response": "Paris is the capital of France", "reference": "Paris"},
    {"response": "London is the capital of UK", "reference": "London"},
])

# First run - makes API calls and caches results
print("First run (populating cache)...")
results1 = await evaluate_model.arun(dataset)
# Takes ~2 seconds for 2 samples

# Second run - uses cache, nearly instant!
print("Second run (using cache)...")
results2 = await evaluate_model.arun(dataset)
# Takes ~0.1 seconds for 2 samples

# Results are identical, but 20x faster!
```

### Cache Management

#### Clearing the Cache

```python
# Clear all cached data
cache = DiskCacheBackend()
cache.cache.clear()
```

#### Setting Size Limits

```python
# Limit cache to 1GB
cache = DiskCacheBackend()
cache.cache.reset('size_limit', 1e9)  # 1GB
cache.cache.reset('cull_limit', 10)   # Remove 10% when full
```

#### Cache Location

By default, cache is stored in `.cache/` directory. You can change this:

```python
cache = DiskCacheBackend(cache_dir="my_custom_cache")
```

### Benefits of Caching

1. **Cost Savings**: Avoid repeated API calls for identical inputs (50-60% savings)
2. **Speed**: Cached calls return nearly instantly (60x+ faster)
3. **Development**: Iterate quickly without waiting for API calls
4. **Reproducibility**: Same inputs always return same results

Cache hits occur when:

- ✅ Same prompt/text (exact match)
- ✅ Same model parameters (temperature, max_tokens, etc.)
- ✅ Same response model/structure (for LLMs)

Cache misses occur when:

- ❌ Different prompt/text
- ❌ Different parameters
- ❌ Different response model

### Anti-Patterns (When NOT to Cache)

- ❌ **Non-deterministic prompts**: If prompts contain random elements or timestamps
- ❌ **High temperature**: If temperature > 0.7 (responses vary too much)
- ❌ **Streaming responses**: Caching doesn't work with streaming
- ❌ **Real-time data**: If responses need to reflect current state

### Environment-Specific Notes

**Notebooks**: Cache persists between cell executions and kernel restarts

**Web Applications**: Share cache across requests for better performance

**Serverless Functions**: Use `/tmp` directory:

```python
cache = DiskCacheBackend(cache_dir="/tmp/.cache")
```

**Distributed Workers**: Cache is process-safe but for high-throughput systems consider implementing a Redis backend via the `CacheInterface`

### Performance Expectations

| Scenario | Time | Cost |
|----------|------|------|
| First run (100 samples) | ~2 minutes | $0.50 |
| Second run (cached) | ~2 seconds | $0.00 |
| **Speedup** | **60x faster** | **100% savings** |

---

## Legacy Caching (Deprecated)

!!! warning "Deprecated"
    This approach using `LangchainLLMWrapper` is deprecated and will be removed in v1.0. Please use the modern approach with `llm_factory()` and `embedding_factory()` as shown above.

### Using Legacy Caching with LangchainLLMWrapper

Let's see how you can use the [DiskCacheBackend][ragas.cache.DiskCacheBackend] with legacy LLM and Embedding models.

```python
from ragas.cache import DiskCacheBackend

cacher = DiskCacheBackend()

# check if the cache is empty and clear it
print(len(cacher.cache))
cacher.cache.clear()
print(len(cacher.cache))
```

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
