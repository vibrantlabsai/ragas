# Customize Timeouts and Rate Limits

Configure timeouts and retries directly on your LLM client when using the collections API with `llm_factory`.

## OpenAI Client Configuration

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

# Configure timeout and retries on the client
client = AsyncOpenAI(
    timeout=60.0,        # 60 second timeout
    max_retries=5,       # Retry up to 5 times on failures
)

llm = llm_factory("gpt-4o-mini", client=client)

# Use with metrics
scorer = Faithfulness(llm=llm)
result = scorer.score(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=[
        "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
    ]
)
```

### Available Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timeout` | 600.0 | Request timeout in seconds |
| `max_retries` | 2 | Number of retry attempts for failed requests |

### Fine-Grained Timeout Control

For more control over different timeout types:

```python
import httpx
from openai import AsyncOpenAI

client = AsyncOpenAI(
    timeout=httpx.Timeout(
        60.0,           # Total timeout
        connect=5.0,    # Connection timeout
        read=30.0,      # Read timeout
        write=10.0,     # Write timeout
    ),
    max_retries=3,
)
```

!!! tip "Provider Documentation"
    Each LLM provider has its own client configuration options. Refer to your provider's SDK documentation:
    
    - [OpenAI Python SDK](https://github.com/openai/openai-python)
    - [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)


## Legacy Metrics API

The following examples use the legacy metrics API pattern with `RunConfig`. For new projects, we recommend using the collections-based API with client-level configuration as shown above.

!!! warning "Deprecation Timeline"
    This API will be deprecated in version 0.4 and removed in version 1.0. Please migrate to the collections-based API.

### RunConfig Parameters

```python
from ragas.run_config import RunConfig

run_config = RunConfig(
    timeout=180,        # Max seconds per operation (default: 180)
    max_retries=10,     # Retry attempts (default: 10)
    max_wait=60,        # Max seconds between retries (default: 60)
    max_workers=16,     # Concurrent workers (default: 16)
    log_tenacity=False, # Log retry attempts (default: False)
    seed=42,            # Random seed (default: 42)
)
```

### Usage with Evaluate

```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness
from ragas.run_config import RunConfig

# Legacy LLM setup
llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

# Configure run settings
run_config = RunConfig(max_workers=64, timeout=60)

# Use with evaluate
results = evaluate(
    dataset=eval_dataset,
    metrics=[Faithfulness(llm=llm)],
    run_config=run_config,
)
```
