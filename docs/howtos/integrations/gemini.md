# Google Gemini Integration Guide

This guide covers setting up and using Google's Gemini models with Ragas for evaluation.

## Overview

Ragas supports Google Gemini models with automatic adapter selection. The framework intelligently routes Gemini requests through the LiteLLM adapter, which provides seamless compatibility with Gemini's API.

## Setup

### Prerequisites

- Google API Key with Gemini API access
- Python 3.8+
- Ragas installed

### Installation

Install required dependencies:

```bash
pip install ragas google-generativeai litellm
```

Or with the Ragas extras:

```bash
pip install "ragas[gemini]"
```

## Configuration

### Option 1: Using Google's Official Library (Recommended)

Google's official generativeai library is the simplest and most direct approach:

```python
import os
import google.generativeai as genai
from ragas.llms import llm_factory

# Configure with your API key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Create client
client = genai.GenerativeModel("gemini-2.0-flash")

# Create LLM - adapter is auto-detected for google provider
llm = llm_factory(
    "gemini-2.0-flash",
    provider="google",
    client=client
)
```

### Option 2: Using LiteLLM Proxy (Advanced)

For advanced use cases where you need LiteLLM's proxy capabilities, set up the LiteLLM proxy server first, then use:

```python
import os
from openai import OpenAI
from ragas.llms import llm_factory

# Requires running: litellm --model gemini-2.0-flash
client = OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"  # LiteLLM proxy endpoint
)

# Create LLM with explicit adapter selection
llm = llm_factory("gemini-2.0-flash", client=client, adapter="litellm")
```

## Supported Models

Ragas works with all Gemini models:

- **Latest**: `gemini-2.0-flash` (recommended)
- **1.5 Series**: `gemini-1.5-pro`, `gemini-1.5-flash`
- **1.0 Series**: `gemini-1.0-pro`

For the latest models and pricing, see [Google AI Studio](https://aistudio.google.com/apikey).

## Embeddings Configuration

Ragas metrics fall into two categories:

1. **LLM-only metrics** (don't require embeddings):
   - ContextPrecision
   - ContextRecall
   - Faithfulness
   - AspectCritic

2. **Embedding-dependent metrics** (require embeddings):
   - AnswerCorrectness
   - AnswerRelevancy
   - AnswerSimilarity
   - SemanticSimilarity
   - ContextEntityRecall

### Automatic Provider Matching

When using Ragas with Gemini, the embedding provider is **automatically matched** to your LLM provider. If you provide a Gemini LLM, Ragas will default to using Google embeddings. **No OpenAI API key is needed.**

### Option 1: Default Embeddings (Recommended)

Let Ragas automatically select the right embeddings based on your LLM:

```python
import os
from datasets import Dataset
import google.generativeai as genai
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    Faithfulness
)

# Initialize Gemini client
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
client = genai.GenerativeModel("gemini-2.0-flash")
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)

# Create sample evaluation data
data = {
    "question": ["What is the capital of France?"],
    "answer": ["Paris is the capital of France."],
    "contexts": [["France is a country in Western Europe. Paris is its capital."]],
    "ground_truth": ["Paris"]
}

dataset = Dataset.from_dict(data)

# Define metrics - embeddings are auto-configured for Google
metrics = [
    ContextPrecision(llm=llm),
    ContextRecall(llm=llm),
    Faithfulness(llm=llm),
    AnswerCorrectness(llm=llm)  # Uses Google embeddings automatically
]

# Run evaluation
results = evaluate(dataset, metrics=metrics)
print(results)
```

### Option 2: Explicit Embeddings

For explicit control over embeddings, you can create them separately. Google embeddings work with multiple configuration options:

```python
import os
import google.generativeai as genai
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.embeddings.base import embedding_factory
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import AnswerCorrectness, ContextPrecision, ContextRecall, Faithfulness

# Configure Google AI
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Initialize Gemini LLM
client = genai.GenerativeModel("gemini-2.0-flash")
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)

# Initialize Google embeddings (multiple options):

# Option A: Simplest - auto-import (recommended)
embeddings = embedding_factory("google", model="text-embedding-004")

# Option B: From genai module directly
embeddings = GoogleEmbeddings(client=genai, model="text-embedding-004")

# Option C: No client (auto-imports genai)
embeddings = GoogleEmbeddings(model="text-embedding-004")

# Create sample evaluation data
data = {
    "question": ["What is the capital of France?"],
    "answer": ["Paris is the capital of France."],
    "contexts": [["France is a country in Western Europe. Paris is its capital."]],
    "ground_truth": ["Paris"]
}

dataset = Dataset.from_dict(data)

# Define metrics with explicit embeddings
metrics = [
    ContextPrecision(llm=llm),
    ContextRecall(llm=llm),
    Faithfulness(llm=llm),
    AnswerCorrectness(llm=llm, embeddings=embeddings)
]

# Run evaluation
results = evaluate(dataset, metrics=metrics)
print(results)
```

## Example: Complete Evaluation

Here's a complete example evaluating a RAG application with Gemini (using automatic embedding provider matching):

```python
import os
from datasets import Dataset
import google.generativeai as genai
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    Faithfulness
)

# Initialize Gemini client
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
client = genai.GenerativeModel("gemini-2.0-flash")
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)

# Create sample evaluation data
data = {
    "question": ["What is the capital of France?"],
    "answer": ["Paris is the capital of France."],
    "contexts": [["France is a country in Western Europe. Paris is its capital."]],
    "ground_truth": ["Paris"]
}

dataset = Dataset.from_dict(data)

# Define metrics - embeddings automatically use Google provider
metrics = [
    ContextPrecision(llm=llm),
    ContextRecall(llm=llm),
    Faithfulness(llm=llm),
    AnswerCorrectness(llm=llm)
]

# Run evaluation
results = evaluate(dataset, metrics=metrics)
print(results)
```

## Performance Considerations

### Model Selection

- **gemini-2.0-flash**: Best for speed and efficiency
- **gemini-1.5-pro**: Better reasoning for complex evaluations
- **gemini-1.5-flash**: Good balance of speed and cost

### Cost Optimization

Gemini models are cost-effective. For large-scale evaluations:

1. Use `gemini-2.0-flash` for most metrics
2. Consider batch processing for multiple evaluations
3. Cache prompts when possible (Gemini supports prompt caching)

### Async Support

For high-throughput evaluations, use async operations with google-generativeai:

```python
import google.generativeai as genai
from ragas.llms import llm_factory

# Configure and create client (same as sync)
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
client = genai.GenerativeModel("gemini-2.0-flash")
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)

# Use in async evaluation
# response = await llm.agenerate(prompt, ResponseModel)
```

## Adapter Selection

Ragas automatically selects the appropriate adapter based on your setup:

```python
# Auto-detection happens automatically
# For Gemini: uses LiteLLM adapter
# For other providers: uses Instructor adapter

# Explicit selection (if needed)
llm = llm_factory(
    "gemini-2.0-flash",
    client=client,
    adapter="litellm"  # Explicit adapter selection
)

# Check auto-detected adapter
from ragas.llms.adapters import auto_detect_adapter
adapter_name = auto_detect_adapter(client, "google")
print(f"Using adapter: {adapter_name}")  # Output: Using adapter: litellm
```

## Troubleshooting

### API Key Issues

```python
# Make sure your API key is set
import os
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set")
```

### Rate Limits

Gemini has rate limits. For production use, the LLM adapter handles retries and timeouts automatically. If you need fine-grained control, ensure your client is properly configured with appropriate timeouts at the HTTP client level.

### Model Availability

If a model isn't available:

1. Check your region/quota in [Google Cloud Console](https://console.cloud.google.com)
2. Try a different model from the supported list
3. Verify your API key has access to the Generative AI API

## Migration from Other Providers

### From OpenAI

```python
# Before: OpenAI-only
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
llm = llm_factory("gpt-4o", client=client)

# After: Gemini with similar code pattern
import google.generativeai as genai
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
client = genai.GenerativeModel("gemini-2.0-flash")
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)
```

### From Anthropic

```python
# Before: Anthropic
from anthropic import Anthropic
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
llm = llm_factory("claude-3-sonnet", provider="anthropic", client=client)

# After: Gemini
import google.generativeai as genai
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
client = genai.GenerativeModel("gemini-2.0-flash")
llm = llm_factory("gemini-2.0-flash", provider="google", client=client)
```

## Using with Metrics Collections (Modern Approach)

For the modern metrics collections API, you need to explicitly create both LLM and embeddings:

```python
import os
import google.generativeai as genai
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerCorrectness, ContextPrecision

# Configure Google AI
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Create LLM
llm_client = genai.GenerativeModel("gemini-2.0-flash")
llm = llm_factory("gemini-2.0-flash", provider="google", client=llm_client)

# Create embeddings - multiple options work:
# Option 1: Auto-import (simplest)
embeddings = embedding_factory("google", model="text-embedding-004")

# Option 2: From LLM client (now works with the fix!)
embeddings = embedding_factory("google", client=llm.client, model="text-embedding-004")

# Create metrics with explicit LLM and embeddings
metrics = [
    ContextPrecision(llm=llm),  # LLM-only metric
    AnswerCorrectness(llm=llm, embeddings=embeddings),  # Needs both
]

# Use metrics with your evaluation workflow
result = await metrics[1].ascore(
    user_input="What is the capital of France?",
    response="Paris",
    reference="Paris is the capital of France."
)
```

**Key difference from legacy approach:**
- Legacy `evaluate()`: Auto-creates embeddings from LLM provider
- Modern collections: You explicitly pass embeddings to each metric

This gives you more control and works seamlessly with Gemini!

## Supported Metrics

All Ragas metrics work with Gemini:

- Answer Correctness
- Answer Relevancy
- Answer Similarity
- Aspect Critique
- Context Precision
- Context Recall
- Context Entities Recall
- Faithfulness
- NLI Eval
- Response Relevancy

See [Metrics Reference](../../concepts/metrics/index.md) for details.

## Advanced: Custom Model Parameters

Pass custom parameters to Gemini:

```python
llm = llm_factory(
    "gemini-2.0-flash",
    client=client,
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9,
    top_k=40,
)
```

## Resources

- [Google Gemini API Docs](https://ai.google.dev/gemini-2/docs)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Ragas Metrics Documentation](../../concepts/metrics/index.md)
- [Ragas LLM Factory Guide](../llm-factory.md)
