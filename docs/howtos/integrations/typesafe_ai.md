# TypeSafe AI (Jev) Integration Guide

This guide covers setting up and using TypeSafe AI's **Jev** model with Ragas for high-speed, calibrated evaluation.

## Overview

TypeSafe AI's **Jev** is a frontier non-autoregressive "System One" decision model. Rather than generating autoregressive text tokens, Jev directly outputs calibrated probabilities for typed decisions:
- **`Noul`**: Calibrated boolean/binary probabilities ($P \in [0, 1]$) for NLI verdicts, claim verification, and factual attribution.
- **`Score`**: Positions along an ordered rubric (e.g., 1–5 or 0–10 scale).
- **`Choice`**: Categorical classification over discrete options.

Jev returns calibrated judgements in **70–500ms** at a fraction of the cost of standard LLMs, making it the ideal engine for regression testing in automated CI/CD evaluation pipelines.

---

## Setup

### Prerequisites

- TypeSafe AI API key ([typesafe.ai](https://typesafe.ai))
- Python 3.9+
- Ragas installed

### Installation

Install the TypeSafe SDK:

```bash
pip install typesafe-sdk
```

Set your API key:

```bash
export TYPESAFE_API_KEY="your-typesafe-api-key"
```

---

## Configuration

### Option 1: Using `llm_factory` (Recommended)

Ragas provides unified LLM instantiation with automatic adapter routing:

```python
import os
from typesafe_sdk import TypeSafeClient
from ragas.llms import llm_factory

client = TypeSafeClient(api_key=os.environ.get("TYPESAFE_API_KEY"))

# Create LLM - adapter is auto-detected for typesafe provider
llm = llm_factory(
    "jev-latest",
    provider="typesafe",
    client=client
)
```

Alternatively, if `TYPESAFE_API_KEY` is set in your environment:

```python
from ragas.llms import llm_factory

llm = llm_factory("jev-latest", provider="typesafe")
```

### Option 2: Direct Initialization via `TypeSafeStructuredLLM`

```python
from ragas.llms import TypeSafeStructuredLLM

llm = TypeSafeStructuredLLM(
    model="jev-latest",
    api_key="your-typesafe-api-key"
)
```

---

## Hybrid Evaluation (Two-Stage Pipeline)

Some metrics (such as `Faithfulness`) require a two-stage evaluation:
1. **Stage 1 (Text Extraction)**: Decompose a response into atomic claims or statements.
2. **Stage 2 (Decision Judging)**: Evaluate whether each statement is supported by the retrieved context.

Because Jev specializes in calibrated decision probabilities rather than free-form text synthesis, you can pair Jev with a lightweight generative LLM using the `generator_llm` parameter:

```python
import os
from openai import OpenAI
from typesafe_sdk import TypeSafeClient
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

# 1. Standard fast generator for sentence extraction
generator = llm_factory("gpt-4o-mini", client=OpenAI())

# 2. TypeSafe Jev for high-speed NLI verdicts
jev_client = TypeSafeClient(api_key=os.environ.get("TYPESAFE_API_KEY"))
llm = llm_factory(
    "jev-latest",
    provider="typesafe",
    client=jev_client,
    generator_llm=generator
)

# 3. Create metric instance
faithfulness = Faithfulness(llm=llm)

# 4. Run evaluation
result = await faithfulness.ascore(
    user_input="Where was Albert Einstein born?",
    response="Einstein was born in Ulm, Germany on 14 March 1879.",
    retrieved_contexts=["Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire..."]
)

print(f"Faithfulness Score: {result.value}")
```

---

## Evaluating Decision Metrics

For metrics that evaluate context directly without open-ended sentence generation (such as `ContextPrecision`), Jev handles the entire evaluation natively:

```python
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision
from typesafe_sdk import TypeSafeClient

llm = llm_factory("jev-latest", provider="typesafe", client=TypeSafeClient())

context_precision = ContextPrecision(llm=llm)

result = await context_precision.ascore(
    user_input="What is the capital of France?",
    response="Paris",
    retrieved_contexts=["Paris is the capital and most populous city of France."]
)

print(f"Context Precision Score: {result.value}")
```

---

## Key Benefits

| Feature | Standard LLM-as-a-Judge | TypeSafe Jev |
| :--- | :--- | :--- |
| **Latency** | 2,000–5,000 ms per check | **70–500 ms** |
| **Scoring Consistency** | Generative variance / temperature noise | **Calibrated probabilities ($P \in [0, 1]$)** |
| **Cost** | Autoregressive output tokens | **Input tokens only** (zero output token charge) |
| **Hallucination** | Risk of output formatting errors | **Structured schema guarantees** |
