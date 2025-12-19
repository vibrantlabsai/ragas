# Tokenizers

Ragas supports multiple tokenizer implementations for text splitting during knowledge graph operations and test data generation.

## Overview

When extracting properties from knowledge graph nodes, text is split into chunks based on token limits. By default, Ragas uses tiktoken (OpenAI's tokenizer), but you can also use HuggingFace tokenizers for better compatibility with open-source models.

## Available Tokenizers

### TiktokenWrapper

Wrapper for OpenAI's tiktoken tokenizers. This is the default tokenizer.

```python
from ragas import TiktokenWrapper

# Using default encoding (o200k_base)
tokenizer = TiktokenWrapper()

# Using a specific encoding
tokenizer = TiktokenWrapper(encoding_name="cl100k_base")

# Using encoding for a specific model
tokenizer = TiktokenWrapper(model_name="gpt-4")
```

### HuggingFaceTokenizer

Wrapper for HuggingFace transformers tokenizers. Use this when working with open-source models.

```python
from ragas import HuggingFaceTokenizer

# Load tokenizer for a specific model
tokenizer = HuggingFaceTokenizer(model_name="meta-llama/Llama-2-7b-hf")

# Use a pre-initialized tokenizer
from transformers import AutoTokenizer
hf_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer)
```

**Note:** HuggingFace tokenizers require the `transformers` package. Install it with:
```sh
pip install transformers
# or
uv add transformers
```

### Factory Function

Use `get_tokenizer()` for a simple way to create tokenizers:

```python
from ragas import get_tokenizer

# Default tiktoken tokenizer
tokenizer = get_tokenizer()

# Tiktoken for a specific model
tokenizer = get_tokenizer("tiktoken", model_name="gpt-4")

# HuggingFace tokenizer
tokenizer = get_tokenizer("huggingface", model_name="meta-llama/Llama-2-7b-hf")
```

## Using Custom Tokenizers

### With LLM-based Extractors

All LLM-based extractors accept a `tokenizer` parameter:

```python
from ragas import HuggingFaceTokenizer
from ragas.testset.transforms import (
    SummaryExtractor,
    KeyphrasesExtractor,
    HeadlinesExtractor,
)

# Create a HuggingFace tokenizer for your model
tokenizer = HuggingFaceTokenizer(model_name="meta-llama/Llama-2-7b-hf")

# Use it with extractors
summary_extractor = SummaryExtractor(llm=your_llm, tokenizer=tokenizer)
keyphrase_extractor = KeyphrasesExtractor(llm=your_llm, tokenizer=tokenizer)
headlines_extractor = HeadlinesExtractor(llm=your_llm, tokenizer=tokenizer)
```

### Custom Tokenizer Implementation

You can create your own tokenizer by extending `BaseTokenizer`:

```python
from ragas.tokenizers import BaseTokenizer

class MyCustomTokenizer(BaseTokenizer):
    def __init__(self, ...):
        # Initialize your tokenizer
        pass

    def encode(self, text: str) -> list[int]:
        # Return token IDs
        pass

    def decode(self, tokens: list[int]) -> str:
        # Return decoded text
        pass
```

## API Reference

::: ragas.tokenizers
