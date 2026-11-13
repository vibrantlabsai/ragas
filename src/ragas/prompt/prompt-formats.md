# Prompt JSON Format Reference

> **Developer Reference for Ragas Contributors**
>
> This document provides technical specifications for the JSON formats used by `Prompt` and `DynamicFewShotPrompt` save/load functionality.

## Overview

Both prompt types use JSON format with optional gzip compression (.json.gz) for persistence. The formats share common base fields but have different type identifiers and extensions.

## Format Comparison

| Feature | Base Prompt | DynamicFewShotPrompt |
|---------|-------------|----------------------|
| Type ID | `"Prompt"` | `"DynamicFewShotPrompt"` |
| Examples Storage | `examples` array | `examples` array (from `example_store`) |
| Response Model | ✅ Supported | ✅ Supported |
| Embedding Model | ❌ Not supported | ✅ Supported |
| Embeddings Data | ❌ Not supported | ✅ Optional |
| Similarity Config | ❌ Not supported | ✅ `max_similar_examples`, `similarity_threshold` |
| File Extensions | `.json`, `.json.gz` | `.json`, `.json.gz` |

## Base Prompt Format

### JSON Schema

```json
{
  "format_version": "1.0",
  "type": "Prompt",
  "instruction": "string",
  "examples": [
    {
      "input": {}, 
      "output": {}
    }
  ],
  "response_model_info": null | {
    "class_name": "string",
    "module": "string", 
    "schema": {},
    "note": "You must provide this model when loading"
  }
}
```

### Field Specifications

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `format_version` | `string` | ✅ | Format version for compatibility (currently "1.0") |
| `type` | `string` | ✅ | Must be "Prompt" for base prompts |
| `instruction` | `string` | ✅ | Template string with {variable} placeholders |
| `examples` | `array` | ✅ | List of input/output example pairs (can be empty) |
| `response_model_info` | `object\|null` | ✅ | Pydantic model metadata (null if no response model) |

### Example: Basic Prompt

```json
{
  "format_version": "1.0",
  "type": "Prompt",
  "instruction": "Answer the question: {question}",
  "examples": [
    {
      "input": {"question": "What is 2+2?"},
      "output": {"answer": "4"}
    },
    {
      "input": {"question": "What is the capital of France?"},
      "output": {"answer": "Paris"}
    }
  ],
  "response_model_info": null
}
```

### Example: Prompt with Response Model

```json
{
  "format_version": "1.0",
  "type": "Prompt", 
  "instruction": "Analyze the sentiment: {text}",
  "examples": [
    {
      "input": {"text": "I love this!"},
      "output": {"sentiment": "positive", "confidence": 0.95}
    }
  ],
  "response_model_info": {
    "class_name": "SentimentResponse",
    "module": "myapp.models",
    "schema": {
      "type": "object",
      "properties": {
        "sentiment": {"type": "string"},
        "confidence": {"type": "number"}
      },
      "required": ["sentiment", "confidence"]
    },
    "note": "You must provide this model when loading"
  }
}
```

## DynamicFewShotPrompt Format

### JSON Schema

```json
{
  "format_version": "1.0",
  "type": "DynamicFewShotPrompt",
  "instruction": "string",
  "examples": [
    {
      "input": {},
      "output": {}
    }
  ],
  "response_model_info": null | {
    "class_name": "string",
    "module": "string",
    "schema": {},
    "note": "You must provide this model when loading"
  },
  "max_similar_examples": "integer",
  "similarity_threshold": "number",
  "embedding_model_info": null | {
    "class_name": "string", 
    "module": "string",
    "note": "You must provide this model when loading"
  },
  "embeddings": [
    [0.1, 0.2, 0.3, ...]
  ]
}
```

### Extended Field Specifications

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `max_similar_examples` | `integer` | ✅ | Maximum number of examples to return from similarity search |
| `similarity_threshold` | `number` | ✅ | Minimum similarity score for including examples (0.0-1.0) |
| `embedding_model_info` | `object\|null` | ✅ | Embedding model metadata (null if no embedding model) |
| `embeddings` | `array\|undefined` | ❌ | Pre-computed embeddings (only present if `include_embeddings=True`) |

### Example: Basic DynamicFewShotPrompt

```json
{
  "format_version": "1.0",
  "type": "DynamicFewShotPrompt",
  "instruction": "Answer the math question: {question}",
  "examples": [
    {
      "input": {"question": "What is 1+1?"},
      "output": {"answer": "2"}
    },
    {
      "input": {"question": "What is 3+3?"},
      "output": {"answer": "6"}
    }
  ],
  "response_model_info": null,
  "max_similar_examples": 2,
  "similarity_threshold": 0.8,
  "embedding_model_info": null
}
```

### Example: DynamicFewShotPrompt with Embeddings

```json
{
  "format_version": "1.0", 
  "type": "DynamicFewShotPrompt",
  "instruction": "Classify the text: {text}",
  "examples": [
    {
      "input": {"text": "I love this product!"},
      "output": {"category": "positive"}
    },
    {
      "input": {"text": "This is terrible."},
      "output": {"category": "negative"}
    }
  ],
  "response_model_info": null,
  "max_similar_examples": 3,
  "similarity_threshold": 0.7,
  "embedding_model_info": {
    "class_name": "OpenAIEmbeddings",
    "module": "ragas.embeddings.openai_provider",
    "note": "You must provide this model when loading"
  },
  "embeddings": [
    [0.1, 0.2, 0.3, -0.1, 0.5, ...],
    [-0.2, 0.4, 0.1, 0.3, -0.4, ...]
  ]
}
```

## Loading Prompts Programmatically

### Basic Loading

```python
from ragas.experimental.prompt.base import Prompt
from ragas.experimental.prompt.dynamic_few_shot import DynamicFewShotPrompt

# Load base prompt
prompt = Prompt.load("my_prompt.json")

# Load dynamic prompt
dynamic_prompt = DynamicFewShotPrompt.load("my_dynamic_prompt.json")

# Load with models
from mymodels import MyResponseModel, MyEmbeddingModel

prompt = Prompt.load("prompt.json", response_model=MyResponseModel())
dynamic_prompt = DynamicFewShotPrompt.load(
    "dynamic.json", response_model=MyResponseModel(), embedding_model=MyEmbeddingModel()
)
```

### File Format Detection

```python
import json
from pathlib import Path


def detect_prompt_type(filepath: str) -> str:
    """Detect prompt type from JSON file."""
    path = Path(filepath)

    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    return data.get("type", "unknown")


# Usage
prompt_type = detect_prompt_type("my_prompt.json")
if prompt_type == "Prompt":
    prompt = Prompt.load("my_prompt.json")
elif prompt_type == "DynamicFewShotPrompt":
    prompt = DynamicFewShotPrompt.load("my_prompt.json")
```

### Validation Helper

```python
def validate_prompt_file(filepath: str) -> dict:
    """Validate prompt file format and return metadata."""
    try:
        path = Path(filepath)
        if path.suffix == ".gz":
            import gzip

            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        # Basic validation
        required_fields = ["format_version", "type", "instruction", "examples"]
        missing_fields = [f for f in required_fields if f not in data]

        if missing_fields:
            return {"valid": False, "errors": f"Missing fields: {missing_fields}"}

        # Type-specific validation
        if data["type"] == "DynamicFewShotPrompt":
            dynamic_fields = ["max_similar_examples", "similarity_threshold"]
            missing_dynamic = [f for f in dynamic_fields if f not in data]
            if missing_dynamic:
                return {
                    "valid": False,
                    "errors": f"Missing dynamic fields: {missing_dynamic}",
                }

        return {
            "valid": True,
            "type": data["type"],
            "format_version": data["format_version"],
            "has_response_model": data.get("response_model_info") is not None,
            "has_embedding_model": data.get("embedding_model_info") is not None,
            "has_embeddings": "embeddings" in data,
            "example_count": len(data.get("examples", [])),
        }

    except Exception as e:
        return {"valid": False, "errors": str(e)}
```

## Working with Embedding Data

### Embedding Storage Considerations

```python
# Save without embeddings (smaller files, recomputation on load)
dynamic_prompt.save("prompt.json", include_embeddings=False)

# Save with embeddings (larger files, faster loading)
dynamic_prompt.save("prompt.json", include_embeddings=True)

# File size comparison
import os

size_without = os.path.getsize("prompt_no_emb.json")
size_with = os.path.getsize("prompt_with_emb.json")
print(f"Size difference: {size_with - size_without} bytes")
```

### Embedding Compatibility Check

```python
def check_embedding_compatibility(filepath: str, embedding_model) -> bool:
    """Check if saved embeddings are compatible with current model."""
    import json
    from pathlib import Path

    path = Path(filepath)
    with open(path, "r") as f:
        data = json.load(f)

    if "embedding_model_info" not in data or not data["embedding_model_info"]:
        return False

    saved_info = data["embedding_model_info"]
    current_class = embedding_model.__class__.__name__
    current_module = embedding_model.__class__.__module__

    return (
        saved_info["class_name"] == current_class
        and saved_info["module"] == current_module
    )
```

## Extending Prompt Types

### Adding New Prompt Type

When creating a new prompt type, follow this pattern:

```python
class MyCustomPrompt(Prompt):
    def __init__(self, instruction: str, my_custom_field: str, **kwargs):
        super().__init__(instruction, **kwargs)
        self.my_custom_field = my_custom_field

    def save(self, path: str) -> None:
        """Override to include custom fields."""
        # Build extended data structure
        data = {
            "format_version": "1.0",
            "type": "MyCustomPrompt",  # Unique type identifier
            "instruction": self.instruction,
            "examples": [{"input": inp, "output": out} for inp, out in self.examples],
            "response_model_info": self._serialize_response_model_info(),
            # Custom fields
            "my_custom_field": self.my_custom_field,
        }

        # Use same file handling as base class
        file_path = Path(path)
        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "wt", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
        except (OSError, IOError) as e:
            raise ValueError(f"Cannot save MyCustomPrompt to {path}: {e}")

    @classmethod
    def load(cls, path: str, response_model=None):
        """Override to handle custom fields."""
        # Use same file loading as base class
        file_path = Path(path)
        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Cannot load MyCustomPrompt from {path}: {e}")

        # Validate type
        if data.get("type") != "MyCustomPrompt":
            raise ValueError(
                f"File is not a MyCustomPrompt (found: {data.get('type')})"
            )

        # Extract data
        examples = [(ex["input"], ex["output"]) for ex in data.get("examples", [])]
        my_custom_field = data["my_custom_field"]

        # Create instance
        return cls(
            instruction=data["instruction"],
            examples=examples,
            response_model=response_model,
            my_custom_field=my_custom_field,
        )
```

## Implementation Details

### Model Serialization Methods

Both prompt types use these internal methods:

```python
def _serialize_response_model_info(self) -> Optional[Dict]:
    """Serialize Pydantic response model information."""
    if not self.response_model:
        return None

    return {
        "class_name": self.response_model.__class__.__name__,
        "module": self.response_model.__class__.__module__,
        "schema": self.response_model.model_json_schema(),
        "note": "You must provide this model when loading",
    }


# DynamicFewShotPrompt only
def _serialize_embedding_model_info(self) -> Optional[Dict]:
    """Serialize embedding model information."""
    if not self.example_store.embedding_model:
        return None

    return {
        "class_name": self.example_store.embedding_model.__class__.__name__,
        "module": self.example_store.embedding_model.__class__.__module__,
        "note": "You must provide this model when loading",
    }
```

### Error Handling Patterns

```python
# File format validation
if data.get("type") != "ExpectedType":
    raise ValueError(f"File is not a {expected_type} (found type: {data.get('type', 'unknown')})")

# Missing model validation  
response_model_info = data.get("response_model_info")
if response_model_info and not response_model:
    raise ValueError(
        f"This prompt requires a response_model of type '{response_model_info['class_name']}'\n"
        f"Usage: PromptClass.load('{path}', response_model=YourModel)"
    )

# File I/O errors
except (OSError, IOError) as e:
    raise ValueError(f"Cannot save/load prompt to/from {path}: {e}")
```

### Performance Considerations

1. **Embedding Storage**: Include embeddings for faster loading, exclude for smaller files
2. **Compression**: Use `.json.gz` for large prompt files (especially with embeddings)
3. **Memory Usage**: Large embedding arrays can consume significant memory
4. **Recomputation**: Without saved embeddings, all examples are re-embedded on load

### Migration Between Formats

```python
def convert_prompt_to_dynamic(
    base_prompt_path: str,
    output_path: str,
    embedding_model=None,
    max_examples: int = 3,
    threshold: float = 0.7,
):
    """Convert base Prompt to DynamicFewShotPrompt."""
    # Load base prompt
    base_prompt = Prompt.load(base_prompt_path)

    # Create dynamic version
    dynamic_prompt = DynamicFewShotPrompt(
        instruction=base_prompt.instruction,
        examples=base_prompt.examples,
        response_model=base_prompt.response_model,
        embedding_model=embedding_model,
        max_similar_examples=max_examples,
        similarity_threshold=threshold,
    )

    # Save new format
    dynamic_prompt.save(output_path)
```

## Format Evolution

### Version Compatibility

- **format_version**: "1.0" - Current version for both prompt types
- **Backwards Compatibility**: New fields should be optional with sensible defaults
- **Forward Compatibility**: Unknown fields should be ignored during loading

### Adding New Fields

When extending formats:

1. **Make fields optional** with defaults
2. **Update format_version** only for breaking changes  
3. **Add validation** for new fields
4. **Document migration path** for existing files
5. **Update tests** to cover new functionality

---

*This documentation is maintained alongside the codebase in `ragas_experimental/prompt/`. Please update when modifying save/load functionality.*