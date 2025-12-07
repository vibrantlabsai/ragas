# OCI Gen AI Integration

This guide shows how to use Oracle Cloud Infrastructure (OCI) Generative AI models with Ragas for evaluation.

## Installation

First, install the OCI dependency:

```bash
pip install ragas[oci]
```

## Setup

### 1. Configure OCI Authentication

Set up your OCI configuration using one of these methods:

#### Option A: OCI CLI Configuration
```bash
oci setup config
```

#### Option B: Environment Variables
```bash
export OCI_CONFIG_FILE=~/.oci/config
export OCI_PROFILE=DEFAULT
```

#### Option C: Manual Configuration
```python
config = {
    "user": "ocid1.user.oc1..example",
    "key_file": "~/.oci/private_key.pem",
    "fingerprint": "your_fingerprint",
    "tenancy": "ocid1.tenancy.oc1..example",
    "region": "us-ashburn-1"
}
```

### 2. Get Required IDs

You'll need:
- **Model ID**: The OCI model ID (e.g., `cohere.command`, `meta.llama-3-8b`)
- **Compartment ID**: Your OCI compartment OCID
- **Endpoint ID** (optional): If using a custom endpoint

## Usage

### Basic Usage

```python
from ragas.llms import oci_genai_factory
from ragas import evaluate
from datasets import Dataset

# Initialize OCI Gen AI LLM
llm = oci_genai_factory(
    model_id="cohere.command",
    compartment_id="ocid1.compartment.oc1..example"
)

# Your dataset
dataset = Dataset.from_dict({
    "question": ["What is the capital of France?"],
    "answer": ["Paris"],
    "contexts": [["France is a country in Europe. Its capital is Paris."]],
    "ground_truth": ["Paris"]
})

# Evaluate with OCI Gen AI
result = evaluate(
    dataset,
    llm=llm,
    embeddings=None  # You can use any embedding model
)
```

### Advanced Configuration

```python
from ragas.llms import oci_genai_factory
from ragas.run_config import RunConfig

# Custom OCI configuration
config = {
    "user": "ocid1.user.oc1..example",
    "key_file": "~/.oci/private_key.pem",
    "fingerprint": "your_fingerprint",
    "tenancy": "ocid1.tenancy.oc1..example",
    "region": "us-ashburn-1"
}

# Custom run configuration
run_config = RunConfig(
    timeout=60,
    max_retries=3
)

# Initialize with custom config and endpoint
llm = oci_genai_factory(
    model_id="cohere.command",
    compartment_id="ocid1.compartment.oc1..example",
    config=config,
    endpoint_id="ocid1.endpoint.oc1..example",  # Optional
    run_config=run_config
)
```

### Using with Different Models

```python
# Cohere Command model
llm_cohere = oci_genai_factory(
    model_id="cohere.command",
    compartment_id="ocid1.compartment.oc1..example"
)

# Meta Llama model
llm_llama = oci_genai_factory(
    model_id="meta.llama-3-8b",
    compartment_id="ocid1.compartment.oc1..example"
)

# Using with different endpoints
llm_endpoint = oci_genai_factory(
    model_id="cohere.command",
    compartment_id="ocid1.compartment.oc1..example",
    endpoint_id="ocid1.endpoint.oc1..example"
)
```

## Available Models

OCI Gen AI supports various models including:

- **Cohere**: `cohere.command`, `cohere.command-light`
- **Meta**: `meta.llama-3-8b`, `meta.llama-3-70b`
- **Mistral**: `mistral.mistral-7b-instruct`
- **And more**: Check OCI documentation for the latest available models

## Error Handling

The OCI Gen AI wrapper includes comprehensive error handling:

```python
try:
    result = evaluate(dataset, llm=llm)
except Exception as e:
    print(f"Evaluation failed: {e}")
```

## Performance Considerations

1. **Rate Limits**: OCI Gen AI has rate limits. Use appropriate retry configurations.
2. **Timeout**: Set appropriate timeouts for your use case.
3. **Batch Processing**: The wrapper supports batch processing for multiple completions.

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```
   Error: OCI SDK authentication failed
   ```
   Solution: Verify your OCI configuration and credentials.

2. **Model Not Found**
   ```
   Error: Model not found in compartment
   ```
   Solution: Check if the model ID exists in your compartment.

3. **Permission Errors**
   ```
   Error: Insufficient permissions
   ```
   Solution: Ensure your user has the necessary IAM policies for Generative AI.

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your OCI Gen AI code here
```

## Examples

### Complete Evaluation Example

```python
from ragas import evaluate
from ragas.llms import oci_genai_factory
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

# Initialize OCI Gen AI
llm = oci_genai_factory(
    model_id="cohere.command",
    compartment_id="ocid1.compartment.oc1..example"
)

# Create dataset
dataset = Dataset.from_dict({
    "question": [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?"
    ],
    "answer": [
        "Paris is the capital of France.",
        "William Shakespeare wrote Romeo and Juliet."
    ],
    "contexts": [
        ["France is a country in Europe. Its capital is Paris."],
        ["Romeo and Juliet is a play by William Shakespeare."]
    ],
    "ground_truth": [
        "Paris",
        "William Shakespeare"
    ]
})

# Evaluate
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=llm
)

print(result)
```

### Custom Metrics with OCI Gen AI

```python
from ragas.metrics import MetricWithLLM

# Create custom metric using OCI Gen AI
class CustomMetric(MetricWithLLM):
    def __init__(self):
        super().__init__()
        self.llm = oci_genai_factory(
            model_id="cohere.command",
            compartment_id="ocid1.compartment.oc1..example"
        )

# Use in evaluation
result = evaluate(
    dataset,
    metrics=[CustomMetric()],
    llm=llm
)
```

## Best Practices

1. **Use Appropriate Models**: Choose models based on your evaluation needs.
2. **Monitor Costs**: OCI Gen AI usage is billed. Monitor your usage.
3. **Handle Errors**: Implement proper error handling for production use.
4. **Use Caching**: Enable caching for repeated evaluations.
5. **Batch Operations**: Use batch operations when possible for efficiency.

## Support

For issues specific to OCI Gen AI integration:
- Check OCI documentation: https://docs.oracle.com/en-us/iaas/Content/generative-ai/
- OCI Python SDK: https://docs.oracle.com/en-us/iaas/tools/python/2.160.1/api/generative_ai.html
- Ragas GitHub issues: https://github.com/vibrantlabsai/ragas/issues
