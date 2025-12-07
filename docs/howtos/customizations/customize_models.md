## Customize Models

Ragas may use a LLM and or Embedding for evaluation and synthetic data generation. Both of these models can be customised according to your availability.

Ragas provides factory functions (`llm_factory` and `embedding_factory`) that support multiple providers:

- **Direct provider support**: OpenAI, Anthropic, Google 
- **Other providers via LiteLLM**: Azure OpenAI, AWS Bedrock, Google Vertex AI, and 100+ other providers

The factory functions use the [Instructor](https://python.useinstructor.com/) library for structured outputs and [LiteLLM](https://docs.litellm.ai/) for unified access to multiple LLM providers.

## Examples

- [Customize Models](#customize-models)
- [Examples](#examples)
  - [Azure OpenAI](#azure-openai)
  - [Google Vertex](#google-vertex)
  - [AWS Bedrock](#aws-bedrock)


### Azure OpenAI

```bash
pip install litellm
```

```python
import litellm
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

azure_configs = {
    "api_base": "https://<your-endpoint>.openai.azure.com/",
    "api_key": "your-api-key",
    "api_version": "2024-02-15-preview",
    "model_deployment": "your-deployment-name",
    "embedding_deployment": "your-embedding-deployment-name",
}

# Configure LiteLLM for Azure OpenAI (used by LLM calls)
litellm.api_base = azure_configs["api_base"]
litellm.api_key = azure_configs["api_key"]
litellm.api_version = azure_configs["api_version"]

# Create LLM using llm_factory with litellm provider
# Note: Use deployment name, not model name for Azure
# Important: Pass litellm.completion (the function), not the module
azure_llm = llm_factory(
    f"azure/{azure_configs['model_deployment']}",
    provider="litellm",
    client=litellm.completion,
)

# Create embeddings using embedding_factory
# Note: Pass Azure config directly to embedding_factory
azure_embeddings = embedding_factory(
    "litellm",
    model=f"azure/{azure_configs['embedding_deployment']}",
    api_base=azure_configs["api_base"],
    api_key=azure_configs["api_key"],
    api_version=azure_configs["api_version"],
)
```
Yay! Now you are ready to use ragas with Azure OpenAI endpoints

### Google Vertex

```bash
pip install litellm google-cloud-aiplatform
```

```python
import litellm
import os
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

config = {
    "project_id": "<your-project-id>",
    "location": "us-central1",  # e.g., "us-central1", "us-east1"
    "chat_model_id": "gemini-1.5-pro-002",
    "embedding_model_id": "text-embedding-005",
}

# Set environment variables for Vertex AI (used by litellm)
os.environ["VERTEXAI_PROJECT"] = config["project_id"]
os.environ["VERTEXAI_LOCATION"] = config["location"]

# Create LLM using llm_factory with litellm provider
# Important: Pass litellm.completion (the function), not the module
vertex_llm = llm_factory(
    f"vertex_ai/{config['chat_model_id']}",
    provider="litellm",
    client=litellm.completion,
)

# Create embeddings using embedding_factory
# Note: Embeddings use the environment variables set above
vertex_embeddings = embedding_factory(
    "litellm",
    model=f"vertex_ai/{config['embedding_model_id']}",
)
```
Yay! Now you are ready to use ragas with Google VertexAI endpoints

### AWS Bedrock

```bash
pip install litellm
```

```python
import litellm
import os
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

config = {
    "region_name": "us-east-1",  # E.g. "us-east-1"
    "llm": "anthropic.claude-3-5-sonnet-20241022-v2:0",  # Your LLM model ID
    "embeddings": "amazon.titan-embed-text-v2:0",  # Your embedding model ID
    "temperature": 0.4,
}

# Set AWS credentials as environment variables
# Option 1: Use AWS credentials file (~/.aws/credentials)
# Option 2: Set environment variables directly
os.environ["AWS_REGION_NAME"] = config["region_name"]
# os.environ["AWS_ACCESS_KEY_ID"] = "your-access-key"
# os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret-key"

# Create LLM using llm_factory with litellm provider
# Important: Pass litellm.completion (the function), not the module
bedrock_llm = llm_factory(
    f"bedrock/{config['llm']}",
    provider="litellm",
    client=litellm.completion,
    temperature=config["temperature"],
)

# Create embeddings using embedding_factory
# Note: Embeddings use the environment variables set above
bedrock_embeddings = embedding_factory(
    "litellm",
    model=f"bedrock/{config['embeddings']}",
)
```
Yay! Now you are ready to use ragas with AWS Bedrock endpoints
