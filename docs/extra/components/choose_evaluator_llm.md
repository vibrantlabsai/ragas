=== "OpenAI"
    Install the langchain-openai package

    ```bash
    pip install langchain-openai
    ```

    Ensure you have your OpenAI key ready and available in your environment.

    ```python
    import os
    os.environ["OPENAI_API_KEY"] = "your-openai-key"
    ```
    Wrap the LLMs in `LangchainLLMWrapper` so that it can be used with ragas.

    ```python
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    import openai
    
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    openai_client = openai.OpenAI()
    evaluator_embeddings = OpenAIEmbeddings(client=openai_client)
    ```


=== "AWS"
    Install the langchain-aws package

    ```bash
    pip install langchain-aws
    ```

    Then you have to set your AWS credentials and configurations

    ```python
    config = {
        "credentials_profile_name": "your-profile-name",  # E.g "default"
        "region_name": "your-region-name",  # E.g. "us-east-1"
        "llm": "your-llm-model-id",  # E.g "anthropic.claude-3-5-sonnet-20241022-v2:0"
        "embeddings": "your-embedding-model-id",  # E.g "amazon.titan-embed-text-v2:0"
        "temperature": 0.4,
    }
    ```

    Define your LLMs and wrap them in `LangchainLLMWrapper` so that it can be used with ragas.

    ```python
    from langchain_aws import ChatBedrockConverse
    from langchain_aws import BedrockEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    evaluator_llm = LangchainLLMWrapper(ChatBedrockConverse(
        credentials_profile_name=config["credentials_profile_name"],
        region_name=config["region_name"],
        base_url=f"https://bedrock-runtime.{config['region_name']}.amazonaws.com",
        model=config["llm"],
        temperature=config["temperature"],
    ))
    evaluator_embeddings = LangchainEmbeddingsWrapper(BedrockEmbeddings(
        credentials_profile_name=config["credentials_profile_name"],
        region_name=config["region_name"],
        model_id=config["embeddings"],
    ))
    ```

    If you want more information on how to use other AWS services, please refer to the [langchain-aws](https://python.langchain.com/docs/integrations/providers/aws/) documentation.

=== "Google Cloud"
    Google offers two ways to access their models: Google AI Studio and Google Cloud Vertex AI. Google AI Studio requires just a Google account and API key, while Vertex AI requires a Google Cloud account. Use Google AI Studio if you're just starting out.

    First, install the required packages (only the packages you need based on your choice of API):

    ```bash
    # for Google AI Studio
    pip install langchain-google-genai
    # for Google Cloud Vertex AI
    pip install langchain-google-vertexai
    ```

    Then set up your credentials based on your chosen API:

    For Google AI Studio:
    ```python
    import os
    os.environ["GOOGLE_API_KEY"] = "your-google-ai-key"  # From https://ai.google.dev/
    ```

    For Google Cloud Vertex AI:
    ```python
    # Ensure you have credentials configured (gcloud, workload identity, etc.)
    # Or set service account JSON path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/service-account.json"
    ```

    Define your configuration:

    ```python
    config = {
        "model": "gemini-1.5-pro",  # or other model IDs
        "temperature": 0.4,
        "max_tokens": None,
        "top_p": 0.8,
        # For Vertex AI only:
        "project": "your-project-id",  # Required for Vertex AI
        "location": "us-central1",     # Required for Vertex AI
    }
    ```

    Initialize the LLM and wrap it for use with ragas:

    ```python
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # Choose the appropriate import based on your API:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_google_vertexai import ChatVertexAI

    # Initialize with Google AI Studio
    evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        top_p=config["top_p"],
    ))

    # Or initialize with Vertex AI
    evaluator_llm = LangchainLLMWrapper(ChatVertexAI(
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        top_p=config["top_p"],
        project=config["project"],
        location=config["location"],
    ))
    ```

    You can optionally configure safety settings:

    ```python
    from langchain_google_genai import HarmCategory, HarmBlockThreshold

    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        # Add other safety settings as needed
    }

    # Apply to your LLM initialization
    evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=config["model"],
        temperature=config["temperature"],
        safety_settings=safety_settings,
    ))
    ```

    Initialize the embeddings and wrap them for use with ragas (choose one of the following):

    ```python
    # Google AI Studio Embeddings
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    evaluator_embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # Google's text embedding model
        task_type="retrieval_document"  # Optional: specify the task type
    ))
    ```

    ```python
    # Vertex AI Embeddings
    from langchain_google_vertexai import VertexAIEmbeddings

    evaluator_embeddings = LangchainEmbeddingsWrapper(VertexAIEmbeddings(
        model_name="textembedding-gecko@001",  # or other available model
        project=config["project"],  # Your GCP project ID
        location=config["location"]  # Your GCP location
    ))
    ```

    For more information on available models, features, and configurations, refer to: [Google AI Studio documentation](https://ai.google.dev/docs), [Google Cloud Vertex AI documentation](https://cloud.google.com/vertex-ai/docs), [LangChain Google AI integration](https://python.langchain.com/docs/integrations/chat/google_generative_ai), [LangChain Vertex AI integration](https://python.langchain.com/docs/integrations/chat/google_vertex_ai)

=== "Azure"
    Install the langchain-openai package

    ```bash
    pip install langchain-openai
    ```

    Ensure you have your Azure OpenAI key ready and available in your environment.

    ```python
    import os
    os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-openai-key"

    # other configuration
    azure_config = {
        "base_url": "",  # your endpoint
        "model_deployment": "",  # your model deployment name
        "model_name": "",  # your model name
        "embedding_deployment": "",  # your embedding deployment name
        "embedding_name": "",  # your embedding name
    }

    ```

    Define your LLMs and wrap them in `LangchainLLMWrapper` so that it can be used with ragas.

    ```python
    from langchain_openai import AzureChatOpenAI
    from langchain_openai import AzureOpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    evaluator_llm = LangchainLLMWrapper(AzureChatOpenAI(
        openai_api_version="2023-05-15",
        azure_endpoint=azure_config["base_url"],
        azure_deployment=azure_config["model_deployment"],
        model=azure_config["model_name"],
        validate_base_url=False,
    ))

    # init the embeddings for answer_relevancy, answer_correctness and answer_similarity
    evaluator_embeddings = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
        openai_api_version="2023-05-15",
        azure_endpoint=azure_config["base_url"],
        azure_deployment=azure_config["embedding_deployment"],
        model=azure_config["embedding_name"],
    ))
    ```

    If you want more information on how to use other Azure services, please refer to the [langchain-azure](https://python.langchain.com/docs/integrations/chat/azure_chat_openai/) documentation.


=== "Others"
    If you are using a different LLM provider and using LangChain to interact with it, you can wrap your LLM in `LangchainLLMWrapper` so that it can be used with ragas.

    ```python
    from ragas.llms import LangchainLLMWrapper
    evaluator_llm = LangchainLLMWrapper(your_llm_instance)
    ```

    For a more detailed guide, checkout [the guide on customizing models](../../howtos/customizations/customize_models.md).

    If you using LlamaIndex, you can use the `LlamaIndexLLMWrapper` to wrap your LLM so that it can be used with ragas.

    ```python
    from ragas.llms import LlamaIndexLLMWrapper
    evaluator_llm = LlamaIndexLLMWrapper(your_llm_instance)
    ```

    For more information on how to use LlamaIndex, please refer to the [LlamaIndex Integration guide](./../../howtos/integrations/_llamaindex.md).

    If your still not able use Ragas with your favorite LLM provider, please let us know by by commenting on this [issue](https://github.com/vibrantlabsai/ragas/issues/1617) and we'll add support for it 🙂.