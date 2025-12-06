# Openlayer
## Evaluating RAG pipelines with Openlayer and Ragas

[Openlayer](https://www.openlayer.com/) is an evaluation tool that fits into your development and production pipelines to help you ship high-quality models with confidence.

This notebook should be used together with [this blog post](https://www.openlayer.com/blog/post/evaluating-rag-pipelines-with-ragas-and-openlayer).

## Pre-requisites


```bash
%%bash
git clone https://huggingface.co/datasets/vibrantlabsai/prompt-engineering-papers
```


```python
import os

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_HERE"
```

## Synthetic test data generation


```python
from llama_index import SimpleDirectoryReader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# load documents
dir_path = "./prompt-engineering-papers"
reader = SimpleDirectoryReader(dir_path, num_files_limit=2)
documents = reader.load_data()

# generator with openai models
generator = TestsetGenerator.with_openai()

# set question type distribution
distribution = {simple: 0.5, reasoning: 0.25, multi_context: 0.25}

# generate testset
testset = generator.generate_with_llamaindex_docs(
    documents, test_size=10, distributions=distribution
)
test_df = testset.to_pandas()
test_df.head()
```

## Building RAG


```python
import nest_asyncio
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings import OpenAIEmbedding


nest_asyncio.apply()


def build_query_engine(documents):
    vector_index = VectorStoreIndex.from_documents(
        documents,
        service_context=ServiceContext.from_defaults(chunk_size=512),
        embed_model=OpenAIEmbedding(),
    )

    query_engine = vector_index.as_query_engine(similarity_top_k=2)
    return query_engine
```


```python
query_engine = build_query_engine(documents)
```


```python
def generate_single_response(query_engine, question):
    response = query_engine.query(question)
    return {
        "answer": response.response,
        "contexts": [c.node.get_content() for c in response.source_nodes],
    }
```


```python
question = "What are some strategies proposed to enhance the in-context learning capability of language models?"
generate_single_response(query_engine, question)
```


```python
from datasets import Dataset


def generate_ragas_dataset(query_engine, test_df):
    test_questions = test_df["question"].values
    responses = [generate_single_response(query_engine, q) for q in test_questions]

    dataset_dict = {
        "question": test_questions,
        "answer": [response["answer"] for response in responses],
        "contexts": [response["contexts"] for response in responses],
        "ground_truth": test_df["ground_truth"].values.tolist(),
    }
    ds = Dataset.from_dict(dataset_dict)
    return ds
```


```python
ragas_dataset = generate_ragas_dataset(query_engine, test_df)
ragas_df = ragas_dataset.to_pandas()
```

## Commit to Openlayer


```python
from openlayer.tasks import TaskType

client = openlayer.OpenlayerClient("YOUR_OPENLAYER_API_KEY_HERE")
```


```python
project = client.create_project(
    name="My-Rag-Project",
    task_type=TaskType.LLM,
    description="Evaluating an LLM used for product development.",
)
```


```python
validation_dataset_config = {
    "contextColumnName": "contexts",
    "questionColumnName": "question",
    "inputVariableNames": ["question"],
    "label": "validation",
    "outputColumnName": "answer",
    "groundTruthColumnName": "ground_truth",
}
project.add_dataframe(
    dataset_df=ragas_df,
    dataset_config=validation_dataset_config,
)
```


```python
model_config = {
    "inputVariableNames": ["question"],
    "modelType": "shell",
    "metadata": {"top_k": 2, "chunk_size": 512, "embeddings": "OpenAI"},
}
project.add_model(model_config=model_config)
```


```python
project.commit("Initial commit!")
project.push()
```


```python

```
