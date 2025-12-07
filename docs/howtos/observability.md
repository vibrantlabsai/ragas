# Observability Tools.

## Phoenix (Arize)

### 1. Introduction

Building a baseline for a RAG pipeline is not usually difficult, but enhancing it to make it suitable for production and ensuring the quality of your responses is almost always hard. Choosing the right tools and parameters for RAG can itself be challenging when there is an abundance of options available. This tutorial shares a robust workflow for making the right choices while building your RAG and ensuring its quality.

This article covers how to evaluate, visualize and analyze your RAG using a combination of open-source libraries.  We will be using:

- [Ragas](https://docs.ragas.io/en/stable/) for synthetic test data generation and evaluation
- Arize AI’s [Phoenix](https://docs.arize.com/phoenix) for tracing, visualization, and cluster analysis
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/) for building RAG pipelines

For the purpose of this article, we’ll be using data from arXiv papers about prompt-engineering to build the RAG pipeline.

ℹ️ This notebook requires an OpenAI API key.

### 2. Install Dependencies and Import Libraries

Run the cell below to install Git LFS, which we use to download our dataset.


```python
!git lfs install
```

Install and import Python dependencies.


```python
!pip install "ragas<0.1.1" pypdf arize-phoenix "openinference-instrumentation-llama-index<1.0.0" "llama-index<0.10.0" pandas
```


```python
import pandas as pd

# Display the complete contents of DataFrame cells.
pd.set_option("display.max_colwidth", None)
```

### 3. Configure Your OpenAI API Key

Set your OpenAI API key if it is not already set as an environment variable.


```python
import os
from getpass import getpass
import openai

if not (openai_api_key := os.getenv("OPENAI_API_KEY")):
    openai_api_key = getpass("🔑 Enter your OpenAI API key: ")
openai.api_key = openai_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key
```

### 4. Generate Your Synthetic Test Dataset

Curating a golden test dataset for evaluation can be a long, tedious, and expensive process that is not pragmatic — especially when starting out or when data sources keep changing. This can be solved by synthetically generating high quality data points, which then can be verified by developers. This can reduce the time and effort in curating test data by 90%.

Run the cell below to download a dataset of prompt engineering papers in PDF format from arXiv and read these documents using LlamaIndex.


```python
!git clone https://huggingface.co/datasets/vibrantlabsai/prompt-engineering-papers
```


```python
from llama_index import SimpleDirectoryReader

dir_path = "./prompt-engineering-papers"
reader = SimpleDirectoryReader(dir_path, num_files_limit=2)
documents = reader.load_data()
```

An ideal test dataset should contain data points of high quality and diverse nature from a similar distribution to the one observed during production. Ragas uses a unique evolution-based synthetic data generation paradigm to generate questions that are of the highest quality which also ensures diversity of questions generated.  Ragas by default uses OpenAI models under the hood, but you’re free to use any model of your choice. Let’s generate 100 data points using Ragas.


```python
from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
import openai

TEST_SIZE = 25

# generator with openai models
generator_llm = ChatOpenAI(model="gpt-4o-mini")
critic_llm = ChatOpenAI(model="gpt-4o")
openai_client = openai.OpenAI()
embeddings = OpenAIEmbeddings(client=openai_client)

generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

# generate testset
testset = generator.generate_with_llamaindex_docs(documents, test_size=TEST_SIZE)
test_df = testset.to_pandas()
test_df.head()
```

You are free to change the question type distribution according to your needs. Since we now have our test dataset ready, let’s move on and build a simple RAG pipeline using LlamaIndex.

### 5. Build Your RAG Application With LlamaIndex

LlamaIndex is an easy-to-use and flexible framework for building RAG applications. For the sake of simplicity, we use the default LLM (gpt-3.5-turbo) and embedding models (openai-ada-2).

Launch Phoenix in the background and instrument your LlamaIndex application so that your OpenInference spans and traces are sent to and collected by Phoenix. [OpenInference](https://github.com/Arize-ai/openinference/tree/main/spec) is an open standard built atop OpenTelemetry that captures and stores LLM application executions. It is designed to be a category of telemetry data that is used to understand the execution of LLMs and the surrounding application context, such as retrieval from vector stores and the usage of external tools such as search engines or APIs.


```python
import phoenix as px
from llama_index import set_global_handler

session = px.launch_app()
set_global_handler("arize_phoenix")
```

Build your query engine.


```python
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding


def build_query_engine(documents):
    vector_index = VectorStoreIndex.from_documents(
        documents,
        service_context=ServiceContext.from_defaults(chunk_size=512),
        embed_model=OpenAIEmbedding(),
    )
    query_engine = vector_index.as_query_engine(similarity_top_k=2)
    return query_engine


query_engine = build_query_engine(documents)
```

If you check Phoenix, you should see embedding spans from when your corpus data was indexed. Export and save those embeddings into a DataFrame for visualization later in the notebook.


```python
from phoenix.trace.dsl import SpanQuery

client = px.Client()
corpus_df = px.Client().query_spans(
    SpanQuery().explode(
        "embedding.embeddings",
        text="embedding.text",
        vector="embedding.vector",
    )
)
corpus_df.head()
```

Relaunch Phoenix to clear the accumulated traces.


```python
px.close_app()
session = px.launch_app()
```

### 6. Evaluate Your LLM Application

Ragas provides a comprehensive list of metrics that can be used to evaluate RAG pipelines both component-wise and end-to-end.

To use Ragas, we first form an evaluation dataset comprised of a question, generated answer, retrieved context, and ground-truth answer (the actual expected answer for the given question).


```python
from datasets import Dataset
from tqdm.auto import tqdm
import pandas as pd


def generate_response(query_engine, question):
    response = query_engine.query(question)
    return {
        "answer": response.response,
        "contexts": [c.node.get_content() for c in response.source_nodes],
    }


def generate_ragas_dataset(query_engine, test_df):
    test_questions = test_df["question"].values
    responses = [generate_response(query_engine, q) for q in tqdm(test_questions)]

    dataset_dict = {
        "question": test_questions,
        "answer": [response["answer"] for response in responses],
        "contexts": [response["contexts"] for response in responses],
        "ground_truth": test_df["ground_truth"].values.tolist(),
    }
    ds = Dataset.from_dict(dataset_dict)
    return ds


ragas_eval_dataset = generate_ragas_dataset(query_engine, test_df)
ragas_evals_df = pd.DataFrame(ragas_eval_dataset)
ragas_evals_df.head()
```

Check out Phoenix to view your LlamaIndex application traces.


```python
print(session.url)
```

![LlamaIndex application traces inside of Phoenix](https://storage.googleapis.com/arize-phoenix-assets/assets/docs/notebooks/ragas/ragas_trace_slide_over.gif)

We save out a couple of DataFrames, one containing embedding data that we'll visualize later, and another containing our exported traces and spans that we plan to evaluate using Ragas.


```python
# dataset containing embeddings for visualization
query_embeddings_df = px.Client().query_spans(
    SpanQuery().explode(
        "embedding.embeddings", text="embedding.text", vector="embedding.vector"
    )
)
query_embeddings_df.head()
```


```python
from phoenix.session.evaluation import get_qa_with_reference

# dataset containing span data for evaluation with Ragas
spans_dataframe = get_qa_with_reference(client)
spans_dataframe.head()
```

Ragas uses LangChain to evaluate your LLM application data. Let's instrument LangChain with OpenInference, so we can see what's going on under the hood when we evaluate our LLM application.


```python
from openinference.instrumentation.langchain import LangChainInstrumentor

LangChainInstrumentor().instrument()
```

Evaluate your LLM traces and view the evaluation scores in DataFrame format.


```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_recall,
    context_precision,
)

evaluation_result = evaluate(
    dataset=ragas_eval_dataset,
    metrics=[faithfulness, answer_correctness, context_recall, context_precision],
)
eval_scores_df = pd.DataFrame(evaluation_result.scores)
```

Submit your evaluations to Phoenix, so they are visible as annotations on your spans.


```python
from phoenix.trace import SpanEvaluations

# Assign span ids to your ragas evaluation scores (needed so Phoenix knows where to attach the spans).
eval_data_df = pd.DataFrame(evaluation_result.dataset)
assert eval_data_df.question.to_list() == list(
    reversed(spans_dataframe.input.to_list())  # The spans are in reverse order.
), "Phoenix spans are in an unexpected order. Re-start the notebook and try again."
eval_scores_df.index = pd.Index(
    list(reversed(spans_dataframe.index.to_list())), name=spans_dataframe.index.name
)

# Log the evaluations to Phoenix.
for eval_name in eval_scores_df.columns:
    evals_df = eval_scores_df[[eval_name]].rename(columns={eval_name: "score"})
    evals = SpanEvaluations(eval_name, evals_df)
    px.Client().log_evaluations(evals)
```

If you check out Phoenix, you'll see your Ragas evaluations as annotations on your application spans.


```python
print(session.url)
```

![ragas evaluations appear as annotations on your spans](https://storage.googleapis.com/arize-phoenix-assets/assets/docs/notebooks/ragas/ragas_evaluation_annotations.gif)

### 7. Visualize and Analyze Your Embeddings

[Embeddings](https://arize.com/blog-course/embeddings-meaning-examples-and-how-to-compute/) encode the meaning of retrieved documents and user queries. Not only are they an essential part of RAG systems, but they are immensely useful for understanding and debugging LLM application performance.

Phoenix takes the high-dimensional embeddings from your RAG application, reduces their dimensionality, and clusters them into semantically meaningful groups of data. You can then select the metric of your choice (e.g., Ragas-computed faithfulness or answer correctness) to visually inspect the performance of your application and surface problematic clusters. The advantage of this approach is that it provides metrics on granular yet meaningful subsets of your data that help you analyze local, not merely global, performance across a dataset. It's also helpful for gaining intuition around what kind of queries your LLM application is struggling to answer.

We'll re-launch Phoenix as an embedding visualizer to inspect the performance of our application on our test dataset.


```python
query_embeddings_df = query_embeddings_df.iloc[::-1]
assert ragas_evals_df.question.tolist() == query_embeddings_df.text.tolist()
assert test_df.question.tolist() == ragas_evals_df.question.tolist()
query_df = pd.concat(
    [
        ragas_evals_df[["question", "answer", "ground_truth"]].reset_index(drop=True),
        query_embeddings_df[["vector"]].reset_index(drop=True),
        test_df[["evolution_type"]],
        eval_scores_df.reset_index(drop=True),
    ],
    axis=1,
)
query_df.head()
```


```python
query_schema = px.Schema(
    prompt_column_names=px.EmbeddingColumnNames(
        raw_data_column_name="question", vector_column_name="vector"
    ),
    response_column_names="answer",
)
corpus_schema = px.Schema(
    prompt_column_names=px.EmbeddingColumnNames(
        raw_data_column_name="text", vector_column_name="vector"
    )
)
# relaunch phoenix with a primary and corpus dataset to view embeddings
px.close_app()
session = px.launch_app(
    primary=px.Dataset(query_df, query_schema, "query"),
    corpus=px.Dataset(corpus_df.reset_index(drop=True), corpus_schema, "corpus"),
)
```

Once you launch Phoenix, you can visualize your data with the metric of your choice with the following steps:

- Select the `vector` embedding,
- Select `Color By > dimension` and then the dimension of your choice to color your data by a particular field, for example, by Ragas evaluation scores such as faithfulness or answer correctness,
- Select the metric of your choice from the `metric` dropdown to view aggregate metrics on a per-cluster basis.

![inspect clusters of embeddings, view aggregate metrics, and color your data by the metric of your choice](https://storage.googleapis.com/arize-phoenix-assets/assets/docs/notebooks/ragas/ragas_correctness_clusters.gif)

### 8. Recap

Congrats! You built and evaluated a LlamaIndex query engine using Ragas and Phoenix. Let's recap what we learned:

- With Ragas, you bootstrapped a test dataset and computed metrics such as faithfulness and answer correctness to evaluate your LlamaIndex query engine.
- With OpenInference, you instrumented your query engine, so you could observe the inner workings of both LlamaIndex and Ragas.
- With Phoenix, you collected your spans and traces, imported your evaluations for easy inspection, and visualized your embedded queries and retrieved documents to identify pockets of poor performance.

This notebook is just an introduction to the capabilities of Ragas and Phoenix. To learn more, see the [Ragas](https://docs.ragas.io/en/stable/) and [Phoenix docs](https://docs.arize.com/phoenix/).

If you enjoyed this tutorial, please leave a ⭐ on GitHub:

- [Ragas](https://github.com/vibrantlabsai/ragas)
- [Phoenix](https://github.com/Arize-ai/phoenix)
- [OpenInference](https://github.com/Arize-ai/openinference)

## LangSmith

[LangSmith](https://docs.smith.langchain.com/) is an advanced tool designed to enhance the development and deployment of applications utilizing large language models (LLMs). It provides a comprehensive framework for tracing, analyzing, and optimizing LLM workflows, making it easier for developers to manage complex interactions within their applications.

This tutorial explains how to log traces of Ragas evaluations using LangSmith. Since Ragas is built on LangChain, you only need to set up LangSmith, and it will handle logging the traces automatically.

### 1. Setting Up LangSmith

To set up LangSmith, make sure you set the following environment variables (refer to the [LangSmith documentation](https://docs.smith.langchain.com/#quick-start) for more details):

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # Defaults to "default" if not set
```

### 2. Getting the Dataset

When creating evaluation dataset or evaluating instance, ensure the terminology matches the schema used in `SingleTurnSample` or `MultiTurnSample`.


```python
from ragas import EvaluationDataset


dataset = [
    {
        "user_input": "Which CEO is widely recognized for democratizing AI education through platforms like Coursera?",
        "retrieved_contexts": [
            "Andrew Ng, CEO of Landing AI, is known for his pioneering work in deep learning and for democratizing AI education through Coursera."
        ],
        "response": "Andrew Ng is widely recognized for democratizing AI education through platforms like Coursera.",
        "reference": "Andrew Ng, CEO of Landing AI, is known for democratizing AI education through Coursera.",
    },
    {
        "user_input": "Who is Sam Altman?",
        "retrieved_contexts": [
            "Sam Altman, CEO of OpenAI, has advanced AI research and advocates for safe, beneficial AI technologies."
        ],
        "response": "Sam Altman is the CEO of OpenAI and advocates for safe, beneficial AI technologies.",
        "reference": "Sam Altman, CEO of OpenAI, has advanced AI research and advocates for safe AI.",
    },
    {
        "user_input": "Who is Demis Hassabis and how did he gain prominence?",
        "retrieved_contexts": [
            "Demis Hassabis, CEO of DeepMind, is known for developing systems like AlphaGo that master complex games."
        ],
        "response": "Demis Hassabis is the CEO of DeepMind, known for developing systems like AlphaGo.",
        "reference": "Demis Hassabis, CEO of DeepMind, is known for developing AlphaGo.",
    },
    {
        "user_input": "Who is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's product ecosystem?",
        "retrieved_contexts": [
            "Sundar Pichai, CEO of Google and Alphabet Inc., leads innovation across Google's product ecosystem."
        ],
        "response": "Sundar Pichai is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's product ecosystem.",
        "reference": "Sundar Pichai, CEO of Google and Alphabet Inc., leads innovation across Google's product ecosystem.",
    },
    {
        "user_input": "How did Arvind Krishna transform IBM?",
        "retrieved_contexts": [
            "Arvind Krishna, CEO of IBM, transformed the company by focusing on cloud computing and AI solutions."
        ],
        "response": "Arvind Krishna transformed IBM by focusing on cloud computing and AI solutions.",
        "reference": "Arvind Krishna, CEO of IBM, transformed the company through cloud computing and AI.",
    },
]

evaluation_dataset = EvaluationDataset.from_list(dataset)
```

### 3. Tracing ragas metrics

Run the Ragas evaluations on your dataset, and the traces will appear in your LangSmith dashboard under the specified project name or "default."


```python
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

llm = ChatOpenAI(model="gpt-4o-mini")
evaluator_llm = LangchainLLMWrapper(llm)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm,
)

result
```

Output
```
Evaluating:   0%|          | 0/15 [00:00<?, ?it/s]

{'context_recall': 1.0000, 'faithfulness': 0.9333, 'factual_correctness': 0.8520}
```

### 4. LangSmith Dashboard
![jpeg](./../_static/langsmith_dashboard.png)
