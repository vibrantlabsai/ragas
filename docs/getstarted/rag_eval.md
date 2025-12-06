# Evaluate a simple RAG system

The purpose of this guide is to illustrate a simple workflow for testing and evaluating a RAG system with `ragas`. It assumes minimum knowledge in building RAG system and evaluation. Please refer to our [installation instruction](./install.md) for installing `ragas`.

## Basic Setup

We will use `langchain_openai` to set the LLM and embedding model for building our simple RAG. You may choose any other LLM and embedding model of your choice, to do that please refer to [customizing models in langchain](https://python.langchain.com/docs/integrations/chat/).


```python
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
import openai

llm = ChatOpenAI(model="gpt-4o")
openai_client = openai.OpenAI()
embeddings = OpenAIEmbeddings(client=openai_client)
```

!!! note "OpenAI Embeddings API"
    `ragas.embeddings.OpenAIEmbeddings` exposes `embed_text` (single) and `embed_texts` (batch), not `embed_query`/`embed_documents` like some LangChain wrappers. The example below uses `embed_texts` for documents and `embed_text` for the query. Please refer to [OpenAI embeddings implementation](https://docs.ragas.io/en/stable/references/embeddings/\#ragas.embeddings.OpenAIEmbeddings)

### Build a Simple RAG System

To build a simple RAG system, we need to define the following components:

- Define a method to vectorize our docs
- Define a method to retrieve the relevant docs
- Define a method to generate the response

??? note "Click to View the Code"

    ```python

    import numpy as np

    class RAG:
        def __init__(self, model="gpt-4o"):
            import openai
            self.llm = ChatOpenAI(model=model)
            openai_client = openai.OpenAI()
            self.embeddings = OpenAIEmbeddings(client=openai_client)
            self.doc_embeddings = None
            self.docs = None

        def load_documents(self, documents):
            """Load documents and compute their embeddings."""
            self.docs = documents
            self.doc_embeddings = self.embeddings.embed_texts(documents)

        def get_most_relevant_docs(self, query):
            """Find the most relevant document for a given query."""
            if not self.docs or not self.doc_embeddings:
                raise ValueError("Documents and their embeddings are not loaded.")
            
            query_embedding = self.embeddings.embed_text(query)
            similarities = [
                np.dot(query_embedding, doc_emb)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
                for doc_emb in self.doc_embeddings
            ]
            most_relevant_doc_index = np.argmax(similarities)
            return [self.docs[most_relevant_doc_index]]

        def generate_answer(self, query, relevant_doc):
            """Generate an answer for a given query based on the most relevant document."""
            prompt = f"question: {query}\n\nDocuments: {relevant_doc}"
            messages = [
                ("system", "You are a helpful assistant that answers questions based on given documents only."),
                ("human", prompt),
            ]
            ai_msg = self.llm.invoke(messages)
            return ai_msg.content
    ```

### Load Documents
Now, let's load some documents and test our RAG system.

```python
sample_docs = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
]
```

```python
# Initialize RAG instance
rag = RAG()

# Load documents
rag.load_documents(sample_docs)

# Query and retrieve the most relevant document
query = "Who introduced the theory of relativity?"
relevant_doc = rag.get_most_relevant_docs(query)

# Generate an answer
answer = rag.generate_answer(query, relevant_doc)

print(f"Query: {query}")
print(f"Relevant Document: {relevant_doc}")
print(f"Answer: {answer}")
```


Output:
```
Query: Who introduced the theory of relativity?
Relevant Document: ['Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.']
Answer: Albert Einstein introduced the theory of relativity.
```

## Collect Evaluation Data

To collect evaluation data, we first need a set of queries to run against our RAG. We can run the queries through the RAG system and collect the `response`, `retrieved_contexts`for each query. You may also optionally prepare a set of golden answers for each query to evaluate the system's performance.



```python


sample_queries = [
    "Who introduced the theory of relativity?",
    "Who was the first computer programmer?",
    "What did Isaac Newton contribute to science?",
    "Who won two Nobel Prizes for research on radioactivity?",
    "What is the theory of evolution by natural selection?"
]

expected_responses = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."
]
```

```python
dataset = []

for query,reference in zip(sample_queries,expected_responses):
    
    relevant_docs = rag.get_most_relevant_docs(query)
    response = rag.generate_answer(query, relevant_docs)
    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts":relevant_docs,
            "response":response,
            "reference":reference
        }
    )
```

Now, load the dataset into `EvaluationDataset` object.

```python
from ragas import EvaluationDataset
evaluation_dataset = EvaluationDataset.from_list(dataset)
```

## Evaluate

We have successfully collected the evaluation data. Now, we can evaluate our RAG system on the collected dataset using a set of commonly used RAG evaluation metrics. You may choose any model as [evaluator LLM](./../howtos/customizations/customize_models.md) for evaluation. 

```python
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper


evaluator_llm = LangchainLLMWrapper(llm)
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
result
```

Output
```
{'context_recall': 1.0000, 'faithfulness': 0.8571, 'factual_correctness': 0.7280}
```

### Want help in improving your AI application using evals?

In the past 2 years, we have seen and helped improve many AI applications using evals. 

We are compressing this knowledge into a product to replace vibe checks with eval loops so that you can focus on building great AI applications.

If you want help with improving and scaling up your AI application using evals.


🔗 Book a [slot](https://bit.ly/3EBYq4J) or drop us a line: [founders@vibrantlabs.com](mailto:founders@vibrantlabs.com).

![](../_static/ragas_app.gif)


## Up Next

- [Generate test data for evaluating RAG](rag_testset_generation.md)
