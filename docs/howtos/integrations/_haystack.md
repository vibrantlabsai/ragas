# Haystack Integration

Haystack is a  LLM orchestration framework to build customizable, production-ready LLM applications. 

The underlying concept of Haystack is that all individual tasks, such as storing documents, retrieving relevant data, and generating responses, are handled by modular components like Document Stores, Retrievers, and Generators, which are seamlessly connected and orchestrated using Pipelines.

## Overview

In this tutorial, we will build a RAG pipeline using Haystack and evaluate it with Ragas. We’ll start by setting up the various components of the RAG pipeline, and for evaluations, we will initialize the RagasEvaluator component. Once the components are set up, we'll connect the components to form the complete pipeline. Later in the tutorial, we will explore how to perform evaluations using custom-defined metrics in Ragas.

## Installing Dependencies


```python
%pip install ragas-haystack
```

#### Getting the data


```python
dataset = [
    "OpenAI is one of the most recognized names in the large language model space, known for its GPT series of models. These models excel at generating human-like text and performing tasks like creative writing, answering questions, and summarizing content. GPT-4, their latest release, has set benchmarks in understanding context and delivering detailed responses.",
    "Anthropic is well-known for its Claude series of language models, designed with a strong focus on safety and ethical AI behavior. Claude is particularly praised for its ability to follow complex instructions and generate text that aligns closely with user intent.",
    "DeepMind, a division of Google, is recognized for its cutting-edge Gemini models, which are integrated into various Google products like Bard and Workspace tools. These models are renowned for their conversational abilities and their capacity to handle complex, multi-turn dialogues.",
    "Meta AI is best known for its LLaMA (Large Language Model Meta AI) series, which has been made open-source for researchers and developers. LLaMA models are praised for their ability to support innovation and experimentation due to their accessibility and strong performance.",
    "Meta AI with it's LLaMA models aims to democratize AI development by making high-quality models available for free, fostering collaboration across industries. Their open-source approach has been a game-changer for researchers without access to expensive resources.",
    "Microsoft’s Azure AI platform is famous for integrating OpenAI’s GPT models, enabling businesses to use these advanced models in a scalable and secure cloud environment. Azure AI powers applications like Copilot in Office 365, helping users draft emails, generate summaries, and more.",
    "Amazon’s Bedrock platform is recognized for providing access to various language models, including its own models and third-party ones like Anthropic’s Claude and AI21’s Jurassic. Bedrock is especially valued for its flexibility, allowing users to choose models based on their specific needs.",
    "Cohere is well-known for its language models tailored for business use, excelling in tasks like search, summarization, and customer support. Their models are recognized for being efficient, cost-effective, and easy to integrate into workflows.",
    "AI21 Labs is famous for its Jurassic series of language models, which are highly versatile and capable of handling tasks like content creation and code generation. The Jurassic models stand out for their natural language understanding and ability to generate detailed and coherent responses.",
    "In the rapidly advancing field of artificial intelligence, several companies have made significant contributions with their large language models. Notable players include OpenAI, known for its GPT Series (including GPT-4); Anthropic, which offers the Claude Series; Google DeepMind with its Gemini Models; Meta AI, recognized for its LLaMA Series; Microsoft Azure AI, which integrates OpenAI’s GPT Models; Amazon AWS (Bedrock), providing access to various models including Claude (Anthropic) and Jurassic (AI21 Labs); Cohere, which offers its own models tailored for business use; and AI21 Labs, known for its Jurassic Series. These companies are shaping the landscape of AI by providing powerful models with diverse capabilities.",
]
```

## Initialize components for RAG pipeline

#### Initializing the DocumentStore


```python
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()
docs = [Document(content=doc) for doc in dataset]
```

#### Initalize the Document and Text Embedder


```python
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder

document_embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small")
text_embedder = OpenAITextEmbedder(model="text-embedding-3-small")
```

Now we have our document store and the document embedder, using them we will fill populate out vector datastore.


```python
docs_with_embeddings = document_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])
```

    Calculating embeddings: 1it [00:01,  1.74s/it]





    10



#### Initialize the Retriever


```python
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

retriever = InMemoryEmbeddingRetriever(document_store, top_k=2)
```

#### Define a Template Prompt


```python
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

template = [
    ChatMessage.from_user(
        """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
    )
]

prompt_builder = ChatPromptBuilder(template=template)
```

#### Initialize a ChatGenerator


```python
from haystack.components.generators.chat import OpenAIChatGenerator

chat_generator = OpenAIChatGenerator(model="gpt-4o-mini")
```

#### Setting up the RagasEvaluator

Pass all the Ragas metrics you want to use for evaluation, ensuring that all the necessary information to calculate each selected metric is provided.

For example:

- **AnswerRelevancy**: requires both the **query** and the **response**.
- **ContextPrecision**: requires the **query**, **retrieved documents**, and the **reference**.
- **Faithfulness**: requires the **query**, **retrieved documents**, and the **response**.

Make sure to include all relevant data for each metric to ensure accurate evaluation.


```python
from haystack_integrations.components.evaluators.ragas import RagasEvaluator
from langchain_openai import ChatOpenAI

from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

llm = ChatOpenAI(model="gpt-4o-mini")
evaluator_llm = LangchainLLMWrapper(llm)

ragas_evaluator = RagasEvaluator(
    ragas_metrics=[AnswerRelevancy(), ContextPrecision(), Faithfulness()],
    evaluator_llm=evaluator_llm,
)
```

## Building and Assembling the Pipeline

#### Creating the Pipeline


```python
from haystack import Pipeline

rag_pipeline = Pipeline()
```

#### Adding the components


```python
from haystack.components.builders import AnswerBuilder

rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", chat_generator)
rag_pipeline.add_component("answer_builder", AnswerBuilder())
rag_pipeline.add_component("ragas_evaluator", ragas_evaluator)
```

#### Connecting the components


```python
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever", "prompt_builder")
rag_pipeline.connect("prompt_builder.prompt", "llm.messages")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("retriever", "answer_builder.documents")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("retriever", "answer_builder.documents")
rag_pipeline.connect("retriever", "ragas_evaluator.documents")
rag_pipeline.connect("llm.replies", "ragas_evaluator.response")
```




    <haystack.core.pipeline.pipeline.Pipeline object at 0x14b20fad0>
    🚅 Components
      - text_embedder: OpenAITextEmbedder
      - retriever: InMemoryEmbeddingRetriever
      - prompt_builder: ChatPromptBuilder
      - llm: OpenAIChatGenerator
      - answer_builder: AnswerBuilder
      - ragas_evaluator: RagasEvaluator
    🛤️ Connections
      - text_embedder.embedding -> retriever.query_embedding (List[float])
      - retriever.documents -> prompt_builder.documents (List[Document])
      - retriever.documents -> answer_builder.documents (List[Document])
      - retriever.documents -> ragas_evaluator.documents (List[Document])
      - prompt_builder.prompt -> llm.messages (List[ChatMessage])
      - llm.replies -> answer_builder.replies (List[ChatMessage])
      - llm.replies -> ragas_evaluator.response (List[ChatMessage])



## Running the Pipeline


```python
question = "What makes Meta AI’s LLaMA models stand out?"

reference = "Meta AI’s LLaMA models stand out for being open-source, supporting innovation and experimentation due to their accessibility and strong performance."


result = rag_pipeline.run(
    {
        "text_embedder": {"text": question},
        "prompt_builder": {"question": question},
        "answer_builder": {"query": question},
        "ragas_evaluator": {"query": question, "reference": reference},
        # Each metric expects a specific set of parameters as input. Refer to the
        # Ragas class' documentation for more details.
    }
)

print(result["answer_builder"]["answers"][0].data, "\n")
print(result["ragas_evaluator"]["result"])
```

    Evaluating: 100%|██████████| 3/3 [00:14<00:00,  4.72s/it]


    Meta AI's LLaMA models stand out due to their open-source nature, which allows researchers and developers easy access to high-quality language models without the need for expensive resources. This accessibility fosters innovation and experimentation, enabling collaboration across various industries. Moreover, the strong performance of the LLaMA models further enhances their appeal, making them valuable tools for advancing AI development. 
    
    {'answer_relevancy': 0.9782, 'context_precision': 1.0000, 'faithfulness': 1.0000}


## Advance Usage

Instead of using the default ragas metrics, you can change them to fit your needs or even create your own custom metrics. After that, you can pass these to the RagasEvaluator component. To learn more about how to customize ragas metrics, check out the [docs](https://docs.ragas.io/en/stable/howtos/customizations/).

In the example below, we will define two custom Ragas metrics:

1. **SportsRelevanceMetric**: This metric evaluates whether a question and its response are related to sports.
2. **AnswerQualityMetric**: This metric measures how well the response provided by the LLM answers the user's question.


```python
from ragas.metrics import AspectCritic, RubricsScore

SportsRelevanceMetric = AspectCritic(
    name="sports_relevance_metric",
    definition="Were the question and response related to sports?",
    llm=evaluator_llm,
)

rubrics = {
    "score1_description": "The response does not answer the user input.",
    "score2_description": "The response partially answers the user input.",
    "score3_description": "The response fully answer the user input",
}

evaluator = RagasEvaluator(
    ragas_metrics=[
        SportsRelevanceMetric,
        RubricsScore(llm=evaluator_llm, rubrics=rubrics),
    ],
    evaluator_llm=evaluator_llm,
)

output = evaluator.run(
    query="Which is the most popular global sport?",
    documents=[
        "Football is undoubtedly the world's most popular sport with"
        " major events like the FIFA World Cup and sports personalities"
        " like Ronaldo and Messi, drawing a followership of more than 4"
        " billion people."
    ],
    response="Football is the most popular sport with around 4 billion"
    " followers worldwide",
)

output["result"]
```

    Evaluating: 100%|██████████| 2/2 [00:01<00:00,  1.62it/s]





    {'sports_relevance_metric': 1.0000, 'domain_specific_rubrics': 3.0000}


