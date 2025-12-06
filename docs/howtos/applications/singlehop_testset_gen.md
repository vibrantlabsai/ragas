# Generating a Synthetic Test Set for RAG-Based Question Answering with Ragas

## Overview

In this tutorial, we'll explore the **test set generation module in Ragas** to create a **synthetic test set** for a **Retrieval-Augmented Generation (RAG)-based question-answering bot**. Our goal is to design a **Ragas Airline Assistant** capable of answering customer queries on various topics, including:

- Flight booking
- Flight changes and cancellations
- Baggage policies
- Viewing reservations
- Flight delays
- In-flight services
- Special assistance

To make sure our synthetic dataset is as **realistic and diverse** as possible, we will create **different customer personas**. Each persona will represent distinct traveler types and behaviors, helping us build a **comprehensive and representative test set**. This approach ensures that we can thoroughly evaluate the effectiveness and robustness of our RAG model.

Let’s get started!

## Download and Load documents

Run the command below to download the dummy Ragas Airline dataset and load the documents using LangChain.

```sh
! git clone https://huggingface.co/datasets/vibrantlabsai/ragas-airline-dataset
```

```python
from langchain_community.document_loaders import DirectoryLoader

path = "ragas-airline-dataset"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()
```

## Set up the LLM and Embedding Model


```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import openai


generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
openai_client = openai.OpenAI()
generator_embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")
```

## Create Knowledge Graph

Create a base knowledge graph with the documents


```python
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType


kg = KnowledgeGraph()

for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )
    
kg
```
Output
```
KnowledgeGraph(nodes: 8, relationships: 0)
```


## Setup the transforms

In this tutorial, we create a Single Hop Query dataset using a knowledge graph built solely from nodes. To enhance our graph and improve query generation, we apply three key transformations:

- **Headline Extraction:** Uses a language model to extract clear section titles from each document (e.g., “Airline Initiated Cancellations” from *flight cancellations.md*). These titles isolate specific topics and provide direct context for generating focused questions.
- **Headline Splitting:** Divides documents into manageable subsections based on the extracted headlines. This increases the number of nodes and ensures more granular, context-specific query generation.
- **Keyphrase Extraction:** Identifies core thematic keyphrases (such as key seating information) that serve as semantic seed points, enriching the diversity and relevance of the generated queries.


```python
from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import HeadlinesExtractor, HeadlineSplitter, KeyphrasesExtractor

headline_extractor = HeadlinesExtractor(llm=generator_llm, max_num=20)
headline_splitter = HeadlineSplitter(max_tokens=1500)
keyphrase_extractor = KeyphrasesExtractor(llm=generator_llm)

transforms = [
    headline_extractor,
    headline_splitter,
    keyphrase_extractor
]

apply_transforms(kg, transforms=transforms)
```
```
Applying HeadlinesExtractor: 100%|██████████| 8/8 [00:00<?, ?it/s]
Applying HeadlineSplitter: 100%|██████████| 8/8 [00:00<?, ?it/s]
Applying KeyphrasesExtractor: 100%|██████████| 25/25 [00:00<?, ?it/s]
```

## Configuring Personas for Query Generation

Personas provide context and perspective, ensuring that generated queries are natural, user-specific, and diverse. By tailoring queries to different user viewpoints, our test set covers a wide range of scenarios:

- **First Time Flier:** Generates queries with detailed, step-by-step guidance, catering to newcomers who need clear instructions.
- **Frequent Flier:** Produces concise, efficiency-focused queries for experienced travelers.
- **Angry Business Class Flier:** Yields queries with a critical, urgent tone to reflect high expectations and immediate resolution demands.


```python
from ragas.testset.persona import Persona

persona_first_time_flier = Persona(
    name="First Time Flier",
    role_description="Is flying for the first time and may feel anxious. Needs clear guidance on flight procedures, safety protocols, and what to expect throughout the journey.",
)

persona_frequent_flier = Persona(
    name="Frequent Flier",
    role_description="Travels regularly and values efficiency and comfort. Interested in loyalty programs, express services, and a seamless travel experience.",
)

persona_angry_business_flier = Persona(
    name="Angry Business Class Flier",
    role_description="Demands top-tier service and is easily irritated by any delays or issues. Expects immediate resolutions and is quick to express frustration if standards are not met.",
)

personas = [persona_first_time_flier, persona_frequent_flier, persona_angry_business_flier]
```

## Query Generation Using Synthesizers

Synthesizers are responsible for converting enriched nodes and personas into queries. They achieve this by selecting a node property (e.g., "entities" or "keyphrases"), pairing it with a persona, style, and query length, and then using a LLM to generate a query-answer pair based on the content of the node.

Two instances of the `SingleHopSpecificQuerySynthesizer` are used to define the query distribution:

- **Headlines-Based Synthesizer** – Generates queries using extracted document headlines, leading to structured questions that reference specific sections.
- **Keyphrases-Based Synthesizer** – Forms queries around key concepts, generating broader, thematic questions.

Both synthesizers are weighted equally (0.5 each), ensuring a balanced mix of specific and conceptual queries, which ultimately enhances the diversity of the test set.


```python
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

query_distibution = [
    (
        SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="headlines"),
        0.5,
    ),
    (
        SingleHopSpecificQuerySynthesizer(
            llm=generator_llm, property_name="keyphrases"
        ),
        0.5,
    ),
]
```

## Testset Generation


```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
    knowledge_graph=kg,
    persona_list=personas,
)
```

Now we can generate the testset.


```python
testset = generator.generate(testset_size=10, query_distribution=query_distibution)
testset.to_pandas()
```
```
Generating Scenarios: 100%|██████████| 2/2 [00:00<?, ?it/s]
Generating Samples: 100%|██████████| 10/10 [00:00<?, ?it/s]
```
Output



<div>
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>reference_contexts</th>
      <th>reference</th>
      <th>synthesizer_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wut do I do if my baggage is Delayed, Lost, or...</td>
      <td>[Baggage Policies\n\nThis section provides a d...</td>
      <td>If your baggage is delayed, lost, or damaged, ...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Wht asistance is provided by the airline durin...</td>
      <td>[Flight Delays\n\nFlight delays can be caused ...</td>
      <td>Depending on the length of the delay, Ragas Ai...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What is Step 1: Check Fare Rules in the contex...</td>
      <td>[Flight Cancellations\n\nFlight cancellations ...</td>
      <td>Step 1: Check Fare Rules involves logging into...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>How can I access my booking online with Ragas ...</td>
      <td>[Managing Reservations\n\nManaging your reserv...</td>
      <td>To access your booking online with Ragas Airli...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>What assistance does Ragas Airlines provide fo...</td>
      <td>[Special Assistance\n\nRagas Airlines provides...</td>
      <td>Ragas Airlines provides special assistance ser...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>5</th>
      <td>What steps should I take if my baggage is dela...</td>
      <td>[Baggage Policies This section provides a deta...</td>
      <td>If your baggage is delayed, lost, or damaged w...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>6</th>
      <td>How can I resubmit the claim for my baggage is...</td>
      <td>[Potential Issues and Resolutions for Baggage ...</td>
      <td>To resubmit the claim for your baggage issue, ...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Wut are the main causes of flight delays and h...</td>
      <td>[Flight Delays Flight delays can be caused by ...</td>
      <td>Flight delays can be caused by weather conditi...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>8</th>
      <td>How can I request reimbursement for additional...</td>
      <td>[2. Additional Expenses Incurred Due to Delay ...</td>
      <td>To request reimbursement for additional expens...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>9</th>
      <td>What are passenger-initiated cancelations?</td>
      <td>[Flight Cancellations Flight cancellations can...</td>
      <td>Passenger-initiated cancellations occur when a...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
  </tbody>
</table>
</div>



## Final Thoughts

In this tutorial, we explored test set generation using the Ragas library, focusing primarily on single-hop queries. In our upcoming tutorial, we’ll dive into multi-hop queries, expanding on these concepts for even richer test set scenarios.
