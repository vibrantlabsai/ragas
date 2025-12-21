# Using Pre-chunked Data for Testset Generation

When you already have a well-defined chunking strategy in place, Ragas allows you to bypass its internal document splitting mechanism and use your own chunks directly. This is particularly useful when:

- You've optimized your chunking strategy for your specific domain
- You want to maintain consistency between your RAG pipeline and evaluation
- You have pre-processed documents with custom metadata
- You need to ensure chunks align with specific business logic or document structure

## Overview

The `generate_with_chunks` method of `TestsetGenerator` accepts pre-chunked data and treats each chunk as a `NodeType.CHUNK` directly, skipping the internal splitting transforms. This means your chunks remain exactly as you provide them, preserving both content and metadata integrity.

## How It Works

When you use `generate_with_chunks`, Ragas:

1. **Accepts your chunks** as-is (either as `Document` objects or strings)
2. **Applies extractors** like `SummaryExtractor`, `ThemesExtractor`, `NERExtractor`, and `EmbeddingExtractor` to enrich each chunk with additional properties
3. **Builds relationships** between chunks using `CosineSimilarityBuilder` and `OverlapScoreBuilder`
4. **Generates personas** based on the content themes
5. **Creates scenarios** for different query types (single-hop, multi-hop)
6. **Synthesizes test samples** including questions, contexts, and reference answers

## Example: Using Pre-chunked Documents

You can pass a list of LangChain `Document` objects. This approach preserves the metadata of your chunks, which can be useful for tracking source documents or other custom information.

```python
import os
from langchain_core.documents import Document
from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize generator with your preferred models
generator = TestsetGenerator(
    llm=llm_factory("gpt-4o-mini", client=client),
    embedding_model=OpenAIEmbeddings(client=client)
)

# Your pre-chunked documents
chunks = [
    Document(
        page_content="""The Eiffel Tower (Tour Eiffel) is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Locally nicknamed "La Dame de Fer" (French for "The Iron Lady"), it was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair. Although initially criticized by some of France's leading artists and intellectuals for its design, it has since become a global cultural icon of France and one of the most recognizable structures in the world.""", 
        metadata={"source": "doc1", "chunk_id": 1}
    ),
    Document(
        page_content="""The tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft).""", 
        metadata={"source": "doc1", "chunk_id": 2}
    )
]

# Generate testset
testset = generator.generate_with_chunks(
    chunks=chunks,
    testset_size=10
)

# Save to CSV
output_file = "testset.csv"
testset.to_csv(output_file)
print(f"Testset saved to {output_file}")
print(testset.to_pandas().head())
```

### Generation Process

During generation, you'll see progress logs showing the various transformation and synthesis stages:

```
Applying SummaryExtractor: 100%|████████████████████████████████| 2/2 [00:07<00:00,  3.67s/it]
Applying CustomNodeFilter: 100%|█████████████████████████████| 2/2 [00:00<00:00, 2226.87it/s]
Applying EmbeddingExtractor: 100%|███████████████████████████| 2/2 [00:02<00:00,  1.19s/it]
Applying ThemesExtractor: 100%|██████████████████████████████| 2/2 [00:06<00:00,  3.07s/it]
Applying NERExtractor: 100%|█████████████████████████████████| 2/2 [00:06<00:00,  3.10s/it]
Applying CosineSimilarityBuilder: 100%|█████████████████████| 1/1 [00:00<00:00, 613.29it/s]
Applying OverlapScoreBuilder: 100%|████████████████████████| 1/1 [00:00<00:00, 1491.57it/s]
Generating personas: 100%|███████████████████████████████████| 2/2 [00:05<00:00,  2.77s/it]
Generating Scenarios: 100%|██████████████████████████████████| 2/2 [00:08<00:00,  4.19s/it]
Generating Samples: 100%|████████████████████████████████| 11/11 [00:45<00:00,  4.13s/it]
Testset saved to testset.csv
```


The testset includes different types of queries:
- **Single-hop queries**: Questions that can be answered from a single chunk
- **Multi-hop queries**: Questions requiring information from multiple chunks (when relationships exist)

## Example: Using Plain Strings

If you don't need to preserve metadata, you can also pass plain strings directly:

```python
from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from openai import OpenAI

# Initialize models
client = OpenAI()
generator = TestsetGenerator(
    llm=llm_factory("gpt-4o-mini", client=client),
    embedding_model=OpenAIEmbeddings(client=client)
)

# Simple text chunks
text_chunks = [
    "Artificial Intelligence (AI) is the simulation of human intelligence by machines. It involves machine learning, natural language processing, and computer vision.",
    "Machine Learning is a subset of AI that enables systems to learn from data without explicit programming. Popular algorithms include neural networks and decision trees.",
    "Deep Learning uses neural networks with multiple layers to process complex patterns in large datasets. It powers modern applications like image recognition and language translation."
]

# Generate testset
testset = generator.generate_with_chunks(
    chunks=text_chunks,
    testset_size=5
)

# Save to CSV
output_file = "testset.csv"
testset.to_csv(output_file)
print(f"Testset saved to {output_file}")
print(testset.to_pandas())
```

## Handling Edge Cases

- **Empty Content**: Chunks with empty or whitespace-only `page_content` will be automatically filtered out.
- **Empty Sequence**: If you provide an empty sequence of chunks, the generation will produce an empty testset.
