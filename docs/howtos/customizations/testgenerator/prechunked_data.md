# Testset Validation with Pre-chunked Data

Ragas allows you to use your own chunks for testset generation, bypassing the internal splitting mechanism. This is useful when you already have a chunking strategy in place and want to evaluate using those specific chunks.

## Using Pre-chunked Data

You can use the `generate_with_chunks` method of `TestsetGenerator` to provide your own documents or strings as chunks. These will be treated directly as `NodeType.CHUNK` and will not be split further.

### Example with Documents

You can pass a list of LangChain `Document` objects. This preserves the metadata of your chunks.

```python
import os
from langchain_core.documents import Document
from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize generator
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

## Handling Edge Cases

- **Empty Content**: Chunks with empty or whitespace-only `page_content` will be automatically filtered out.
- **Empty Sequence**: If you provide an empty sequence of chunks, the generation will produce an empty testset.
