# Testset Validation with Pre-chunked Data

Ragas allows you to use your own chunks for testset generation, bypassing the internal splitting mechanism. This is useful when you already have a chunking strategy in place and want to evaluate using those specific chunks.

## Using Pre-chunked Data

You can use the `generate_with_chunks` method of `TestsetGenerator` to provide your own documents or strings as chunks. These will be treated directly as `NodeType.CHUNK` and will not be split further.

### Example with Documents

You can pass a list of LangChain `Document` objects. This preserves the metadata of your chunks.

```python
from langchain_core.documents import Document
from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.llms import MockLLM
from ragas.embeddings import MockEmbeddings

# Initialize generator
generator = TestsetGenerator(
    llm=MockLLM(), 
    embedding_model=MockEmbeddings()
)

# Your pre-chunked documents
chunks = [
    Document(
        page_content="This is the content of the first chunk.", 
        metadata={"source": "doc1", "chunk_id": 1}
    ),
    Document(
        page_content="This is the content of the second chunk.", 
        metadata={"source": "doc1", "chunk_id": 2}
    )
]

# Generate testset
testset = generator.generate_with_chunks(
    chunks=chunks,
    testset_size=10
)
```

### Example with Strings

You can also pass a list of strings directly. Ragas will create chunks with empty metadata for these.

```python
# Your pre-chunked strings
chunks = [
    "This is the first chunk.",
    "This is the second chunk."
]

# Generate testset
testset = generator.generate_with_chunks(
    chunks=chunks,
    testset_size=5
)
```

## Handling Edge Cases

- **Empty Content**: Chunks with empty or whitespace-only `page_content` will be automatically filtered out.
- **Empty Sequence**: If you provide an empty sequence of chunks, the generation will produce an empty testset.
