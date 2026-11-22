import pytest

from ragas.testset.graph import Node, NodeType
from ragas.testset.transforms.splitters import HeadlineSplitter


@pytest.mark.asyncio
async def test_headline_splitter_missing_property_small_doc():
    # Case 1: Small document (< min_tokens) with missing headlines
    node = Node(
        type=NodeType.DOCUMENT,
        properties={
            "page_content": "This is a small document without headlines."
        }
    )
    splitter = HeadlineSplitter(min_tokens=500)
    nodes, relationships = await splitter.split(node)
    
    assert len(nodes) == 1
    assert nodes[0] == node

@pytest.mark.asyncio
async def test_headline_splitter_missing_property_large_doc():
    # Case 2: Large document (> min_tokens) with missing headlines
    # Should fallback to splitting by chunks (legacy behavior)
    long_content = "word " * 600 # Approx 600 tokens
    node = Node(
        type=NodeType.DOCUMENT,
        properties={
            "page_content": long_content
        }
    )
    splitter = HeadlineSplitter(min_tokens=500)
    nodes, relationships = await splitter.split(node)
    
    # Should contain at least 1 node (chunked or original if not large enough for max_tokens)
    assert len(nodes) >= 1

@pytest.mark.asyncio
async def test_headline_splitter_with_headlines():
    # Case 3: Document with headlines present
    content = "Section 1\nContent.\nSection 2\nContent."
    node = Node(
        type=NodeType.DOCUMENT,
        properties={
            "page_content": content,
            "headlines": ["Section 1", "Section 2"]
        }
    )
    splitter = HeadlineSplitter(min_tokens=2) # Low threshold to force split
    nodes, relationships = await splitter.split(node)
    
    # Should be split into chunks
    assert len(nodes) > 1
