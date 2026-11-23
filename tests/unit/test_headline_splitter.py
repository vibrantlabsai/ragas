import pytest

from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms.engine import apply_transforms
from ragas.testset.transforms.splitters import HeadlineSplitter


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ("Section one two", ["Intro\n", "Section one two"]),
        ("Section one two three", ["Intro\n", "Section one two", "three"]),
        (
            "Section one two three four five",
            ["Intro\n", "Section one two", "three four five"],
        ),
    ],
)
def test_short_intro_precedes_section_chunks(monkeypatch, section, expected):
    monkeypatch.setattr(
        "ragas.testset.transforms.splitters.headline.num_tokens_from_string",
        lambda text: len(text.split()),
    )
    document = Node(
        type=NodeType.DOCUMENT,
        properties={"page_content": f"Intro\n{section}", "headlines": ["Section"]},
    )
    graph = KnowledgeGraph(nodes=[document])

    apply_transforms(graph, HeadlineSplitter(min_tokens=2, max_tokens=3))

    chunks = [node for node in graph.nodes if node.type == NodeType.CHUNK]
    assert [node.get_property("page_content") for node in chunks] == expected
    next_edges = [(r.source, r.target) for r in graph.relationships if r.type == "next"]
    assert next_edges == list(zip(chunks, chunks[1:]))
