import logging

import pytest

from ragas.llms import BaseRagasLLM
from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
from ragas.testset.transforms.filters import CustomNodeFilter, QuestionPotentialOutput


class MockLLM(BaseRagasLLM):
    def generate_text(self, *args, **kwargs):
        pass

    async def agenerate_text(self, *args, **kwargs):
        pass

    def is_finished(self, response):
        return True


class RecordingScoringPrompt:
    def __init__(self, score=3):
        self.score = score
        self.inputs = []
        self.llms = []

    async def generate(self, data, llm):
        self.inputs.append(data)
        self.llms.append(llm)
        return QuestionPotentialOutput(score=self.score)


def make_filter(scoring_prompt):
    return CustomNodeFilter(llm=MockLLM(), scoring_prompt=scoring_prompt)


@pytest.mark.asyncio
async def test_custom_node_filter_uses_chunk_summary_without_parent(caplog):
    prompt = RecordingScoringPrompt()
    node = Node(
        type=NodeType.CHUNK,
        properties={
            "summary": "chunk summary",
            "page_content": "chunk content",
        },
    )
    kg = KnowledgeGraph(nodes=[node])

    with caplog.at_level(logging.WARNING, logger="ragas.testset.transforms.filters"):
        await make_filter(prompt).custom_filter(node, kg)

    assert len(prompt.inputs) == 1
    assert prompt.inputs[0].document_summary == "chunk summary"
    assert prompt.inputs[0].node_content == "chunk content"
    assert "Skipping filtering" not in caplog.text


@pytest.mark.asyncio
async def test_custom_node_filter_prefers_parent_summary_for_chunk():
    prompt = RecordingScoringPrompt()
    parent = Node(
        type=NodeType.DOCUMENT,
        properties={"summary": "parent summary"},
    )
    node = Node(
        type=NodeType.CHUNK,
        properties={
            "summary": "chunk summary",
            "page_content": "chunk content",
        },
    )
    kg = KnowledgeGraph(
        nodes=[parent, node],
        relationships=[Relationship(type="child", source=parent, target=node)],
    )

    await make_filter(prompt).custom_filter(node, kg)

    assert len(prompt.inputs) == 1
    assert prompt.inputs[0].document_summary == "parent summary"


@pytest.mark.asyncio
async def test_custom_node_filter_falls_back_when_parent_summary_is_empty():
    prompt = RecordingScoringPrompt()
    parent = Node(type=NodeType.DOCUMENT, properties={"summary": ""})
    node = Node(
        type=NodeType.CHUNK,
        properties={
            "summary": "chunk summary",
            "page_content": "chunk content",
        },
    )
    kg = KnowledgeGraph(
        nodes=[parent, node],
        relationships=[Relationship(type="child", source=parent, target=node)],
    )

    await make_filter(prompt).custom_filter(node, kg)

    assert len(prompt.inputs) == 1
    assert prompt.inputs[0].document_summary == "chunk summary"


@pytest.mark.asyncio
async def test_custom_node_filter_skips_chunk_without_any_summary(caplog):
    prompt = RecordingScoringPrompt()
    node = Node(
        type=NodeType.CHUNK,
        properties={"page_content": "chunk content"},
    )
    kg = KnowledgeGraph(nodes=[node])

    with caplog.at_level(logging.WARNING, logger="ragas.testset.transforms.filters"):
        result = await make_filter(prompt).custom_filter(node, kg)

    assert result is False
    assert prompt.inputs == []
    assert "does not have a summary. Skipping filtering." in caplog.text
