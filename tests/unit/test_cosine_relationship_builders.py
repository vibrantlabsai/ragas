import copy
import random
from typing import Optional
from uuid import UUID

import numpy as np
import pytest

from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
from ragas.testset.transforms.relationship_builders.cosine import (
    CosineSimilarityBuilder,
    SummaryCosineSimilarityBuilder,
)


def generate_test_vectors(
    n: int = 16,
    d: int = 32,
    min_similarity: float = 0.5,
    similar_fraction: float = 0.3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate `n` unit vectors of dimension `d`, where at least `similar_fraction` of them
    are similar to each other (cosine similarity > `min_similarity`), and the result is shuffled.

    Parameters:
    - n (int): Total number of vectors to generate.
    - d (int): Dimensionality of each vector.
    - min_similarity (float): Minimum cosine similarity for similar pairs.
    - similar_fraction (float): Fraction (0-1) of vectors that should be similar.
    - seed (int): Optional random seed for reproducibility.

    Returns:
    - np.ndarray: Array of shape (n, d) of unit vectors.
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    num_similar = max(2, int(n * similar_fraction))  # at least two similar vectors
    num_random = n - num_similar

    # Step 1: Create a base vector
    base = np.random.randn(d)
    base /= np.linalg.norm(base)

    # Step 2: Generate similar vectors
    similar_vectors = [base]
    angle = np.arccos(min_similarity)

    for _ in range(num_similar - 1):
        perturbation = np.random.randn(d)
        perturbation -= perturbation.dot(base) * base  # make orthogonal
        perturbation /= np.linalg.norm(perturbation)

        similar_vec = np.cos(angle * 0.9) * base + np.sin(angle * 0.9) * perturbation
        similar_vec /= np.linalg.norm(similar_vec)
        similar_vectors.append(similar_vec)

    # Step 3: Generate additional random unit vectors
    random_vectors = []
    for _ in range(num_random):
        v = np.random.randn(d)
        v /= np.linalg.norm(v)
        random_vectors.append(v)

    # Step 4: Combine and shuffle
    all_vectors = similar_vectors + random_vectors
    random.shuffle(all_vectors)

    return np.stack(all_vectors)


def cosine_similarity_matrix(embeddings: np.ndarray):
    """Calculate cosine similarity matrix for a set of embeddings."""
    from scipy.spatial.distance import cdist

    similarity = 1 - cdist(embeddings, embeddings, metric="cosine")

    # normalized = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    # similarity = np.dot(normalized, normalized.T)
    return similarity


def cosine_similarity_pair(embeddings: np.ndarray, threshold: float):
    """Find pairs of embeddings with cosine similarity >= threshold."""
    # Find pairs with similarity >= threshold
    similarity_matrix = cosine_similarity_matrix(embeddings)
    similar_pairs = np.argwhere(similarity_matrix >= threshold)

    # Filter out self-comparisons and duplicate pairs
    return [
        (int(pair[0]), int(pair[1]), float(similarity_matrix[pair[0], pair[1]]))
        for pair in similar_pairs
        if pair[0] < pair[1]
    ]


def vector_cosine_similarity(a, b):
    """Find pairwise cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@pytest.fixture
def simple_kg():
    # Arrange: create a simple knowledge graph with embeddings
    # roughly, we expect the following relationships:
    # 1 <-> 2 (0.1928 similarity)
    # 2 <-> 3 (0.6520 similarity)
    # 1 <-> 3 (0.8258 similarity)
    nodes = [
        Node(
            id=UUID("4da47a69-539c-49a2-b289-01780989d82c"),
            type=NodeType.DOCUMENT,
            properties={
                "embedding": [0.2313, -0.362, 0.5875, -0.0526, -0.0954],
                "summary_embedding": [0.2313, -0.362, 0.5875, -0.0526, -0.0954],
            },
        ),
        Node(
            id=UUID("f353e5c2-e432-4d1e-84a8-d750c93d4edf"),
            type=NodeType.DOCUMENT,
            properties={
                "embedding": [0.9066, 0.786, 0.6925, 0.8022, 0.5297],
                "summary_embedding": [0.9066, 0.786, 0.6925, 0.8022, 0.5297],
            },
        ),
        Node(
            id=UUID("437c8c08-cef6-4ebf-a35f-93d6168b61a4"),
            type=NodeType.DOCUMENT,
            properties={
                "embedding": [0.5555, -0.1074, 0.8454, 0.3499, -0.1669],
                "summary_embedding": [0.5555, -0.1074, 0.8454, 0.3499, -0.1669],
            },
        ),
    ]
    return KnowledgeGraph(nodes=nodes)


# node order
# UUID("4da47a69-539c-49a2-b289-01780989d82c")
# UUID("f353e5c2-e432-4d1e-84a8-d750c93d4edf")
# UUID("437c8c08-cef6-4ebf-a35f-93d6168b61a4")


@pytest.mark.parametrize(
    "n_test_embeddings",
    [
        (16),
        (256),
        (1024),
    ],
)
def test__cosine_similarity(n_test_embeddings):
    """
    Validate that the cosine similarity function correctly computes pairwise similarities
    and that the results match expected values.
    """

    threshold = 0.7
    embeddings = generate_test_vectors(
        n=n_test_embeddings,
        d=64,
        min_similarity=min(threshold + 0.025, 1.0),
        similar_fraction=0.3,
    )
    expected = cosine_similarity_matrix(embeddings)

    builder = CosineSimilarityBuilder(property_name="embedding", threshold=threshold)
    result = builder._block_cosine_similarity(embeddings, embeddings)

    assert result.shape == expected.shape, "Result shape does not match expected shape"
    assert np.allclose(result, expected, atol=1e-5), (
        "Cosine similarity does not match expected values"
    )


# Test for the internal _find_similar_embedding_pairs method
@pytest.mark.parametrize(
    "n_test_embeddings, threshold, block_size",
    [
        (16, 0.5, 16),
        (16, 0.7, 16),
        (16, 0.9, 16),
        (16, 0.7, 32),  # block size >> n_test_embeddings
        (16, 0.7, 37),  # block size >> n_test_embeddings
        (32, 0.7, 16),  # block size 1/2 n_test_embeddings
        (37, 0.7, 4),  # block size doesn't shard evenly
    ],
)
def test__find_similar_embedding_pairs(n_test_embeddings, threshold, block_size):
    """Validate that _find_similar_embedding_pairs correctly identifies pairs when compared with scipy's cosine distance."""

    embeddings = generate_test_vectors(
        n=n_test_embeddings,
        d=64,
        min_similarity=min(threshold + 0.025, 1.0),
        similar_fraction=0.3,
    )
    expected = cosine_similarity_pair(embeddings, threshold)

    builder = CosineSimilarityBuilder(
        property_name="embedding", threshold=threshold, block_size=block_size
    )
    result = builder._find_similar_embedding_pairs(embeddings, threshold=threshold)

    assert len(result) == len(expected)

    for i, j, similarity_float in result:
        assert i < j, "Pairs should be ordered (i < j)"
        assert similarity_float >= threshold, (
            f"Similarity {similarity_float} should be >= {threshold}"
        )
        for x, y, expected_similarity in expected:
            if i == x and j == y:
                assert similarity_float == pytest.approx(expected_similarity), (
                    "Cosine similarity does not match expected value"
                )

                break


class TestCosineSimilarityBuilder:
    @pytest.mark.asyncio
    async def test_no_self_similarity_relationships(self, simple_kg):
        builder = CosineSimilarityBuilder(property_name="embedding", threshold=0.1)
        relationships = await builder.transform(copy.deepcopy(simple_kg))
        for r in relationships:
            assert r.source.id != r.target.id, (
                "Self-relationships should not be created"
            )

    @pytest.mark.asyncio
    async def test_no_duplicate_relationships(self, simple_kg):
        builder = CosineSimilarityBuilder(property_name="embedding", threshold=0.1)
        relationships = await builder.transform(copy.deepcopy(simple_kg))
        seen = set()
        for r in relationships:
            pair = tuple(sorted([r.source.id, r.target.id]))
            assert pair not in seen, "Duplicate relationships found"
            seen.add(pair)

    @pytest.mark.asyncio
    async def test_similarity_at_threshold(self):
        node1 = Node(type=NodeType.CHUNK, properties={"embedding": [1, 0, 0]})
        node2 = Node(type=NodeType.CHUNK, properties={"embedding": [1, 0, 0]})
        kg = KnowledgeGraph(nodes=[node1, node2])
        builder = CosineSimilarityBuilder(property_name="embedding", threshold=1.0)
        relationships = await builder.transform(kg)
        assert len(relationships) == 1, "Should create relationship at threshold"

    @pytest.mark.asyncio
    async def test_all_below_threshold(self):
        node1 = Node(type=NodeType.CHUNK, properties={"embedding": [1, 0, 0]})
        node2 = Node(type=NodeType.CHUNK, properties={"embedding": [-1, 0, 0]})
        kg = KnowledgeGraph(nodes=[node1, node2])
        builder = CosineSimilarityBuilder(property_name="embedding", threshold=0.5)
        relationships = await builder.transform(kg)
        assert len(relationships) == 0, (
            "No relationships should be created below threshold"
        )

    @pytest.mark.asyncio
    async def test_all_above_threshold(self):
        node1 = Node(type=NodeType.CHUNK, properties={"embedding": [1, 0, 0]})
        node2 = Node(type=NodeType.CHUNK, properties={"embedding": [1, 0, 0]})
        node3 = Node(type=NodeType.CHUNK, properties={"embedding": [1, 0, 0]})
        kg = KnowledgeGraph(nodes=[node1, node2, node3])
        builder = CosineSimilarityBuilder(property_name="embedding", threshold=0.9)
        relationships = await builder.transform(kg)
        assert len(relationships) == 3

    @pytest.mark.asyncio
    async def test_malformed_embedding_raises(self):
        node1 = Node(type=NodeType.CHUNK, properties={"embedding": [1, 0, 0]})
        node2 = Node(type=NodeType.CHUNK, properties={"embedding": ["a", 0, 0]})
        kg = KnowledgeGraph(nodes=[node1, node2])
        builder = CosineSimilarityBuilder(property_name="embedding", threshold=0.5)
        with pytest.raises(Exception):
            await builder.transform(kg)

    @pytest.mark.asyncio
    async def test_cosine_similarity_builder_empty_graph(self):
        kg = KnowledgeGraph(nodes=[])
        builder = CosineSimilarityBuilder(property_name="embedding")
        with pytest.raises(ValueError):
            await builder.transform(kg)

    @pytest.mark.asyncio
    async def test_cosine_similarity_builder_basic(self, simple_kg):
        # Act
        builder = CosineSimilarityBuilder(property_name="embedding", threshold=0.5)
        relationships = await builder.transform(simple_kg)
        # Assert
        assert all(isinstance(r, Relationship) for r in relationships)
        assert all(r.type == "cosine_similarity" for r in relationships)
        # 2 <-> 3 (~0.6520 similarity)
        assert any(
            str(r.source.id) == "f353e5c2-e432-4d1e-84a8-d750c93d4edf"
            and str(r.target.id) == "437c8c08-cef6-4ebf-a35f-93d6168b61a4"
            for r in relationships
        )
        # 1 <-> 3 (~0.8258 similarity)
        assert any(
            str(r.source.id) == "4da47a69-539c-49a2-b289-01780989d82c"
            and str(r.target.id) == "437c8c08-cef6-4ebf-a35f-93d6168b61a4"
            for r in relationships
        )

    @pytest.mark.asyncio
    async def test_cosine_similarity_builder_no_embeddings(self):
        kg = KnowledgeGraph(
            nodes=[
                Node(type=NodeType.DOCUMENT, properties={}),
                Node(type=NodeType.DOCUMENT, properties={}),
            ]
        )
        builder = CosineSimilarityBuilder(property_name="embedding")
        with pytest.raises(ValueError, match="has no embedding"):
            await builder.transform(kg)

    @pytest.mark.asyncio
    async def test_cosine_similarity_builder_shape_validation(self):
        kg = KnowledgeGraph(
            nodes=[
                Node(type=NodeType.DOCUMENT, properties={"embedding": [1.0, 0.0]}),
                Node(
                    type=NodeType.DOCUMENT,
                    properties={"embedding": [0.0, 1.0, 2.0]},
                ),
            ]
        )
        builder = CosineSimilarityBuilder(property_name="embedding")
        with pytest.raises(
            ValueError, match="Embedding at index 1 has length 3, expected 2"
        ):
            await builder.transform(kg)

    @pytest.mark.asyncio
    async def test_apply_transforms_cosine_similarity_builder(self, simple_kg):
        from ragas.run_config import RunConfig
        from ragas.testset.transforms.engine import apply_transforms

        # CosineSimilarityBuilder should add relationships to the graph
        builder = CosineSimilarityBuilder(property_name="embedding", threshold=0.5)
        kg = simple_kg
        # Should mutate kg in-place
        apply_transforms(kg, builder, run_config=RunConfig(max_workers=2))
        # Check that relationships were added
        assert any(r.type == "cosine_similarity" for r in kg.relationships), (
            "No cosine_similarity relationships found after apply_transforms"
        )
        # Check that expected relationship exists
        assert any(
            str(r.source.id) == "f353e5c2-e432-4d1e-84a8-d750c93d4edf"
            and str(r.target.id) == "437c8c08-cef6-4ebf-a35f-93d6168b61a4"
            for r in kg.relationships
        )
        # 1 <-> 3 (~0.8258 similarity)
        assert any(
            str(r.source.id) == "4da47a69-539c-49a2-b289-01780989d82c"
            and str(r.target.id) == "437c8c08-cef6-4ebf-a35f-93d6168b61a4"
            for r in kg.relationships
        )


class TestSummaryCosineSimilarityBuilder:
    @pytest.mark.asyncio
    async def test_summary_cosine_similarity_builder_basic(self, simple_kg):
        builder = SummaryCosineSimilarityBuilder(
            property_name="summary_embedding", threshold=0.5
        )
        relationships = await builder.transform(simple_kg)
        assert all(isinstance(r, Relationship) for r in relationships)
        assert all(r.type == "summary_cosine_similarity" for r in relationships)
        assert any(
            str(r.source.id) == "f353e5c2-e432-4d1e-84a8-d750c93d4edf"
            and str(r.target.id) == "437c8c08-cef6-4ebf-a35f-93d6168b61a4"
            for r in relationships
        )
        assert any(
            str(r.source.id) == "4da47a69-539c-49a2-b289-01780989d82c"
            and str(r.target.id) == "437c8c08-cef6-4ebf-a35f-93d6168b61a4"
            for r in relationships
        )

    @pytest.mark.asyncio
    async def test_summary_cosine_similarity_only_document_nodes(self):
        node1 = Node(
            type=NodeType.DOCUMENT, properties={"summary_embedding": [1, 0, 0]}
        )
        node2 = Node(type=NodeType.CHUNK, properties={"summary_embedding": [1, 0, 0]})
        kg = KnowledgeGraph(nodes=[node1, node2])
        builder = SummaryCosineSimilarityBuilder(
            property_name="summary_embedding", threshold=0.5
        )
        relationships = await builder.transform(kg)
        assert len(relationships) == 0

    @pytest.mark.asyncio
    async def test_summary_cosine_similarity_builder_filter_and_error(self):
        kg = KnowledgeGraph(nodes=[Node(type=NodeType.DOCUMENT, properties={})])
        builder = SummaryCosineSimilarityBuilder(property_name="summary_embedding")
        with pytest.raises(ValueError, match="has no summary_embedding"):
            await builder.transform(kg)


@pytest.mark.asyncio
async def test_apply_transforms_summary_cosine_similarity_builder(simple_kg):
    from ragas.run_config import RunConfig
    from ragas.testset.transforms.engine import apply_transforms

    builder = SummaryCosineSimilarityBuilder(
        property_name="summary_embedding", threshold=0.5
    )
    kg = simple_kg
    apply_transforms(kg, builder, run_config=RunConfig(max_workers=2))
    assert any(r.type == "summary_cosine_similarity" for r in kg.relationships), (
        "No summary_cosine_similarity relationships found after apply_transforms"
    )
    assert any(
        str(r.source.id) == "f353e5c2-e432-4d1e-84a8-d750c93d4edf"
        and str(r.target.id) == "437c8c08-cef6-4ebf-a35f-93d6168b61a4"
        for r in kg.relationships
    )
    # 1 <-> 3 (~0.8258 similarity)
    assert any(
        str(r.source.id) == "4da47a69-539c-49a2-b289-01780989d82c"
        and str(r.target.id) == "437c8c08-cef6-4ebf-a35f-93d6168b61a4"
        for r in kg.relationships
    )


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("chunk_embedding", [None, [1.0, 0.0], [1.0]])
def test_summary_similarity_pipeline_keeps_document_endpoints(
    parallel, chunk_embedding
):
    """Scheduling must preserve the subclass's document-only graph semantics."""
    import asyncio

    from ragas.testset.transforms.engine import Parallel, apply_transforms

    first = Node(type=NodeType.DOCUMENT, properties={"summary_embedding": [1.0, 0.0]})
    chunk = Node(type=NodeType.CHUNK, properties={})
    if chunk_embedding is not None:
        chunk.properties["summary_embedding"] = chunk_embedding
    second = Node(type=NodeType.DOCUMENT, properties={"summary_embedding": [1.0, 0.0]})
    kg = KnowledgeGraph(nodes=[first, chunk, second])
    previous = Relationship(source=first, target=chunk, type="child")
    kg.relationships.append(previous)
    builder = SummaryCosineSimilarityBuilder(threshold=0.5)
    expected = asyncio.run(builder.transform(kg))

    apply_transforms(kg, Parallel(builder) if parallel else builder)

    assert kg.relationships[0] is previous
    assert len(kg.relationships) == len(expected) + 1 == 2
    actual = kg.relationships[1]
    assert actual.source is first
    assert actual.target is second
    assert actual.type == expected[0].type
    assert actual.properties == expected[0].properties
    assert actual.bidirectional
    assert kg.nodes == [first, chunk, second]


def test_summary_similarity_plan_applies_custom_filter_before_validation():
    import asyncio

    from ragas.testset.transforms.engine import apply_transforms

    first = Node(type=NodeType.DOCUMENT, properties={"custom": [1.0, 0.0]})
    rejected = Node(type=NodeType.DOCUMENT, properties={})
    chunk = Node(type=NodeType.CHUNK, properties={})
    second = Node(type=NodeType.DOCUMENT, properties={"custom": [1.0, 0.0]})
    kg = KnowledgeGraph(nodes=[first, rejected, chunk, second])
    builder = SummaryCosineSimilarityBuilder(
        property_name="custom",
        new_property_name="document_similarity",
        threshold=0.5,
        filter_nodes=lambda node: node.id != rejected.id,
    )
    expected = asyncio.run(builder.transform(builder.filter(kg)))
    apply_transforms(kg, builder)
    assert len(kg.relationships) == len(expected) == 1
    relationship = kg.relationships[0]
    assert relationship.source is first and relationship.target is second
    assert relationship.type == "document_similarity"
    assert relationship.properties == expected[0].properties


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_document", [False, True])
async def test_invalid_summary_plan_does_not_append_relationships(missing_document):
    first = Node(type=NodeType.DOCUMENT, properties={"summary_embedding": [1.0, 0.0]})
    chunk = Node(type=NodeType.CHUNK, properties={"summary_embedding": [1.0, 0.0]})
    invalid = Node(type=NodeType.DOCUMENT, properties={})
    kg = KnowledgeGraph(nodes=[first, chunk, invalid] if missing_document else [chunk])
    prior = Relationship(source=first, target=chunk, type="child")
    kg.relationships.append(prior)
    builder = SummaryCosineSimilarityBuilder()
    # Both public paths must report the same invalid selected document set.
    message = (
        "has no summary_embedding" if missing_document else "No nodes have a valid"
    )
    with pytest.raises(ValueError, match=message):
        await builder.transform(kg)
    with pytest.raises(ValueError, match=message):
        plan = builder.generate_execution_plan(kg)
        for coroutine in plan:
            await coroutine
    assert kg.relationships == [prior]
    assert kg.relationships[0] is prior
