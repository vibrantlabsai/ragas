import copy
import math
import random
import string
from typing import List, Set, Tuple
from uuid import UUID

import numpy as np
import pytest

from ragas.metrics._string import DistanceMeasure
from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
from ragas.testset.transforms.relationship_builders.traditional import (
    JaccardSimilarityBuilder,
    OverlapScoreBuilder,
)


def generate_test_sets(
    n: int = 16,
    max_len: int = 32,
    min_similarity: float = 0.5,
    similar_fraction: float = 0.3,
) -> List[Set[str]]:
    """
    Generate `n` sets up to `max_len`, where at least `similar_fraction` of all possible
    pairs have Jaccard similarity >= `min_similarity`. The result is shuffled.

    Parameters:
    - n (int): Total number of sets to generate.
    - max_len (int): Maximum length of each set.
    - min_similarity (float): Minimum Jaccard similarity for similar pairs.
    - similar_fraction (float): Fraction (0-1) of sets that should be similar.

    Returns:
    - list: List of generated sets.
    """

    if not (0 < min_similarity <= 1):
        raise ValueError("min_similarity must be between 0 and 1.")
    if not (0 <= similar_fraction <= 1):
        raise ValueError("similar_fraction must be between 0 and 1.")

    def generate_entity(k: int = 5) -> str:
        """Generate a random entity of length k."""
        return "".join(random.choices(string.ascii_lowercase, k=k))

    def jaccard(a: set[str], b: set[str]) -> float:
        from scipy.spatial.distance import jaccard as jaccard_dist

        # union of elements -> boolean indicator vectors
        elems = sorted(a | b)
        va = np.array([e in a for e in elems], dtype=bool)
        vb = np.array([e in b for e in elems], dtype=bool)
        # SciPy returns the Jaccard distance; similarity = 1 - distance
        return 1.0 - jaccard_dist(va, vb)

    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return [set() for _ in range(n)]

    target_similar_pairs = math.ceil(total_pairs * similar_fraction)

    if target_similar_pairs == 0:
        # Generate n random, dissimilar sets
        sets = []
        pool = {generate_entity() for _ in range(n * max_len)}
        for _ in range(n):
            length = random.randint(0, max_len)
            s = set(random.sample(list(pool), min(length, len(pool))))
            pool -= s
            sets.append(s)
        random.shuffle(sets)
        return sets

    # Calculate the size of a clique of similar sets needed
    # n_clique * (n_clique - 1) / 2 >= target_similar_pairs
    n_clique = math.ceil((1 + math.sqrt(1 + 8 * target_similar_pairs)) / 2)
    n_clique = min(n, n_clique)
    n_dissimilar = n - n_clique

    # To guarantee a given similarity, the size of the core set
    # and the number of unique elements added are constrained by the max_len.
    # We need cs + unique_per_set <= max_len.
    # And unique_per_set is a function of cs and min_similarity.
    core_size = math.floor((2 * max_len * min_similarity) / (1 + min_similarity))
    if core_size == 0 and max_len > 0 and min_similarity > 0:
        raise ValueError(
            "Cannot generate sets with these constraints. "
            "Try increasing max_len or decreasing min_similarity."
        )

    if min_similarity == 1.0:
        max_additional_elements = 0
    else:
        # This is the max number of elements that can be non-core across TWO sets
        max_additional_elements = math.floor(core_size * (1 / min_similarity - 1))

    core = {generate_entity() for _ in range(core_size)}

    # A large pool of entities to draw from
    pool_size = (n * max_len) * 2  # just to be safe
    pool = {generate_entity() for _ in range(pool_size)} - core

    similar_sets = []
    for _ in range(n_clique):
        s = core.copy()

        # Max unique elements per set to guarantee similarity
        max_unique_for_set = math.floor(max_additional_elements / 2)
        # Also respect max_len
        max_unique_for_set = min(max_unique_for_set, max_len - core_size)

        if max_unique_for_set > 0:
            num_unique = random.randint(0, max_unique_for_set)
            if len(pool) < num_unique:
                # Replenish pool if needed
                pool.update({generate_entity() for _ in range(num_unique * 2)} - core)
            new_elements = set(random.sample(list(pool), num_unique))
            s.update(new_elements)
            pool -= new_elements
        similar_sets.append(s)

    # --- Generate the dissimilar sets ---
    dissimilar_sets = []
    for _ in range(n_dissimilar):
        length = random.randint(0, max_len)
        length = min(length, len(pool))
        if length > 0:
            s = set(random.sample(list(pool), length))
            pool -= s
        else:
            s = set()
        dissimilar_sets.append(s)

    sets = similar_sets + dissimilar_sets
    random.shuffle(sets)

    # --- Verify the result ---
    actual_similar_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if jaccard(sets[i], sets[j]) >= min_similarity:
                actual_similar_pairs += 1

    assert actual_similar_pairs >= target_similar_pairs, (
        f"Failed to generate the required number of similar pairs. "
        f"Target: {target_similar_pairs}, Actual: {actual_similar_pairs}"
    )

    return sets


def validate_sets(sets: list[set[str]], min_similarity: float, similar_fraction: float):
    n = len(sets)
    n_similar_needed = int(n * similar_fraction)

    similar_pairs = jaccard_similarity_pair(sets, min_similarity)
    n_similar_pairs = len(similar_pairs)
    actual_similar_fraction = n_similar_pairs / (n * (n - 1) // 2)

    print(f"Expected similar pairs: {n_similar_needed}")
    print(f"Actual similar pairs: {n_similar_pairs}")
    print(f"Actual similar fraction: {actual_similar_fraction:.2f}")
    print(f"Similarity threshold: {min_similarity}")


def jaccard_similarity_matrix(sets: List[Set[str]]) -> np.ndarray:
    """Calculate Jaccard similarity matrix for a list of string sets."""
    n = len(sets)
    similarity = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i, n):
            intersection = sets[i].intersection(sets[j])
            union = sets[i].union(sets[j])
            score = len(intersection) / len(union) if union else 0.0
            similarity[i, j] = similarity[j, i] = score

    return similarity


def jaccard_similarity_pair(
    sets: List[Set[str]], threshold: float
) -> List[Tuple[int, int, float]]:
    """Find pairs of sets with Jaccard similarity >= threshold."""
    similarity_matrix = jaccard_similarity_matrix(sets)
    similar_pairs = np.argwhere(similarity_matrix >= threshold)

    return [
        (int(i), int(j), float(similarity_matrix[i, j]))
        for i, j in similar_pairs
        if i < j  # avoid self-pairs and duplicates
    ]


@pytest.fixture
def simple_kg():
    # Arrange: create a simple knowledge graph with embeddings
    # roughly, we expect the following relationships:
    # 1 <-> 2 (0.0 similarity)
    # 2 <-> 3 (0.1667 similarity)
    # 1 <-> 3 (0.25 similarity)
    nodes = [
        Node(
            id=UUID("4da47a69-539c-49a2-b289-01780989d82c"),
            type=NodeType.DOCUMENT,
            properties={
                "entities": {"cat", "dog", "fish", "fox", "bird"},
            },
        ),
        Node(
            id=UUID("f353e5c2-e432-4d1e-84a8-d750c93d4edf"),
            type=NodeType.DOCUMENT,
            properties={
                "entities": {"apple", "banana"},
            },
        ),
        Node(
            id=UUID("437c8c08-cef6-4ebf-a35f-93d6168b61a4"),
            type=NodeType.DOCUMENT,
            properties={
                "entities": {"cat", "banana", "dog", "rock", "tree"},
            },
        ),
    ]
    return KnowledgeGraph(nodes=nodes)


# node order
# UUID("4da47a69-539c-49a2-b289-01780989d82c")
# UUID("f353e5c2-e432-4d1e-84a8-d750c93d4edf")
# UUID("437c8c08-cef6-4ebf-a35f-93d6168b61a4")


@pytest.mark.parametrize(
    "n_test_sets, max_len, threshold",
    [
        (8, 100, 0.2),
        (16, 8, 0.1),
        (16, 16, 0.5),
        (32, 5, 0.3),
    ],
)
def test__find_similar_embedding_pairs_jaccard(n_test_sets, max_len, threshold):
    """
    Validate that _find_similar_embedding_pairs correctly identifies pairs when compared with scipy's jaccard distance.
    """
    sets = generate_test_sets(
        n=n_test_sets,
        max_len=max_len,
        min_similarity=min(threshold + 0.05, 1.0),
        similar_fraction=0.3,
    )
    expected = jaccard_similarity_pair(sets, threshold)

    kg = KnowledgeGraph(
        nodes=[Node(type=NodeType.DOCUMENT, properties={"entities": s}) for s in sets]
    )
    builder = JaccardSimilarityBuilder(property_name="entities", threshold=threshold)
    result = builder._find_similar_embedding_pairs(kg)

    assert len(result) == len(expected)
    for i, j, similarity_float in result:
        assert i < j, "Pairs should be ordered (i < j)"
        assert similarity_float >= threshold, (
            f"Similarity {similarity_float} should be >= {threshold}"
        )
        for x, y, expected_similarity in expected:
            if i == x and j == y:
                assert similarity_float == pytest.approx(expected_similarity)
                break


class TestJaccardSimilarityBuilder:
    @pytest.mark.asyncio
    async def test_no_self_similarity_relationships(self, simple_kg):
        builder = JaccardSimilarityBuilder(property_name="entities", threshold=0.1)
        relationships = await builder.transform(copy.deepcopy(simple_kg))
        for r in relationships:
            assert r.source.id != r.target.id, (
                "Self-relationships should not be created"
            )

    @pytest.mark.asyncio
    async def test_no_duplicate_relationships(self, simple_kg):
        builder = JaccardSimilarityBuilder(property_name="entities", threshold=0.1)
        relationships = await builder.transform(copy.deepcopy(simple_kg))
        seen = set()
        for r in relationships:
            pair = tuple(sorted([r.source.id, r.target.id]))
            assert pair not in seen, "Duplicate relationships found"
            seen.add(pair)

    @pytest.mark.asyncio
    async def test_similarity_at_threshold(self):
        node1 = Node(type=NodeType.DOCUMENT, properties={"entities": {"a", "b", "c"}})
        node2 = Node(type=NodeType.DOCUMENT, properties={"entities": {"a", "b", "c"}})
        kg = KnowledgeGraph(nodes=[node1, node2])
        builder = JaccardSimilarityBuilder(property_name="entities", threshold=1.0)
        relationships = await builder.transform(kg)
        assert len(relationships) == 1, "Should create relationship at threshold"

    @pytest.mark.asyncio
    async def test_all_below_threshold(self):
        node1 = Node(type=NodeType.DOCUMENT, properties={"entities": {"a", "b", "c"}})
        node2 = Node(type=NodeType.DOCUMENT, properties={"entities": {"x", "y", "z"}})
        kg = KnowledgeGraph(nodes=[node1, node2])
        builder = JaccardSimilarityBuilder(property_name="entities", threshold=0.1)
        relationships = await builder.transform(kg)
        assert len(relationships) == 0, (
            "No relationships should be created below threshold"
        )

    @pytest.mark.asyncio
    async def test_all_above_threshold(self):
        node1 = Node(type=NodeType.DOCUMENT, properties={"entities": {"a", "b", "c"}})
        node2 = Node(type=NodeType.DOCUMENT, properties={"entities": {"a", "b", "c"}})
        node3 = Node(type=NodeType.DOCUMENT, properties={"entities": {"a", "b", "c"}})
        kg = KnowledgeGraph(nodes=[node1, node2, node3])
        builder = JaccardSimilarityBuilder(property_name="entities", threshold=0.9)
        relationships = await builder.transform(kg)
        assert len(relationships) == 3

    @pytest.mark.asyncio
    async def test_malformed_entities_raises(self):
        node1 = Node(type=NodeType.DOCUMENT, properties={"entities": {"a", "b", "c"}})
        node2 = Node(type=NodeType.DOCUMENT, properties={"entities": None})
        kg = KnowledgeGraph(nodes=[node1, node2])
        builder = JaccardSimilarityBuilder(property_name="entities", threshold=0.5)
        with pytest.raises(ValueError):
            await builder.transform(kg)

    @pytest.mark.asyncio
    async def test_jaccard_similarity_builder_empty_graph(self):
        kg = KnowledgeGraph(nodes=[])
        builder = JaccardSimilarityBuilder(property_name="entities")
        relationships = await builder.transform(kg)
        assert relationships == []

    @pytest.mark.asyncio
    async def test_jaccard_similarity_builder_basic(self, simple_kg):
        builder = JaccardSimilarityBuilder(property_name="entities", threshold=0.15)
        relationships = await builder.transform(simple_kg)
        assert all(isinstance(r, Relationship) for r in relationships)
        assert all(r.type == "jaccard_similarity" for r in relationships)
        # 2 <-> 3 (~0.1667 similarity)
        assert any(
            str(r.source.id) == "f353e5c2-e432-4d1e-84a8-d750c93d4edf"
            and str(r.target.id) == "437c8c08-cef6-4ebf-a35f-93d6168b61a4"
            for r in relationships
        )
        # 1 <-> 3 (~0.25 similarity)
        assert any(
            str(r.source.id) == "4da47a69-539c-49a2-b289-01780989d82c"
            and str(r.target.id) == "437c8c08-cef6-4ebf-a35f-93d6168b61a4"
            for r in relationships
        )

    @pytest.mark.asyncio
    async def test_jaccard_similarity_builder_no_entities(self):
        kg = KnowledgeGraph(
            nodes=[
                Node(type=NodeType.DOCUMENT, properties={}),
                Node(type=NodeType.DOCUMENT, properties={}),
            ]
        )
        builder = JaccardSimilarityBuilder(property_name="entities")
        with pytest.raises(ValueError, match="has no entities"):
            await builder.transform(kg)

    @pytest.mark.asyncio
    async def test_apply_transforms_cosine_similarity_builder(self, simple_kg):
        from ragas.run_config import RunConfig
        from ragas.testset.transforms.engine import apply_transforms

        # JaccardSimilarityBuilder should add relationships to the graph
        builder = JaccardSimilarityBuilder(property_name="entities", threshold=0.15)
        kg = simple_kg
        # Should mutate kg in-place
        apply_transforms(kg, builder, run_config=RunConfig(max_workers=2))
        # Check that relationships were added
        assert any(r.type == "jaccard_similarity" for r in kg.relationships), (
            "No jaccard_similarity relationships found after apply_transforms"
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


def overlap_kg(entity_x: str, entity_y: str) -> KnowledgeGraph:
    """Build a two-node graph that compares `entity_x` with `entity_y`.

    OverlapScoreBuilder ignores the most frequent entities in the graph as noise,
    so both nodes also share a "noise" entity that gets filtered out instead.
    """
    return KnowledgeGraph(
        nodes=[
            Node(type=NodeType.CHUNK, properties={"entities": [entity_x, "noise"]}),
            Node(type=NodeType.CHUNK, properties={"entities": [entity_y, "noise"]}),
        ]
    )


class TestOverlapScoreBuilder:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("distance_measure", list(DistanceMeasure))
    async def test_near_identical_entities_overlap(self, distance_measure):
        kg = overlap_kg(
            "Retrieval Augmented Generation", "Retrieval-Augmented Generation"
        )
        builder = OverlapScoreBuilder(
            distance_measure=distance_measure, distance_threshold=0.9
        )
        relationships = await builder.transform(kg)
        assert len(relationships) == 1
        assert relationships[0].properties["entities_overlap_score"] == 1.0
        assert relationships[0].properties["overlapped_items"] == [
            ("Retrieval Augmented Generation", "Retrieval-Augmented Generation")
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("distance_measure", list(DistanceMeasure))
    async def test_different_entities_do_not_overlap(self, distance_measure):
        kg = overlap_kg("Retrieval Augmented Generation", "Knowledge Graph")
        builder = OverlapScoreBuilder(
            distance_measure=distance_measure, distance_threshold=0.9
        )
        relationships = await builder.transform(kg)
        assert relationships == []
