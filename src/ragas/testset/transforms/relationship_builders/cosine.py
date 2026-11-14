import typing as t
from dataclasses import dataclass

import numpy as np

from ragas.testset.graph import KnowledgeGraph, NodeType, Relationship
from ragas.testset.transforms.base import RelationshipBuilder


@dataclass
class CosineSimilarityBuilder(RelationshipBuilder):
    property_name: str = "embedding"
    new_property_name: str = "cosine_similarity"
    threshold: float = 0.9
    block_size: int = 1024

    def _block_cosine_similarity(self, i: np.ndarray, j: np.ndarray):
        """Calculate cosine similarity matrix between two sets of embeddings."""
        i_norm = i / np.linalg.norm(i, axis=1, keepdims=True)
        j_norm = j / np.linalg.norm(j, axis=1, keepdims=True)
        return np.dot(i_norm, j_norm.T)

    def _find_similar_embedding_pairs(
        self, embeddings: np.ndarray, threshold: float
    ) -> t.List[t.Tuple[int, int, float]]:
        """Sharded computation of cosine similarity to find similar pairs."""

        def process_block(i: int, j: int) -> t.Set[t.Tuple[int, int, float]]:
            end_i = min(i + self.block_size, n_embeddings)
            end_j = min(j + self.block_size, n_embeddings)
            block = self._block_cosine_similarity(
                embeddings[i:end_i, :], embeddings[j:end_j, :]
            )
            similar_idx = np.argwhere(block >= threshold)
            return {
                (int(i + ii), int(j + jj), float(block[ii, jj]))
                for ii, jj in similar_idx
                if int(i + ii) < int(j + jj)
            }

        n_embeddings, _dimension = embeddings.shape
        triplets = set()

        for i in range(0, n_embeddings, self.block_size):
            for j in range(i, n_embeddings, self.block_size):
                triplets.update(process_block(i, j))

        return list(triplets)

    def _validate_embedding_shapes(self, embeddings: t.List[t.Any]):
        if not embeddings:
            return
        first_len = len(embeddings[0])
        for idx, emb in enumerate(embeddings):
            if len(emb) != first_len:
                raise ValueError(
                    f"Embedding at index {idx} has length {len(emb)}, expected {first_len}. "
                    "All embeddings must have the same length."
                )

    async def transform(self, kg: KnowledgeGraph) -> t.List[Relationship]:
        embeddings = []
        for node in kg.nodes:
            embedding = node.get_property(self.property_name)
            if embedding is None:
                raise ValueError(f"Node {node.id} has no {self.property_name}")
            embeddings.append(embedding)
        self._validate_embedding_shapes(embeddings)
        similar_pairs = self._find_similar_embedding_pairs(
            np.array(embeddings), self.threshold
        )
        return [
            Relationship(
                source=kg.nodes[i],
                target=kg.nodes[j],
                type=self.new_property_name,
                properties={self.new_property_name: similarity_float},
                bidirectional=True,
            )
            for i, j, similarity_float in similar_pairs
        ]

    def generate_execution_plan(self, kg: KnowledgeGraph) -> t.List[t.Coroutine]:
        """
        Generates a coroutine task for finding similar embedding pairs, which can be scheduled/executed by an Executor.
        """
        filtered_kg = self.filter(kg)

        embeddings = []
        for node in filtered_kg.nodes:
            embedding = node.get_property(self.property_name)
            if embedding is None:
                raise ValueError(f"Node {node.id} has no {self.property_name}")
            embeddings.append(embedding)
        self._validate_embedding_shapes(embeddings)

        async def find_and_add_relationships():
            similar_pairs = self._find_similar_embedding_pairs(
                np.array(embeddings), self.threshold
            )
            for i, j, similarity_float in similar_pairs:
                rel = Relationship(
                    source=filtered_kg.nodes[i],
                    target=filtered_kg.nodes[j],
                    type=self.new_property_name,
                    properties={self.new_property_name: similarity_float},
                    bidirectional=True,
                )
                kg.relationships.append(rel)

        return [find_and_add_relationships()]


@dataclass
class SummaryCosineSimilarityBuilder(CosineSimilarityBuilder):
    property_name: str = "summary_embedding"
    new_property_name: str = "summary_cosine_similarity"
    threshold: float = 0.1
    block_size: int = 1024

    def generate_execution_plan(self, kg: KnowledgeGraph) -> t.List[t.Coroutine]:
        """Schedule the document-only transform after applying the node filter."""
        # The cosine builder's optimized plan bypasses transform(), including
        # this subclass's document selection. Use the shared virtual dispatch.
        return list(RelationshipBuilder.generate_execution_plan(self, kg))

    def _document_summary_filter(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """
        Filters the knowledge graph to only include nodes with a summary embedding.
        """
        nodes = []
        for node in kg.nodes:
            if node.type == NodeType.DOCUMENT:
                emb = node.get_property(self.property_name)
                if emb is None:
                    raise ValueError(f"Node {node.id} has no {self.property_name}")
                nodes.append(node)
        return KnowledgeGraph(nodes=nodes)

    async def transform(self, kg: KnowledgeGraph) -> t.List[Relationship]:
        filtered_kg = self._document_summary_filter(kg)
        embeddings = [
            node.get_property(self.property_name)
            for node in filtered_kg.nodes
            if node.get_property(self.property_name) is not None
        ]
        if not embeddings:
            raise ValueError(f"No nodes have a valid {self.property_name}")
        similar_pairs = self._find_similar_embedding_pairs(
            np.array(embeddings), self.threshold
        )
        return [
            Relationship(
                source=filtered_kg.nodes[i],
                target=filtered_kg.nodes[j],
                type=self.new_property_name,
                properties={self.new_property_name: similarity_float},
                bidirectional=True,
            )
            for i, j, similarity_float in similar_pairs
        ]
