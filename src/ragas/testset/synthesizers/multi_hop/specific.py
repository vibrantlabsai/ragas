from __future__ import annotations

import logging
import typing as t
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.multi_hop.base import (
    MultiHopQuerySynthesizer,
    MultiHopScenario,
)
from ragas.testset.synthesizers.multi_hop.prompts import QueryAnswerGenerationPrompt
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


@dataclass
class MultiHopSpecificQuerySynthesizer(MultiHopQuerySynthesizer):
    """Synthesize multi-hop queries based on a chunk cluster defined by entity overlap."""

    name: str = "multi_hop_specific_query_synthesizer"
    property_name: str = "entities"
    relation_type: str = "entities_overlap"
    relation_overlap_property: str = "overlapped_items"
    theme_persona_matching_prompt: PydanticPrompt = ThemesPersonasMatchingPrompt()
    generate_query_reference_prompt: PydanticPrompt = QueryAnswerGenerationPrompt()

    def get_node_clusters(self, knowledge_graph: KnowledgeGraph) -> t.List[t.Tuple]:
        """Identify clusters of nodes based on the specified relationship condition."""
        node_clusters = knowledge_graph.find_two_nodes_single_rel(
            relationship_condition=lambda rel: rel.type == self.relation_type
        )
        logger.info("found %d clusters", len(node_clusters))
        return node_clusters

    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        persona_list: t.List[Persona],
        callbacks: Callbacks,
    ) -> t.List[MultiHopScenario]:
        """
        Generate a list of scenarios of type MultiHopScenario.

        Steps to generate scenarios:
        1. Filter the knowledge graph to find cluster of nodes or defined relation type. Here entities_overlap
        2. Calculate the number of samples that should be created per cluster to get n samples in total
        3. For each cluster of nodes
            a. Find the entities that are common between the nodes
            b. Find list of personas that can be associated with the entities to create query
            c. Create all possible combinations of (nodes, entities, personas, style, length) as scenarios
            3. Sample num_sample_per_cluster scenarios from the list of scenarios
        4. Return the list of scenarios of length n
        """

        triplets = self.get_node_clusters(knowledge_graph)

        if len(triplets) == 0:
            raise ValueError(
                "No clusters found in the knowledge graph. Try changing the relationship condition."
            )

        num_sample_per_cluster = int(np.ceil(n / len(triplets)))
        scenarios = []

        for triplet in triplets:
            if len(scenarios) < n:
                node_a, node_b = triplet[0], triplet[-1]
                overlapped_items = triplet[1].properties[self.relation_overlap_property]
                if overlapped_items:
                    if not all(
                        isinstance(item, (str, Iterable)) for item in overlapped_items
                    ):
                        logger.debug("Overlapped items are not strings or iterables.")
                        continue
                    themes = self._extract_themes_from_overlaps(overlapped_items)

                    prompt_input = ThemesPersonasInput(
                        themes=themes, personas=persona_list
                    )
                    persona_concepts = (
                        await self.theme_persona_matching_prompt.generate(
                            data=prompt_input, llm=self.llm, callbacks=callbacks
                        )
                    )

                    combinations = self._extract_theme_groups_from_overlaps(
                        overlapped_items
                    )

                    base_scenarios = self.prepare_combinations(
                        [node_a, node_b],
                        combinations,
                        personas=persona_list,
                        persona_item_mapping=persona_concepts.mapping,
                        property_name=self.property_name,
                    )
                    base_scenarios = self.sample_diverse_combinations(
                        base_scenarios, num_sample_per_cluster
                    )
                    scenarios.extend(base_scenarios)

        return scenarios

    def _extract_themes_from_overlaps(self, overlapped_items: t.Any) -> t.List[str]:
        """
        Extract unique entity names from overlapped items.

        Handles multiple formats:
        - List[Tuple[str, str]]: Entity pairs from overlap detection
        - List[List[str]]: Entity pairs as lists
        - List[str]: Direct entity names
        - Dict[str, Any]: Keys as entity names
        """
        if isinstance(overlapped_items, dict):
            return list(overlapped_items.keys())

        if not isinstance(overlapped_items, list):
            return []

        unique_entities = set()
        for item in overlapped_items:
            if isinstance(item, (tuple, list)):
                # Extract both entities from the pair
                for entity in item:
                    if isinstance(entity, str):
                        unique_entities.add(entity)
            elif isinstance(item, str):
                unique_entities.add(item)

        return list(unique_entities)

    def _extract_theme_groups_from_overlaps(
        self, overlapped_items: t.Any
    ) -> t.List[t.List[str]]:
        """
        Extract unique groups of entity names from overlapped items.

        Handles multiple formats:
        - List[Tuple[str, str]]: Entity pairs from overlap detection
        - List[List[str]]: Entity pairs as lists
        - List[str]: Direct entity names
        - Dict[str, Any]: Keys as entity names
        """
        if isinstance(overlapped_items, dict):
            return [[key] for key in overlapped_items]

        if not isinstance(overlapped_items, list):
            return []

        unique_groups = set()
        for item in overlapped_items:
            if isinstance(item, tuple):
                unique_groups.add(item)
            elif isinstance(item, list):
                unique_groups.add(tuple(item))
            elif isinstance(item, str):
                unique_groups.add((item,))

        return [list(group) for group in unique_groups]
