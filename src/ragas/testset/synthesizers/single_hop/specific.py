from __future__ import annotations

import logging
import typing as t
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.base import BaseScenario
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)

from .base import SingleHopQuerySynthesizer

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class SingleHopScenario(BaseScenario):
    """
    Scenario for single-hop queries.

    Attributes
    ----------
    term: str
        The theme of the query.
    """

    term: str


@dataclass
class SingleHopSpecificQuerySynthesizer(SingleHopQuerySynthesizer):
    name: str = "single_hop_specific_query_synthesizer"
    theme_persona_matching_prompt: PydanticPrompt = ThemesPersonasMatchingPrompt()
    property_name: str = "entities"

    def _extract_themes_from_items(self, items: t.Any) -> t.List[str]:
        """
        Extract unique theme names from various formats.

        Handles multiple data formats that might appear during synthesis:
        - List[Tuple[str, str]]: Entity pairs (from overlap detection)
        - List[List[str]]: Entity pairs as lists
        - List[str]: Direct entity names
        - Dict[str, Any]: Keys as entity names

        Parameters
        ----------
        items : t.Any
            The items to extract themes from.

        Returns
        -------
        t.List[str]
            List of unique theme strings.
        """
        if isinstance(items, dict):
            return list(items.keys())

        if not isinstance(items, list):
            return []

        unique_themes = set()
        for item in items:
            if isinstance(item, (tuple, list)):
                # Extract strings from pairs/sequences
                for element in item:
                    if isinstance(element, str):
                        unique_themes.add(element)
            elif isinstance(item, str):
                unique_themes.add(item)

        return list(unique_themes)

    def get_node_clusters(self, knowledge_graph: KnowledgeGraph) -> t.List[Node]:
        node_type_dict = defaultdict(int)
        for node in knowledge_graph.nodes:
            if (
                node.type.name == "CHUNK"
                and node.get_property(self.property_name) is not None
            ):
                node_type_dict["CHUNK"] += 1
            elif (
                node.type.name == "DOCUMENT"
                and node.get_property(self.property_name) is not None
            ):
                node_type_dict["DOCUMENT"] += 1
            else:
                pass

        node_filter = (
            "CHUNK"
            if node_type_dict["CHUNK"] > node_type_dict["DOCUMENT"]
            else "DOCUMENT"
        )

        nodes = []
        for node in knowledge_graph.nodes:
            if node.type.name == node_filter:
                nodes.append(node)

        return nodes

    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        persona_list: t.List[Persona],
        callbacks: Callbacks,
    ) -> t.List[SingleHopScenario]:
        """
        Generates a list of scenarios on type SingleHopSpecificQuerySynthesizer
        Steps to generate scenarios:
        1. Find nodes with CHUNK type and entities property
        2. Calculate the number of samples that should be created per node to get n samples in total
        3. For each node
            a. Find the entities associated with the node
            b. Map personas to the entities to create query
            c. Prepare all possible combinations of (node, entities, personas, style, length) as base scenarios
            d. Sample num_sample_per_node (step 2) scenarios from base scenarios
        4. Return the list of scenarios
        """

        nodes = self.get_node_clusters(knowledge_graph)
        if not nodes:
            raise ValueError("No nodes found with the `entities` property.")
        samples_per_node = int(np.ceil(n / len(nodes)))

        scenarios = []
        for node in nodes:
            if len(scenarios) >= n:
                break
            raw_themes = node.properties.get(self.property_name, [])
            # Extract themes from potentially mixed data types (handles tuples, lists, strings)
            themes = self._extract_themes_from_items(raw_themes)

            if not themes:  # Skip if no themes extracted
                logger.debug("No themes extracted from node %s. Skipping.", node.id)
                continue

            prompt_input = ThemesPersonasInput(themes=themes, personas=persona_list)
            persona_concepts = await self.theme_persona_matching_prompt.generate(
                data=prompt_input, llm=self.llm, callbacks=callbacks
            )
            base_scenarios = self.prepare_combinations(
                node,
                themes,
                personas=persona_list,
                persona_concepts=persona_concepts.mapping,
            )
            scenarios.extend(self.sample_combinations(base_scenarios, samples_per_node))

        return scenarios
