import hashlib
import json
import random
import typing as t
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_serializer
from tqdm.auto import tqdm


class UUIDEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, uuid.UUID):
            return str(o)
        return super().default(o)


class NodeType(str, Enum):
    """
    Enumeration of node types in the knowledge graph.

    Currently supported node types are: UNKNOWN, DOCUMENT, CHUNK
    """

    UNKNOWN = ""
    DOCUMENT = "document"
    CHUNK = "chunk"


class Node(BaseModel):
    """
    Represents a node in the knowledge graph.

    Attributes
    ----------
    id : uuid.UUID
        Unique identifier for the node.
    properties : dict
        Dictionary of properties associated with the node.
    type : NodeType
        Type of the node.

    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    properties: dict = Field(default_factory=dict)
    type: NodeType = NodeType.UNKNOWN

    def __repr__(self) -> str:
        return f"Node(id: {str(self.id)[:6]}, type: {self.type}, properties: {list(self.properties.keys())})"

    def __str__(self) -> str:
        return self.__repr__()

    def add_property(self, key: str, value: t.Any):
        """
        Adds a property to the node.

        Raises
        ------
        ValueError
            If the property already exists.
        """
        if key.lower() in self.properties:
            raise ValueError(f"Property {key} already exists")
        self.properties[key.lower()] = value

    def get_property(self, key: str) -> t.Optional[t.Any]:
        """
        Retrieves a property value by key.

        Notes
        -----
        The key is case-insensitive.
        """
        return self.properties.get(key.lower(), None)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Node):
            return self.id == other.id
        return False


class Relationship(BaseModel):
    """
    Represents a relationship between two nodes in a knowledge graph.

    Attributes
    ----------
    id : uuid.UUID, optional
        Unique identifier for the relationship. Defaults to a new UUID.
    type : str
        The type of the relationship.
    source : Node
        The source node of the relationship.
    target : Node
        The target node of the relationship.
    bidirectional : bool, optional
        Whether the relationship is bidirectional. Defaults to False.
    properties : dict, optional
        Dictionary of properties associated with the relationship. Defaults to an empty dict.

    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    type: str
    source: Node
    target: Node
    bidirectional: bool = False
    properties: dict = Field(default_factory=dict)

    def get_property(self, key: str) -> t.Optional[t.Any]:
        """
        Retrieves a property value by key. The key is case-insensitive.
        """
        return self.properties.get(key.lower(), None)

    def __repr__(self) -> str:
        return f"Relationship(Node(id: {str(self.source.id)[:6]}) {'<->' if self.bidirectional else '->'} Node(id: {str(self.target.id)[:6]}), type: {self.type}, properties: {list(self.properties.keys())})"

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Relationship):
            return self.id == other.id
        return False

    @field_serializer("source", "target")
    def serialize_node(self, node: Node):
        return node.id


@dataclass
class KnowledgeGraph:
    """
    Represents a knowledge graph containing nodes and relationships.

    Attributes
    ----------
    nodes : List[Node]
        List of nodes in the knowledge graph.
    relationships : List[Relationship]
        List of relationships in the knowledge graph.
    """

    nodes: t.List[Node] = field(default_factory=list)
    relationships: t.List[Relationship] = field(default_factory=list)

    def add(self, item: t.Union[Node, Relationship]):
        """
        Adds a node or relationship to the knowledge graph.

        Raises
        ------
        ValueError
            If the item type is not Node or Relationship.
        """
        if isinstance(item, Node):
            self._add_node(item)
        elif isinstance(item, Relationship):
            self._add_relationship(item)
        else:
            raise ValueError(f"Invalid item type: {type(item)}")

    def _add_node(self, node: Node):
        self.nodes.append(node)

    def _add_relationship(self, relationship: Relationship):
        self.relationships.append(relationship)

    def save(self, path: t.Union[str, Path]):
        """Saves the knowledge graph to a JSON file.

        Parameters
        ----------
        path : Union[str, Path]
            Path where the JSON file should be saved.

        Notes
        -----
        The file is saved using UTF-8 encoding to ensure proper handling of Unicode characters
        across different platforms.
        """
        if isinstance(path, str):
            path = Path(path)

        data = {
            "nodes": [node.model_dump() for node in self.nodes],
            "relationships": [rel.model_dump() for rel in self.relationships],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=UUIDEncoder, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: t.Union[str, Path]) -> "KnowledgeGraph":
        """Loads a knowledge graph from a path.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the JSON file containing the knowledge graph.

        Returns
        -------
        KnowledgeGraph
            The loaded knowledge graph.

        Notes
        -----
        The file is read using UTF-8 encoding to ensure proper handling of Unicode characters
        across different platforms.
        """
        if isinstance(path, str):
            path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = [Node(**node_data) for node_data in data["nodes"]]

        nodes_map = {str(node.id): node for node in nodes}
        relationships = [
            Relationship(
                id=rel_data["id"],
                type=rel_data["type"],
                source=nodes_map[rel_data["source"]],
                target=nodes_map[rel_data["target"]],
                bidirectional=rel_data["bidirectional"],
                properties=rel_data["properties"],
            )
            for rel_data in data["relationships"]
        ]

        kg = cls()
        kg.nodes.extend(nodes)
        kg.relationships.extend(relationships)
        return kg

    def __repr__(self) -> str:
        return f"KnowledgeGraph(nodes: {len(self.nodes)}, relationships: {len(self.relationships)})"

    def __str__(self) -> str:
        return self.__repr__()

    def get_node_by_id(self, node_id: t.Union[uuid.UUID, str]) -> t.Optional[Node]:
        """
        Retrieves a node by its ID.

        Parameters
        ----------
        node_id : uuid.UUID
            The ID of the node to retrieve.

        Returns
        -------
        Node or None
            The node with the specified ID, or None if not found.
        """
        if isinstance(node_id, str):
            node_id = uuid.UUID(node_id)

        return next(filter(lambda n: n.id == node_id, self.nodes), None)

    def find_indirect_clusters(
        self,
        relationship_condition: t.Callable[[Relationship], bool] = lambda _: True,
        depth_limit: int = 3,
    ) -> t.List[t.Set[Node]]:
        """
        Finds "indirect clusters" of nodes in the knowledge graph based on a relationship condition.
        Uses Leiden algorithm for community detection and identifies unique paths within each cluster.

        NOTE: "indirect clusters" as used in the method name are
        "groups of nodes that are not directly connected
        but share a common relationship through other nodes",
        while the Leiden algorithm is a "clustering" algorithm that defines
        neighborhoods of nodes based on their connections --
        these definitions of "cluster" are NOT equivalent.

        Parameters
        ----------
        relationship_condition : Callable[[Relationship], bool], optional
            A function that takes a Relationship and returns a boolean, by default lambda _: True
        depth_limit : int, optional
            The maximum depth of relationships (number of edges) to consider for clustering, by default 3.

        Returns
        -------
        List[Set[Node]]
            A list of sets, where each set contains nodes that form a cluster.
        """

        import networkx as nx

        def get_node_clusters(
            relationships: list[Relationship],
        ) -> dict[int, set[uuid.UUID]]:
            """Identify clusters of nodes using Leiden algorithm."""
            import numpy as np
            from sknetwork.clustering import Leiden
            from sknetwork.data import Dataset as SKDataset, from_edge_list

            # NOTE: the upstream sknetwork Dataset has some issues with type hints,
            # so we use type: ignore to bypass them.
            # Use hex representation to ensure proper UUID strings for clustering
            graph: SKDataset = from_edge_list(  # type: ignore
                [(rel.source.id.hex, rel.target.id.hex) for rel in relationships],
                directed=True,
            )

            # Apply Leiden clustering
            leiden = Leiden(random_state=42)
            cluster_labels: np.ndarray = leiden.fit_predict(graph["adjacency"])

            # Group nodes by cluster
            clusters: defaultdict[int, set[uuid.UUID]] = defaultdict(set)
            for label, node_id_hex in zip(cluster_labels, graph["names"]):
                # node_id_hex is the hex string representation of the UUID
                clusters[int(label)].add(uuid.UUID(hex=node_id_hex))

            return dict(clusters)

        def to_nx_digraph(
            nodes: set[uuid.UUID], relationships: list[Relationship]
        ) -> nx.DiGraph:
            """Convert a set of nodes and relationships to a directed graph."""
            # Create directed subgraph for this cluster
            graph = nx.DiGraph()
            for node_id in nodes:
                graph.add_node(
                    node_id,
                    node_obj=self.get_node_by_id(node_id),
                )
            for rel in relationships:
                if rel.source.id in nodes and rel.target.id in nodes:
                    graph.add_edge(rel.source.id, rel.target.id, relationship_obj=rel)
            return graph

        def max_simple_paths(n: int, k: int = depth_limit) -> int:
            """Estimate the number of paths up to depth_limit that would exist in a fully-connected graph of size cluster_nodes."""
            from math import prod

            if n - k - 1 <= 0:
                return 0

            return prod(n - i for i in range(k + 1))

        def exhaustive_paths(
            graph: nx.DiGraph, depth_limit: int
        ) -> list[list[uuid.UUID]]:
            """Find all simple paths in the subgraph up to depth_limit."""
            import itertools

            # Check if graph has enough nodes for meaningful paths
            if len(graph) < 2:
                return []

            all_paths: list[list[uuid.UUID]] = []
            for source, target in itertools.permutations(graph.nodes(), 2):
                if not nx.has_path(graph, source, target):
                    continue
                try:
                    paths = nx.all_simple_paths(
                        graph,
                        source,
                        target,
                        cutoff=depth_limit,
                    )
                    all_paths.extend(paths)
                except nx.NetworkXNoPath:
                    continue

            return all_paths

        def sample_paths_from_graph(
            graph: nx.DiGraph, depth_limit: int, sample_size: int = 1000
        ) -> list[list[uuid.UUID]]:
            """Sample random paths in the graph up to depth_limit."""
            # we're using a DiGraph, so we need to account for directionality
            # if a node has no out-paths, then it will cause an error in `generate_random_paths`

            # Iteratively remove nodes with no out-paths to handle cascading effects
            while True:
                nodes_with_no_outpaths = [
                    n for n in graph.nodes() if graph.out_degree(n) == 0
                ]
                if not nodes_with_no_outpaths:
                    break
                graph.remove_nodes_from(nodes_with_no_outpaths)

            # Check if graph is empty after node removal
            if not graph:
                return []

            sampled_paths: list[list[uuid.UUID]] = []
            for depth in range(2, depth_limit + 1):
                # Additional safety check before generating paths
                if (
                    len(graph) < depth + 1
                ):  # Need at least depth+1 nodes for a path of length depth
                    continue

                paths = nx.generate_random_paths(
                    graph,
                    sample_size=sample_size,
                    path_length=depth,
                )
                sampled_paths.extend(paths)
            return sampled_paths

        # depth 2: 3 nodes, 2 edges (A -> B -> C)
        if depth_limit < 2:
            raise ValueError("Depth limit must be at least 2")

        # Filter relationships based on the condition
        filtered_relationships: list[Relationship] = []
        relationship_map: defaultdict[uuid.UUID, set[uuid.UUID]] = defaultdict(set)
        for rel in self.relationships:
            if relationship_condition(rel):
                filtered_relationships.append(rel)
                relationship_map[rel.source.id].add(rel.target.id)
                if rel.bidirectional:
                    relationship_map[rel.target.id].add(rel.source.id)

        if not filtered_relationships:
            return []

        clusters = get_node_clusters(filtered_relationships)

        # For each cluster, find valid paths up to depth_limit
        cluster_sets: set[frozenset] = set()
        for _cluster_label, cluster_nodes in tqdm(
            clusters.items(), desc="Processing clusters"
        ):
            # Skip clusters that are too small to form any meaningful paths (need at least 2 nodes)
            if len(cluster_nodes) < 2:
                continue

            subgraph = to_nx_digraph(
                nodes=cluster_nodes, relationships=filtered_relationships
            )

            sampled_paths: list[list[uuid.UUID]] = []
            # if the expected number of paths is small, use exhaustive search
            # otherwise sample with random walks
            if max_simple_paths(n=len(cluster_nodes), k=depth_limit) < 1000:
                sampled_paths.extend(exhaustive_paths(subgraph, depth_limit))
            else:
                sampled_paths.extend(sample_paths_from_graph(subgraph, depth_limit))

            # convert paths (node IDs) to sets of Node objects
            # and deduplicate
            for path in sampled_paths:
                path_nodes = {subgraph.nodes[node_id]["node_obj"] for node_id in path}
                cluster_sets.add(frozenset(path_nodes))

        return [set(path_nodes) for path_nodes in cluster_sets]

    def find_n_indirect_clusters(
        self,
        n: int,
        relationship_condition: t.Callable[[Relationship], bool] = lambda _: True,
        depth_limit: int = 3,
    ) -> t.List[t.Set[Node]]:
        """
        Return n indirect clusters of nodes in the knowledge graph based on a relationship condition.
        Optimized for large datasets by using an adjacency index for lookups and limiting path exploration
        relative to n.

        A cluster represents a path through the graph. For example, if A -> B -> C -> D exists in the graph,
        then {A, B, C, D} forms a cluster. If there's also a path A -> B -> C -> E, it forms a separate cluster.

        The method returns a list of up to n sets, where each set contains nodes forming a complete path
        from a starting node to a leaf node or a path segment up to depth_limit nodes long. The result may contain
        fewer than n clusters if the graph is very sparse or if there aren't enough nodes to form n distinct clusters.

        To maximize diversity in the results:
        1. Random starting nodes are selected
        2. Paths from each starting node are grouped
        3. Clusters are selected in round-robin fashion from each group until n unique clusters are found
        4. Duplicate clusters are eliminated
        5. When a superset cluster is found (e.g., {A,B,C,D}), any existing subset clusters (e.g., {A,B,C})
           are removed to avoid redundancy

        Parameters
        ----------
        n : int
            Target number of clusters to return. Must be at least 1. Should return n clusters unless the graph is
            extremely sparse.
        relationship_condition : Callable[[Relationship], bool], optional
            A function that takes a Relationship and returns a boolean, by default lambda _: True
        depth_limit : int, optional
            Maximum depth for path exploration, by default 3. Must be at least 2 to form clusters by definition.

        Returns
        -------
        List[Set[Node]]
            A list of sets, where each set contains nodes that form a cluster.

        Raises
        ------
        ValueError
            If depth_limit < 2, n < 1, or no relationships match the provided condition.
        """
        if depth_limit < 2:
            raise ValueError("depth_limit must be at least 2 to form valid clusters")

        if n < 1:
            raise ValueError("n must be at least 1")

        # Filter relationships once upfront
        filtered_relationships: list[Relationship] = [
            rel for rel in self.relationships if relationship_condition(rel)
        ]

        if not filtered_relationships:
            raise ValueError(
                "No relationships match the provided condition. Cannot form clusters."
            )

        # Build adjacency list for faster neighbor lookup - optimized for large datasets
        adjacency_list: dict[Node, set[Node]] = {}
        unique_edges: set[frozenset[Node]] = set()
        for rel in filtered_relationships:
            # Lazy initialization since we only care about nodes with relationships
            if rel.source not in adjacency_list:
                adjacency_list[rel.source] = set()
            adjacency_list[rel.source].add(rel.target)
            unique_edges.add(frozenset({rel.source, rel.target}))
            if rel.bidirectional:
                if rel.target not in adjacency_list:
                    adjacency_list[rel.target] = set()
                adjacency_list[rel.target].add(rel.source)

        # Aggregate clusters for each start node
        start_node_clusters: dict[Node, set[frozenset[Node]]] = {}
        # sample enough starting nodes to handle worst case grouping scenario where nodes are grouped
        # in independent clusters of size equal to depth_limit. This only surfaces when there are less
        # unique edges than nodes.
        connected_nodes: set[Node] = set().union(*unique_edges)
        sample_size: int = (
            (n - 1) * depth_limit + 1
            if len(unique_edges) < len(connected_nodes)
            else max(n, depth_limit, 10)
        )

        def dfs(node: Node, start_node: Node, current_path: t.Set[Node]):
            # Terminate exploration when max usable clusters is reached so complexity doesn't spiral
            if len(start_node_clusters.get(start_node, [])) > sample_size:
                return

            current_path.add(node)
            path_length = len(current_path)
            at_max_depth = path_length >= depth_limit
            neighbors = adjacency_list.get(node, None)

            # If this is a leaf node or we've reached depth limit
            # and we have a valid path of at least 2 nodes, add it as a cluster
            if path_length > 1 and (
                at_max_depth
                or not neighbors
                or all(n in current_path for n in neighbors)
            ):
                # Lazy initialization of the set for this start node
                if start_node not in start_node_clusters:
                    start_node_clusters[start_node] = set()
                start_node_clusters[start_node].add(frozenset(current_path))
            elif neighbors:
                for neighbor in neighbors:
                    # Block cycles
                    if neighbor not in current_path:
                        dfs(neighbor, start_node, current_path)

            # Backtrack by removing the current node from path
            current_path.remove(node)

        # Shuffle nodes for random starting points
        # Use adjacency list since that has filtered out isolated nodes
        # Sort by node ID for consistent ordering while maintaining algorithm effectiveness
        start_nodes = sorted(adjacency_list.keys(), key=lambda n: n.id.hex)
        # Use a hash-based seed for reproducible but varied shuffling based on the nodes themselves
        node_ids_str = "".join(n.id.hex for n in start_nodes)
        node_hash = hashlib.sha256(node_ids_str.encode("utf-8")).hexdigest()
        rng = random.Random(int(node_hash[:8], 16))  # Use first 8 hex chars as seed
        rng.shuffle(start_nodes)
        samples: list[Node] = start_nodes[:sample_size]
        for start_node in samples:
            dfs(start_node, start_node, set())

        start_node_clusters_list: list[set[frozenset[Node]]] = list(
            start_node_clusters.values()
        )

        # Iteratively pop from each start_node_clusters until we have n unique clusters
        # Avoid adding duplicates and subset/superset pairs so we have diversity. We
        # favor supersets over subsets if we are given a choice.
        unique_clusters: set[frozenset[Node]] = set()
        i = 0
        while len(unique_clusters) < n and start_node_clusters_list:
            # Cycle through the start node clusters
            current_index = i % len(start_node_clusters_list)

            current_start_node_clusters: set[frozenset[Node]] = (
                start_node_clusters_list[current_index]
            )
            cluster: frozenset[Node] = current_start_node_clusters.pop()

            # Check if the new cluster is a subset of any existing cluster
            # and collect any existing clusters that are subsets of this cluster
            is_subset = False
            subsets_to_remove: set[frozenset[Node]] = set()

            for existing in unique_clusters:
                if cluster.issubset(existing):
                    is_subset = True
                    break
                elif cluster.issuperset(existing):
                    subsets_to_remove.add(existing)

            # Only add the new cluster if it's not a subset of any existing cluster
            if not is_subset:
                # Remove any subsets of the new cluster
                unique_clusters -= subsets_to_remove
                unique_clusters.add(cluster)

            # If this set is now empty, remove it
            if not current_start_node_clusters:
                start_node_clusters_list.pop(current_index)
                # Don't increment i since we removed an element to account for shift
            else:
                i += 1

        return [set(cluster) for cluster in unique_clusters]

    def remove_node(
        self, node: Node, inplace: bool = True
    ) -> t.Optional["KnowledgeGraph"]:
        """
        Removes a node and its associated relationships from the knowledge graph.

        Parameters
        ----------
        node : Node
            The node to be removed from the knowledge graph.
        inplace : bool, optional
            If True, modifies the knowledge graph in place.
            If False, returns a modified copy with the node removed.

        Returns
        -------
        KnowledgeGraph or None
            Returns a modified copy of the knowledge graph if `inplace` is False.
            Returns None if `inplace` is True.

        Raises
        ------
        ValueError
            If the node is not present in the knowledge graph.
        """
        if node not in self.nodes:
            raise ValueError("Node is not present in the knowledge graph.")

        if inplace:
            # Modify the current instance
            self.nodes.remove(node)
            self.relationships = [
                rel
                for rel in self.relationships
                if rel.source != node and rel.target != node
            ]
        else:
            # Create a deep copy and modify it
            new_graph = deepcopy(self)
            new_graph.nodes.remove(node)
            new_graph.relationships = [
                rel
                for rel in new_graph.relationships
                if rel.source != node and rel.target != node
            ]
            return new_graph

    def find_two_nodes_single_rel(
        self, relationship_condition: t.Callable[[Relationship], bool] = lambda _: True
    ) -> t.List[t.Tuple[Node, Relationship, Node]]:
        """
        Finds nodes in the knowledge graph based on a relationship condition.
        (NodeA, NodeB, Rel) triples are considered as multi-hop nodes.

        Parameters
        ----------
        relationship_condition : Callable[[Relationship], bool], optional
            A function that takes a Relationship and returns a boolean, by default lambda _: True

        Returns
        -------
        List[Set[Node, Relationship, Node]]
            A list of sets, where each set contains two nodes and a relationship forming a multi-hop node.
        """

        relationships = [
            relationship
            for relationship in self.relationships
            if relationship_condition(relationship)
        ]

        triplets = set()

        for relationship in relationships:
            if relationship.source != relationship.target:
                node_a = relationship.source
                node_b = relationship.target
                # Ensure the smaller ID node is always first
                if node_a.id < node_b.id:
                    normalized_tuple = (node_a, relationship, node_b)
                else:
                    normalized_relationship = Relationship(
                        source=node_b,
                        target=node_a,
                        type=relationship.type,
                        properties=relationship.properties,
                    )
                    normalized_tuple = (node_b, normalized_relationship, node_a)

                triplets.add(normalized_tuple)

        return list(triplets)
