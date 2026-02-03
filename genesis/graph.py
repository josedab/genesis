"""Graph Data Synthesis for Genesis.

This module provides synthetic graph data generation including knowledge graphs,
social networks, and relationship data with NetworkX and Neo4j integration.

Example:
    >>> from genesis.graph import GraphGenerator, GraphType
    >>>
    >>> # Generate a social network
    >>> generator = GraphGenerator(graph_type=GraphType.SOCIAL_NETWORK)
    >>> G = generator.generate(n_nodes=1000, n_edges=5000)
    >>>
    >>> # Export to Neo4j
    >>> generator.to_neo4j(G, uri="bolt://localhost:7687")
    >>>
    >>> # Generate a knowledge graph
    >>> kg_gen = KnowledgeGraphGenerator()
    >>> kg = kg_gen.generate(
    ...     entity_types=["Person", "Company", "Product"],
    ...     relationship_types=["works_for", "manufactures", "purchases"],
    ...     n_entities=500,
    ... )
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from genesis.core.config import PrivacyConfig
from genesis.core.exceptions import ConfigurationError, ValidationError
from genesis.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy import for optional networkx dependency
NETWORKX_AVAILABLE = False
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None  # type: ignore


def _check_networkx_available() -> None:
    """Check if NetworkX is available."""
    if not NETWORKX_AVAILABLE:
        raise ImportError(
            "NetworkX is required for graph generation. "
            "Install it with: pip install networkx"
        )


class GraphType(Enum):
    """Types of graphs that can be generated."""

    RANDOM = "random"  # Erdős–Rényi random graph
    SCALE_FREE = "scale_free"  # Barabási–Albert preferential attachment
    SMALL_WORLD = "small_world"  # Watts–Strogatz small-world
    SOCIAL_NETWORK = "social_network"  # Social network with communities
    KNOWLEDGE_GRAPH = "knowledge_graph"  # Entity-relationship knowledge graph
    HIERARCHICAL = "hierarchical"  # Tree-like hierarchical structure
    BIPARTITE = "bipartite"  # Two-mode network
    CITATION = "citation"  # Citation network (DAG)
    TRANSACTION = "transaction"  # Financial transaction network


class NodeType(Enum):
    """Common node types for knowledge graphs."""

    PERSON = "Person"
    ORGANIZATION = "Organization"
    COMPANY = "Company"
    PRODUCT = "Product"
    LOCATION = "Location"
    EVENT = "Event"
    CONCEPT = "Concept"
    DOCUMENT = "Document"
    CUSTOM = "Custom"


class RelationshipType(Enum):
    """Common relationship types."""

    # Social
    KNOWS = "knows"
    FOLLOWS = "follows"
    FRIENDS_WITH = "friends_with"
    WORKS_WITH = "works_with"

    # Organizational
    WORKS_FOR = "works_for"
    MANAGES = "manages"
    REPORTS_TO = "reports_to"
    MEMBER_OF = "member_of"

    # Commercial
    PURCHASES = "purchases"
    SELLS = "sells"
    MANUFACTURES = "manufactures"
    SUPPLIES = "supplies"

    # Knowledge
    IS_A = "is_a"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    HAS_PROPERTY = "has_property"

    # Location
    LOCATED_IN = "located_in"
    BORN_IN = "born_in"
    HEADQUARTERED_IN = "headquartered_in"

    # Custom
    CUSTOM = "custom"


@dataclass
class NodeSchema:
    """Schema for node attributes.

    Attributes:
        node_type: Type of node
        attributes: Dictionary of attribute names to types/specs
        required_attributes: List of required attribute names
    """

    node_type: Union[NodeType, str]
    attributes: Dict[str, str] = field(default_factory=dict)
    required_attributes: List[str] = field(default_factory=list)
    count_range: Tuple[int, int] = (10, 100)

    def to_dict(self) -> Dict[str, Any]:
        node_type_val = self.node_type.value if isinstance(self.node_type, NodeType) else self.node_type
        return {
            "node_type": node_type_val,
            "attributes": self.attributes,
            "required_attributes": self.required_attributes,
            "count_range": self.count_range,
        }


@dataclass
class EdgeSchema:
    """Schema for edge/relationship attributes.

    Attributes:
        relationship_type: Type of relationship
        source_types: Valid source node types
        target_types: Valid target node types
        attributes: Edge attribute specifications
        directed: Whether edges are directed
        allow_self_loops: Whether self-loops are allowed
        cardinality: 'one-to-one', 'one-to-many', 'many-to-many'
    """

    relationship_type: Union[RelationshipType, str]
    source_types: List[Union[NodeType, str]]
    target_types: List[Union[NodeType, str]]
    attributes: Dict[str, str] = field(default_factory=dict)
    directed: bool = True
    allow_self_loops: bool = False
    cardinality: str = "many-to-many"
    probability: float = 0.1  # Probability of edge existing between valid node pairs

    def to_dict(self) -> Dict[str, Any]:
        rel_type = self.relationship_type.value if isinstance(self.relationship_type, RelationshipType) else self.relationship_type

        def type_to_str(t: Union[NodeType, str]) -> str:
            return t.value if isinstance(t, NodeType) else t

        return {
            "relationship_type": rel_type,
            "source_types": [type_to_str(t) for t in self.source_types],
            "target_types": [type_to_str(t) for t in self.target_types],
            "attributes": self.attributes,
            "directed": self.directed,
            "allow_self_loops": self.allow_self_loops,
            "cardinality": self.cardinality,
            "probability": self.probability,
        }


@dataclass
class GraphSchema:
    """Complete schema for a graph.

    Attributes:
        name: Schema name
        node_schemas: List of node schemas
        edge_schemas: List of edge schemas
        properties: Global graph properties
    """

    name: str
    node_schemas: List[NodeSchema] = field(default_factory=list)
    edge_schemas: List[EdgeSchema] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    def add_node_type(
        self,
        node_type: Union[NodeType, str],
        attributes: Optional[Dict[str, str]] = None,
        count_range: Tuple[int, int] = (10, 100),
    ) -> "GraphSchema":
        """Add a node type to the schema."""
        self.node_schemas.append(
            NodeSchema(
                node_type=node_type,
                attributes=attributes or {},
                count_range=count_range,
            )
        )
        return self

    def add_relationship(
        self,
        relationship_type: Union[RelationshipType, str],
        source_types: List[Union[NodeType, str]],
        target_types: List[Union[NodeType, str]],
        probability: float = 0.1,
        directed: bool = True,
        attributes: Optional[Dict[str, str]] = None,
    ) -> "GraphSchema":
        """Add a relationship type to the schema."""
        self.edge_schemas.append(
            EdgeSchema(
                relationship_type=relationship_type,
                source_types=source_types,
                target_types=target_types,
                probability=probability,
                directed=directed,
                attributes=attributes or {},
            )
        )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "node_schemas": [ns.to_dict() for ns in self.node_schemas],
            "edge_schemas": [es.to_dict() for es in self.edge_schemas],
            "properties": self.properties,
        }


@dataclass
class GraphConfig:
    """Configuration for graph generation.

    Attributes:
        graph_type: Type of graph to generate
        n_nodes: Number of nodes
        n_edges: Target number of edges (approximate)
        directed: Whether graph is directed
        weighted: Whether edges have weights
        seed: Random seed
        preserve_degree_distribution: Preserve realistic degree distribution
        community_structure: Generate community structure
        n_communities: Number of communities (if community_structure=True)
    """

    graph_type: GraphType = GraphType.RANDOM
    n_nodes: int = 100
    n_edges: Optional[int] = None
    directed: bool = False
    weighted: bool = False
    seed: Optional[int] = None
    preserve_degree_distribution: bool = True
    community_structure: bool = False
    n_communities: int = 5
    clustering_coefficient: Optional[float] = None
    average_degree: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_type": self.graph_type.value,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "directed": self.directed,
            "weighted": self.weighted,
            "seed": self.seed,
            "preserve_degree_distribution": self.preserve_degree_distribution,
            "community_structure": self.community_structure,
            "n_communities": self.n_communities,
        }


class NodeAttributeGenerator:
    """Generate attributes for graph nodes."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        self._random = random.Random(seed)

        # Attribute pools
        self._first_names = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer",
            "Michael", "Linda", "William", "Elizabeth", "David", "Barbara",
            "Emma", "Olivia", "Liam", "Noah", "Oliver", "Sophia",
        ]
        self._last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
            "Miller", "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson",
        ]
        self._companies = [
            "TechCorp", "GlobalSoft", "DataSystems", "CloudNet", "InnovateTech",
            "FutureLabs", "SmartSolutions", "NextGen Inc", "Digital Dynamics",
        ]
        self._cities = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "London", "Paris", "Tokyo", "Berlin", "Sydney", "Toronto",
        ]

    def generate_attributes(
        self,
        node_type: Union[NodeType, str],
        attribute_specs: Dict[str, str],
        n_nodes: int,
    ) -> List[Dict[str, Any]]:
        """Generate attributes for multiple nodes.

        Args:
            node_type: Type of node
            attribute_specs: Attribute specifications
            n_nodes: Number of nodes

        Returns:
            List of attribute dictionaries
        """
        attributes = []
        for i in range(n_nodes):
            node_attrs = {"_id": str(uuid.uuid4())[:8]}

            # Add type-specific default attributes
            if isinstance(node_type, NodeType):
                node_attrs.update(self._default_attributes(node_type))

            # Add specified attributes
            for attr_name, attr_type in attribute_specs.items():
                node_attrs[attr_name] = self._generate_attribute(attr_type)

            attributes.append(node_attrs)

        return attributes

    def _default_attributes(self, node_type: NodeType) -> Dict[str, Any]:
        """Generate default attributes based on node type."""
        if node_type == NodeType.PERSON:
            return {
                "name": f"{self._random.choice(self._first_names)} {self._random.choice(self._last_names)}",
                "age": int(self._rng.integers(18, 80)),
                "city": self._random.choice(self._cities),
            }
        elif node_type == NodeType.COMPANY:
            return {
                "name": f"{self._random.choice(self._companies)} {self._rng.integers(1, 100)}",
                "industry": self._random.choice(["Technology", "Finance", "Healthcare", "Retail"]),
                "employees": int(self._rng.integers(10, 10000)),
            }
        elif node_type == NodeType.PRODUCT:
            return {
                "name": f"Product-{self._rng.integers(1000, 9999)}",
                "category": self._random.choice(["Electronics", "Software", "Services", "Hardware"]),
                "price": round(float(self._rng.uniform(10, 1000)), 2),
            }
        elif node_type == NodeType.LOCATION:
            return {
                "name": self._random.choice(self._cities),
                "country": self._random.choice(["USA", "UK", "Germany", "Japan", "Australia"]),
                "population": int(self._rng.integers(10000, 10000000)),
            }
        return {}

    def _generate_attribute(self, attr_type: str) -> Any:
        """Generate a single attribute value."""
        if attr_type == "int":
            return int(self._rng.integers(0, 1000))
        elif attr_type == "float":
            return round(float(self._rng.uniform(0, 100)), 2)
        elif attr_type == "bool":
            return bool(self._rng.choice([True, False]))
        elif attr_type == "string":
            return f"value_{self._rng.integers(1, 10000)}"
        elif attr_type == "date":
            from datetime import date, timedelta

            return str(date(2020, 1, 1) + timedelta(days=int(self._rng.integers(0, 1000))))
        elif attr_type == "name":
            return f"{self._random.choice(self._first_names)} {self._random.choice(self._last_names)}"
        elif attr_type == "email":
            return f"{self._random.choice(self._first_names).lower()}{self._rng.integers(1, 100)}@example.com"
        else:
            return f"value_{self._rng.integers(1, 10000)}"


class GraphGenerator:
    """Main class for generating synthetic graphs.

    Example:
        >>> generator = GraphGenerator(
        ...     config=GraphConfig(
        ...         graph_type=GraphType.SOCIAL_NETWORK,
        ...         n_nodes=1000,
        ...         community_structure=True,
        ...     )
        ... )
        >>> G = generator.generate()
        >>> print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    """

    def __init__(
        self,
        config: Optional[GraphConfig] = None,
        schema: Optional[GraphSchema] = None,
        privacy: Optional[PrivacyConfig] = None,
    ):
        """Initialize the graph generator.

        Args:
            config: Graph generation configuration
            schema: Optional graph schema for typed graphs
            privacy: Privacy configuration
        """
        _check_networkx_available()

        self.config = config or GraphConfig()
        self.schema = schema
        self.privacy = privacy

        self._rng = np.random.default_rng(self.config.seed)
        self._random = random.Random(self.config.seed)
        self._attr_generator = NodeAttributeGenerator(self.config.seed)

    def generate(
        self,
        n_nodes: Optional[int] = None,
        n_edges: Optional[int] = None,
    ) -> "nx.Graph":
        """Generate a synthetic graph.

        Args:
            n_nodes: Override number of nodes
            n_edges: Override number of edges

        Returns:
            NetworkX graph object
        """
        n = n_nodes or self.config.n_nodes
        m = n_edges or self.config.n_edges

        # Select generation method based on type
        if self.config.graph_type == GraphType.RANDOM:
            G = self._generate_random(n, m)
        elif self.config.graph_type == GraphType.SCALE_FREE:
            G = self._generate_scale_free(n)
        elif self.config.graph_type == GraphType.SMALL_WORLD:
            G = self._generate_small_world(n)
        elif self.config.graph_type == GraphType.SOCIAL_NETWORK:
            G = self._generate_social_network(n)
        elif self.config.graph_type == GraphType.HIERARCHICAL:
            G = self._generate_hierarchical(n)
        elif self.config.graph_type == GraphType.BIPARTITE:
            G = self._generate_bipartite(n)
        elif self.config.graph_type == GraphType.CITATION:
            G = self._generate_citation(n)
        else:
            G = self._generate_random(n, m)

        # Add weights if requested
        if self.config.weighted:
            self._add_weights(G)

        # Add node attributes
        self._add_node_attributes(G)

        return G

    def _generate_random(self, n: int, m: Optional[int] = None) -> "nx.Graph":
        """Generate Erdős–Rényi random graph."""
        if m is None:
            # Calculate edge probability for average degree ~6
            p = min(6.0 / n, 1.0)
            if self.config.directed:
                return nx.erdos_renyi_graph(n, p, directed=True, seed=self.config.seed)
            return nx.erdos_renyi_graph(n, p, seed=self.config.seed)
        else:
            if self.config.directed:
                return nx.gnm_random_graph(n, m, directed=True, seed=self.config.seed)
            return nx.gnm_random_graph(n, m, seed=self.config.seed)

    def _generate_scale_free(self, n: int) -> "nx.Graph":
        """Generate Barabási–Albert preferential attachment graph."""
        m = self.config.average_degree or 3
        return nx.barabasi_albert_graph(n, int(m), seed=self.config.seed)

    def _generate_small_world(self, n: int) -> "nx.Graph":
        """Generate Watts–Strogatz small-world graph."""
        k = int(self.config.average_degree or 6)
        p = 0.3  # Rewiring probability
        return nx.watts_strogatz_graph(n, k, p, seed=self.config.seed)

    def _generate_social_network(self, n: int) -> "nx.Graph":
        """Generate social network with community structure."""
        if self.config.community_structure:
            # Use LFR benchmark for realistic community structure
            try:
                G = nx.LFR_benchmark_graph(
                    n,
                    tau1=2.5,  # Power law exponent for degree distribution
                    tau2=1.5,  # Power law exponent for community sizes
                    mu=0.1,  # Mixing parameter
                    average_degree=self.config.average_degree or 10,
                    min_community=max(10, n // 20),
                    seed=self.config.seed,
                )
                return G
            except Exception:
                # Fall back to simpler community graph
                pass

        # Generate using stochastic block model
        sizes = self._partition_nodes(n, self.config.n_communities)
        p_in = 0.3  # Within-community edge probability
        p_out = 0.01  # Between-community edge probability

        probs = [[p_in if i == j else p_out for j in range(len(sizes))] for i in range(len(sizes))]

        return nx.stochastic_block_model(sizes, probs, seed=self.config.seed)

    def _generate_hierarchical(self, n: int) -> "nx.DiGraph":
        """Generate hierarchical tree-like graph."""
        # Generate a random tree
        G = nx.random_tree(n, seed=self.config.seed)

        # Convert to directed (parent -> child)
        DG = nx.DiGraph()
        root = 0
        visited = {root}
        queue = [root]

        while queue:
            node = queue.pop(0)
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    DG.add_edge(node, neighbor)
                    visited.add(neighbor)
                    queue.append(neighbor)

        return DG

    def _generate_bipartite(self, n: int) -> "nx.Graph":
        """Generate bipartite graph."""
        n1 = n // 2
        n2 = n - n1
        p = 0.1  # Edge probability
        return nx.bipartite.random_graph(n1, n2, p, seed=self.config.seed)

    def _generate_citation(self, n: int) -> "nx.DiGraph":
        """Generate citation network (DAG)."""
        # Generate a random DAG
        G = nx.DiGraph()
        G.add_nodes_from(range(n))

        for i in range(1, n):
            # Each node cites earlier nodes with decaying probability
            for j in range(i):
                # Citations more likely to recent papers
                age_factor = (i - j) / i
                p = 0.1 * (1 - age_factor)  # Higher prob for closer nodes
                if self._random.random() < p:
                    G.add_edge(i, j)  # i cites j

        return G

    def _partition_nodes(self, n: int, k: int) -> List[int]:
        """Partition n nodes into k groups."""
        base_size = n // k
        remainder = n % k
        sizes = [base_size + (1 if i < remainder else 0) for i in range(k)]
        return sizes

    def _add_weights(self, G: "nx.Graph") -> None:
        """Add random weights to edges."""
        for u, v in G.edges():
            G[u][v]["weight"] = round(float(self._rng.uniform(0.1, 1.0)), 3)

    def _add_node_attributes(self, G: "nx.Graph") -> None:
        """Add attributes to nodes."""
        for node in G.nodes():
            G.nodes[node]["id"] = str(node)
            G.nodes[node]["label"] = f"Node_{node}"

    def to_pandas(self, G: "nx.Graph") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert graph to pandas DataFrames.

        Args:
            G: NetworkX graph

        Returns:
            Tuple of (nodes_df, edges_df)
        """
        # Nodes DataFrame
        nodes_data = []
        for node, attrs in G.nodes(data=True):
            row = {"node_id": node, **attrs}
            nodes_data.append(row)
        nodes_df = pd.DataFrame(nodes_data)

        # Edges DataFrame
        edges_data = []
        for u, v, attrs in G.edges(data=True):
            row = {"source": u, "target": v, **attrs}
            edges_data.append(row)
        edges_df = pd.DataFrame(edges_data)

        return nodes_df, edges_df

    def to_cypher(self, G: "nx.Graph", node_label: str = "Node") -> str:
        """Convert graph to Cypher CREATE statements for Neo4j.

        Args:
            G: NetworkX graph
            node_label: Label for nodes

        Returns:
            Cypher query string
        """
        statements = []

        # Create nodes
        for node, attrs in G.nodes(data=True):
            props = ", ".join(f'{k}: "{v}"' if isinstance(v, str) else f"{k}: {v}" for k, v in attrs.items())
            statements.append(f"CREATE (n{node}:{node_label} {{{props}}})")

        # Create edges
        rel_type = "CONNECTED_TO"
        for u, v, attrs in G.edges(data=True):
            props = ", ".join(f'{k}: "{v}"' if isinstance(v, str) else f"{k}: {v}" for k, v in attrs.items())
            if props:
                statements.append(f"CREATE (n{u})-[:{rel_type} {{{props}}}]->(n{v})")
            else:
                statements.append(f"CREATE (n{u})-[:{rel_type}]->(n{v})")

        return ";\n".join(statements) + ";"

    def to_neo4j(
        self,
        G: "nx.Graph",
        uri: str,
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        node_label: str = "Node",
        batch_size: int = 1000,
    ) -> int:
        """Export graph to Neo4j database.

        Args:
            G: NetworkX graph
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Database name
            node_label: Label for nodes
            batch_size: Batch size for transactions

        Returns:
            Number of nodes created
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("neo4j driver is required. Install with: pip install neo4j")

        driver = GraphDatabase.driver(uri, auth=(username, password))

        with driver.session(database=database) as session:
            # Create nodes in batches
            nodes = list(G.nodes(data=True))
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i : i + batch_size]
                session.run(
                    f"""
                    UNWIND $nodes AS node
                    CREATE (n:{node_label})
                    SET n = node.props, n.id = node.id
                    """,
                    nodes=[{"id": n, "props": dict(attrs)} for n, attrs in batch],
                )

            # Create edges in batches
            edges = list(G.edges(data=True))
            for i in range(0, len(edges), batch_size):
                batch = edges[i : i + batch_size]
                session.run(
                    f"""
                    UNWIND $edges AS edge
                    MATCH (a:{node_label} {{id: edge.source}})
                    MATCH (b:{node_label} {{id: edge.target}})
                    CREATE (a)-[r:CONNECTED_TO]->(b)
                    SET r = edge.props
                    """,
                    edges=[{"source": u, "target": v, "props": dict(attrs)} for u, v, attrs in batch],
                )

        driver.close()
        return len(nodes)

    def get_statistics(self, G: "nx.Graph") -> Dict[str, Any]:
        """Get statistics about the generated graph.

        Args:
            G: NetworkX graph

        Returns:
            Dictionary of graph statistics
        """
        stats = {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_connected": nx.is_connected(G) if not G.is_directed() else nx.is_weakly_connected(G),
        }

        if G.number_of_nodes() > 0:
            degrees = [d for _, d in G.degree()]
            stats["avg_degree"] = np.mean(degrees)
            stats["max_degree"] = max(degrees)
            stats["min_degree"] = min(degrees)

            # Clustering coefficient (for undirected graphs)
            if not G.is_directed():
                stats["avg_clustering"] = nx.average_clustering(G)

        return stats


class KnowledgeGraphGenerator:
    """Generator for knowledge graphs with typed entities and relationships.

    Example:
        >>> generator = KnowledgeGraphGenerator(seed=42)
        >>> kg = generator.generate(
        ...     entity_types=["Person", "Company", "Product"],
        ...     relationship_types=[
        ...         ("Person", "works_for", "Company"),
        ...         ("Company", "manufactures", "Product"),
        ...         ("Person", "purchases", "Product"),
        ...     ],
        ...     n_entities_per_type=100,
        ... )
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the knowledge graph generator."""
        _check_networkx_available()

        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._random = random.Random(seed)
        self._attr_generator = NodeAttributeGenerator(seed)

    def generate(
        self,
        entity_types: List[str],
        relationship_types: List[Tuple[str, str, str]],
        n_entities_per_type: int = 50,
        edge_probability: float = 0.1,
    ) -> "nx.MultiDiGraph":
        """Generate a knowledge graph.

        Args:
            entity_types: List of entity type names
            relationship_types: List of (source_type, rel_name, target_type) tuples
            n_entities_per_type: Number of entities per type
            edge_probability: Probability of edge between valid pairs

        Returns:
            NetworkX MultiDiGraph
        """
        G = nx.MultiDiGraph()

        # Create entities
        entity_ids: Dict[str, List[str]] = {etype: [] for etype in entity_types}

        node_id = 0
        for entity_type in entity_types:
            # Get default attributes for this type
            try:
                node_type = NodeType(entity_type)
            except ValueError:
                node_type = NodeType.CUSTOM

            attrs_list = self._attr_generator.generate_attributes(
                node_type, {}, n_entities_per_type
            )

            for attrs in attrs_list:
                G.add_node(
                    node_id,
                    entity_type=entity_type,
                    **attrs,
                )
                entity_ids[entity_type].append(node_id)
                node_id += 1

        # Create relationships
        for source_type, rel_name, target_type in relationship_types:
            source_nodes = entity_ids.get(source_type, [])
            target_nodes = entity_ids.get(target_type, [])

            for source in source_nodes:
                for target in target_nodes:
                    if source != target and self._random.random() < edge_probability:
                        G.add_edge(
                            source,
                            target,
                            relationship_type=rel_name,
                            weight=round(self._rng.uniform(0.1, 1.0), 3),
                        )

        return G

    def generate_from_schema(self, schema: GraphSchema) -> "nx.MultiDiGraph":
        """Generate knowledge graph from a schema definition.

        Args:
            schema: Graph schema

        Returns:
            NetworkX MultiDiGraph
        """
        G = nx.MultiDiGraph()

        # Create entities for each node schema
        entity_ids: Dict[str, List[int]] = {}
        node_id = 0

        for node_schema in schema.node_schemas:
            node_type = (
                node_schema.node_type.value
                if isinstance(node_schema.node_type, NodeType)
                else node_schema.node_type
            )

            n_nodes = self._rng.integers(
                node_schema.count_range[0], node_schema.count_range[1]
            )

            entity_ids[node_type] = []

            attrs_list = self._attr_generator.generate_attributes(
                node_schema.node_type, node_schema.attributes, int(n_nodes)
            )

            for attrs in attrs_list:
                G.add_node(node_id, entity_type=node_type, **attrs)
                entity_ids[node_type].append(node_id)
                node_id += 1

        # Create relationships
        for edge_schema in schema.edge_schemas:
            rel_type = (
                edge_schema.relationship_type.value
                if isinstance(edge_schema.relationship_type, RelationshipType)
                else edge_schema.relationship_type
            )

            for source_type in edge_schema.source_types:
                src_type_str = source_type.value if isinstance(source_type, NodeType) else source_type

                for target_type in edge_schema.target_types:
                    tgt_type_str = target_type.value if isinstance(target_type, NodeType) else target_type

                    source_nodes = entity_ids.get(src_type_str, [])
                    target_nodes = entity_ids.get(tgt_type_str, [])

                    for source in source_nodes:
                        for target in target_nodes:
                            if source != target or edge_schema.allow_self_loops:
                                if self._random.random() < edge_schema.probability:
                                    G.add_edge(
                                        source,
                                        target,
                                        relationship_type=rel_type,
                                    )

        return G


class SocialNetworkGenerator:
    """Specialized generator for social network graphs.

    Generates realistic social networks with:
    - Community structure
    - Power-law degree distribution
    - High clustering coefficient
    - Small-world properties

    Example:
        >>> generator = SocialNetworkGenerator(seed=42)
        >>> G = generator.generate(
        ...     n_users=1000,
        ...     n_communities=10,
        ...     avg_friends=20,
        ... )
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the social network generator."""
        _check_networkx_available()

        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._random = random.Random(seed)
        self._attr_generator = NodeAttributeGenerator(seed)

    def generate(
        self,
        n_users: int = 1000,
        n_communities: int = 10,
        avg_friends: int = 20,
        add_user_attributes: bool = True,
        add_interaction_weights: bool = True,
    ) -> "nx.Graph":
        """Generate a social network.

        Args:
            n_users: Number of users
            n_communities: Number of communities
            avg_friends: Average number of friends per user
            add_user_attributes: Add demographic attributes to users
            add_interaction_weights: Add interaction weights to edges

        Returns:
            NetworkX Graph representing the social network
        """
        # Generate community structure using stochastic block model
        sizes = self._partition_nodes(n_users, n_communities)

        # Within-community and between-community edge probabilities
        p_in = min(2 * avg_friends / (n_users / n_communities), 0.5)
        p_out = p_in / 10

        probs = [
            [p_in if i == j else p_out for j in range(n_communities)]
            for i in range(n_communities)
        ]

        G = nx.stochastic_block_model(sizes, probs, seed=self.seed)

        # Add community labels
        community_id = 0
        node_idx = 0
        for size in sizes:
            for _ in range(size):
                G.nodes[node_idx]["community"] = community_id
                node_idx += 1
            community_id += 1

        # Add user attributes
        if add_user_attributes:
            attrs_list = self._attr_generator.generate_attributes(
                NodeType.PERSON, {}, n_users
            )
            for node, attrs in zip(G.nodes(), attrs_list):
                G.nodes[node].update(attrs)

        # Add interaction weights
        if add_interaction_weights:
            for u, v in G.edges():
                # Higher weight for same community
                same_community = G.nodes[u].get("community") == G.nodes[v].get("community")
                base_weight = 0.5 if same_community else 0.2
                G[u][v]["weight"] = round(
                    base_weight + float(self._rng.uniform(0, 0.5)), 3
                )
                G[u][v]["interaction_count"] = int(self._rng.integers(1, 100))

        return G

    def _partition_nodes(self, n: int, k: int) -> List[int]:
        """Partition n nodes into k groups."""
        base_size = n // k
        remainder = n % k
        return [base_size + (1 if i < remainder else 0) for i in range(k)]


# Convenience functions
def generate_graph(
    graph_type: Union[GraphType, str] = "random",
    n_nodes: int = 100,
    n_edges: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> "nx.Graph":
    """Generate a synthetic graph.

    Args:
        graph_type: Type of graph to generate
        n_nodes: Number of nodes
        n_edges: Number of edges (optional)
        seed: Random seed
        **kwargs: Additional configuration options

    Returns:
        NetworkX graph

    Example:
        >>> G = generate_graph("scale_free", n_nodes=500, seed=42)
    """
    if isinstance(graph_type, str):
        graph_type = GraphType(graph_type)

    config = GraphConfig(
        graph_type=graph_type,
        n_nodes=n_nodes,
        n_edges=n_edges,
        seed=seed,
        **kwargs,
    )

    generator = GraphGenerator(config=config)
    return generator.generate()


def generate_knowledge_graph(
    entity_types: List[str],
    relationship_types: List[Tuple[str, str, str]],
    n_entities: int = 50,
    seed: Optional[int] = None,
) -> "nx.MultiDiGraph":
    """Generate a knowledge graph.

    Args:
        entity_types: List of entity type names
        relationship_types: List of (source, rel, target) tuples
        n_entities: Entities per type
        seed: Random seed

    Returns:
        NetworkX MultiDiGraph

    Example:
        >>> kg = generate_knowledge_graph(
        ...     entity_types=["Person", "Company"],
        ...     relationship_types=[("Person", "works_for", "Company")],
        ...     n_entities=100,
        ... )
    """
    generator = KnowledgeGraphGenerator(seed=seed)
    return generator.generate(entity_types, relationship_types, n_entities)


def generate_social_network(
    n_users: int = 1000,
    n_communities: int = 10,
    seed: Optional[int] = None,
) -> "nx.Graph":
    """Generate a social network graph.

    Args:
        n_users: Number of users
        n_communities: Number of communities
        seed: Random seed

    Returns:
        NetworkX Graph

    Example:
        >>> G = generate_social_network(n_users=500, n_communities=5)
    """
    generator = SocialNetworkGenerator(seed=seed)
    return generator.generate(n_users=n_users, n_communities=n_communities)


class LearnedGraphGenerator:
    """Graph generator that learns from real graph data.

    Extracts statistics and structure from a real graph and generates
    synthetic graphs with similar properties.

    Example:
        >>> from genesis.graph import LearnedGraphGenerator
        >>>
        >>> generator = LearnedGraphGenerator()
        >>> generator.fit(real_graph)
        >>> synthetic = generator.generate(n_nodes=1000)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize learned generator.

        Args:
            seed: Random seed
        """
        _check_networkx_available()
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._is_fitted = False

        # Learned parameters
        self._degree_sequence: List[int] = []
        self._clustering_coeff: float = 0.0
        self._density: float = 0.0
        self._n_nodes: int = 0
        self._n_edges: int = 0
        self._is_directed: bool = False
        self._community_structure: Optional[Dict[int, int]] = None
        self._attribute_distributions: Dict[str, Dict[str, Any]] = {}

    def fit(
        self,
        graph: "nx.Graph",
        detect_communities: bool = True,
        learn_attributes: bool = True,
    ) -> "LearnedGraphGenerator":
        """Fit the generator to a real graph.

        Args:
            graph: NetworkX graph to learn from
            detect_communities: Whether to detect and preserve community structure
            learn_attributes: Whether to learn node/edge attribute distributions

        Returns:
            Self for method chaining
        """
        _check_networkx_available()

        self._n_nodes = graph.number_of_nodes()
        self._n_edges = graph.number_of_edges()
        self._is_directed = graph.is_directed()

        # Learn degree sequence
        if self._is_directed:
            self._degree_sequence = [d for n, d in graph.out_degree()]
        else:
            self._degree_sequence = [d for n, d in graph.degree()]

        # Learn clustering and density
        self._clustering_coeff = nx.average_clustering(graph)
        self._density = nx.density(graph)

        # Detect communities
        if detect_communities:
            try:
                import community as community_louvain

                self._community_structure = community_louvain.best_partition(
                    graph.to_undirected() if self._is_directed else graph
                )
            except ImportError:
                logger.warning("python-louvain not installed, skipping community detection")
                self._community_structure = None

        # Learn attribute distributions
        if learn_attributes:
            self._learn_attributes(graph)

        self._is_fitted = True
        logger.info(
            f"Fitted to graph with {self._n_nodes} nodes, {self._n_edges} edges"
        )
        return self

    def _learn_attributes(self, graph: "nx.Graph") -> None:
        """Learn attribute distributions from graph."""
        # Node attributes
        node_attrs: Dict[str, List[Any]] = {}
        for node, data in graph.nodes(data=True):
            for attr, value in data.items():
                if attr not in node_attrs:
                    node_attrs[attr] = []
                node_attrs[attr].append(value)

        for attr, values in node_attrs.items():
            self._attribute_distributions[f"node_{attr}"] = self._fit_distribution(values)

        # Edge attributes
        edge_attrs: Dict[str, List[Any]] = {}
        for u, v, data in graph.edges(data=True):
            for attr, value in data.items():
                if attr not in edge_attrs:
                    edge_attrs[attr] = []
                edge_attrs[attr].append(value)

        for attr, values in edge_attrs.items():
            self._attribute_distributions[f"edge_{attr}"] = self._fit_distribution(values)

    def _fit_distribution(self, values: List[Any]) -> Dict[str, Any]:
        """Fit distribution to values."""
        if not values:
            return {"type": "empty"}

        # Check if numeric
        try:
            numeric_values = [float(v) for v in values if v is not None]
            if numeric_values:
                return {
                    "type": "numeric",
                    "mean": np.mean(numeric_values),
                    "std": np.std(numeric_values),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                }
        except (ValueError, TypeError):
            pass

        # Categorical
        value_counts: Dict[Any, int] = {}
        for v in values:
            if v is not None:
                value_counts[v] = value_counts.get(v, 0) + 1

        total = sum(value_counts.values())
        return {
            "type": "categorical",
            "categories": list(value_counts.keys()),
            "probabilities": [c / total for c in value_counts.values()],
        }

    def generate(
        self,
        n_nodes: Optional[int] = None,
        preserve_communities: bool = True,
        with_attributes: bool = True,
    ) -> "nx.Graph":
        """Generate a synthetic graph with learned properties.

        Args:
            n_nodes: Number of nodes (default: same as learned)
            preserve_communities: Whether to preserve community structure
            with_attributes: Whether to generate attributes

        Returns:
            Generated NetworkX graph
        """
        if not self._is_fitted:
            raise ValidationError("Generator not fitted. Call fit() first.")

        _check_networkx_available()

        n_nodes = n_nodes or self._n_nodes

        # Scale degree sequence to target size
        if n_nodes != self._n_nodes:
            scaled_degrees = list(
                self._rng.choice(self._degree_sequence, size=n_nodes, replace=True)
            )
        else:
            scaled_degrees = list(self._degree_sequence)

        # Ensure sum is even for undirected graphs
        if not self._is_directed and sum(scaled_degrees) % 2 == 1:
            scaled_degrees[self._rng.integers(n_nodes)] += 1

        # Generate base graph using configuration model
        try:
            if self._is_directed:
                # For directed, need in and out degree sequences
                in_degrees = list(
                    self._rng.choice(scaled_degrees, size=n_nodes, replace=True)
                )
                G = nx.directed_configuration_model(
                    in_degrees, scaled_degrees, seed=self.seed
                )
            else:
                G = nx.configuration_model(scaled_degrees, seed=self.seed)
        except nx.NetworkXError:
            # Fall back to random graph
            logger.warning("Configuration model failed, using random graph")
            p = self._density
            G = nx.erdos_renyi_graph(n_nodes, p, directed=self._is_directed, seed=self.seed)

        # Remove parallel edges and self-loops
        G = nx.Graph(G) if not self._is_directed else nx.DiGraph(G)
        G.remove_edges_from(nx.selfloop_edges(G))

        # Apply community structure if available
        if preserve_communities and self._community_structure:
            G = self._apply_community_structure(G, n_nodes)

        # Add attributes
        if with_attributes:
            self._add_attributes(G)

        return G

    def _apply_community_structure(self, G: "nx.Graph", n_nodes: int) -> "nx.Graph":
        """Apply learned community structure to graph."""
        if not self._community_structure:
            return G

        # Get unique communities
        communities = set(self._community_structure.values())
        n_communities = len(communities)

        # Assign nodes to communities
        community_sizes = [0] * n_communities
        for comm in self._community_structure.values():
            community_sizes[comm] += 1

        # Scale to new size
        total = sum(community_sizes)
        scaled_sizes = [int(n_nodes * s / total) for s in community_sizes]
        scaled_sizes[-1] = n_nodes - sum(scaled_sizes[:-1])  # Adjust for rounding

        # Assign new nodes to communities
        node_communities = {}
        node_idx = 0
        for comm_idx, size in enumerate(scaled_sizes):
            for _ in range(size):
                node_communities[node_idx] = comm_idx
                node_idx += 1

        # Rewire edges to preserve community structure
        # Higher probability within community, lower between
        p_within = min(0.5, self._density * 3)
        p_between = self._density / 3

        nodes = list(G.nodes())
        edges_to_remove = []
        edges_to_add = []

        for u, v in G.edges():
            u_comm = node_communities.get(u, 0)
            v_comm = node_communities.get(v, 0)

            # Randomly rewire some edges
            if self._rng.random() < 0.3:  # 30% rewire rate
                edges_to_remove.append((u, v))

                # Find new target
                if self._rng.random() < p_within / (p_within + p_between):
                    # Within community
                    same_comm = [n for n in nodes if node_communities.get(n, 0) == u_comm]
                    if same_comm:
                        new_v = self._rng.choice(same_comm)
                        if new_v != u:
                            edges_to_add.append((u, new_v))
                else:
                    # Between communities
                    diff_comm = [n for n in nodes if node_communities.get(n, 0) != u_comm]
                    if diff_comm:
                        new_v = self._rng.choice(diff_comm)
                        edges_to_add.append((u, new_v))

        G.remove_edges_from(edges_to_remove)
        G.add_edges_from(edges_to_add)

        # Store community as node attribute
        nx.set_node_attributes(G, node_communities, "community")

        return G

    def _add_attributes(self, G: "nx.Graph") -> None:
        """Add generated attributes to graph."""
        # Node attributes
        for attr_key, dist in self._attribute_distributions.items():
            if not attr_key.startswith("node_"):
                continue

            attr_name = attr_key[5:]  # Remove "node_" prefix
            values = self._sample_distribution(dist, G.number_of_nodes())

            for i, node in enumerate(G.nodes()):
                G.nodes[node][attr_name] = values[i]

        # Edge attributes
        for attr_key, dist in self._attribute_distributions.items():
            if not attr_key.startswith("edge_"):
                continue

            attr_name = attr_key[5:]  # Remove "edge_" prefix
            values = self._sample_distribution(dist, G.number_of_edges())

            for i, (u, v) in enumerate(G.edges()):
                G.edges[u, v][attr_name] = values[i]

    def _sample_distribution(self, dist: Dict[str, Any], n: int) -> List[Any]:
        """Sample n values from distribution."""
        if dist["type"] == "empty":
            return [None] * n

        if dist["type"] == "numeric":
            values = self._rng.normal(dist["mean"], dist["std"], n)
            return np.clip(values, dist["min"], dist["max"]).tolist()

        # Categorical
        return list(
            self._rng.choice(dist["categories"], n, p=dist["probabilities"])
        )

    @property
    def is_fitted(self) -> bool:
        """Check if generator is fitted."""
        return self._is_fitted


@dataclass
class GraphQualityReport:
    """Quality evaluation report for synthetic graphs."""

    real_n_nodes: int
    real_n_edges: int
    synthetic_n_nodes: int
    synthetic_n_edges: int
    degree_ks_statistic: float
    degree_ks_pvalue: float
    clustering_difference: float
    density_ratio: float
    community_nmi: Optional[float]
    overall_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "real_n_nodes": self.real_n_nodes,
            "real_n_edges": self.real_n_edges,
            "synthetic_n_nodes": self.synthetic_n_nodes,
            "synthetic_n_edges": self.synthetic_n_edges,
            "degree_ks_statistic": self.degree_ks_statistic,
            "degree_ks_pvalue": self.degree_ks_pvalue,
            "clustering_difference": self.clustering_difference,
            "density_ratio": self.density_ratio,
            "community_nmi": self.community_nmi,
            "overall_score": self.overall_score,
        }

    def summary(self) -> str:
        """Get summary string."""
        lines = [
            "Graph Quality Report",
            "=" * 40,
            f"Real Graph: {self.real_n_nodes} nodes, {self.real_n_edges} edges",
            f"Synthetic Graph: {self.synthetic_n_nodes} nodes, {self.synthetic_n_edges} edges",
            "",
            "Metrics:",
            f"  Degree KS Statistic: {self.degree_ks_statistic:.4f} (p={self.degree_ks_pvalue:.4f})",
            f"  Clustering Difference: {self.clustering_difference:.4f}",
            f"  Density Ratio: {self.density_ratio:.4f}",
        ]
        if self.community_nmi is not None:
            lines.append(f"  Community NMI: {self.community_nmi:.4f}")
        lines.append("")
        lines.append(f"Overall Score: {self.overall_score:.1%}")
        return "\n".join(lines)


class GraphQualityEvaluator:
    """Evaluates quality of synthetic graphs vs real graphs."""

    def evaluate(
        self,
        real_graph: "nx.Graph",
        synthetic_graph: "nx.Graph",
    ) -> GraphQualityReport:
        """Evaluate synthetic graph quality.

        Args:
            real_graph: Original real graph
            synthetic_graph: Generated synthetic graph

        Returns:
            Quality report
        """
        _check_networkx_available()
        from scipy import stats

        # Basic stats
        real_n = real_graph.number_of_nodes()
        real_e = real_graph.number_of_edges()
        synth_n = synthetic_graph.number_of_nodes()
        synth_e = synthetic_graph.number_of_edges()

        # Degree distribution comparison (KS test)
        real_degrees = [d for n, d in real_graph.degree()]
        synth_degrees = [d for n, d in synthetic_graph.degree()]

        ks_stat, ks_pval = stats.ks_2samp(real_degrees, synth_degrees)

        # Clustering comparison
        real_clustering = nx.average_clustering(real_graph)
        synth_clustering = nx.average_clustering(synthetic_graph)
        clustering_diff = abs(real_clustering - synth_clustering)

        # Density comparison
        real_density = nx.density(real_graph)
        synth_density = nx.density(synthetic_graph)
        density_ratio = synth_density / real_density if real_density > 0 else 0

        # Community structure comparison (if available)
        community_nmi = None
        try:
            import community as community_louvain
            from sklearn.metrics import normalized_mutual_info_score

            real_partition = community_louvain.best_partition(
                real_graph.to_undirected() if real_graph.is_directed() else real_graph
            )
            synth_partition = community_louvain.best_partition(
                synthetic_graph.to_undirected() if synthetic_graph.is_directed() else synthetic_graph
            )

            # Map nodes to common space for comparison
            if len(real_partition) == len(synth_partition):
                real_labels = list(real_partition.values())
                synth_labels = list(synth_partition.values())
                community_nmi = normalized_mutual_info_score(real_labels, synth_labels)
        except ImportError:
            pass

        # Calculate overall score
        # Lower KS statistic is better (max 1)
        degree_score = 1 - min(ks_stat, 1)
        # Lower clustering diff is better
        clustering_score = 1 - min(clustering_diff, 1)
        # Density ratio close to 1 is better
        density_score = 1 - min(abs(1 - density_ratio), 1)

        if community_nmi is not None:
            overall = (degree_score * 0.35 + clustering_score * 0.25 +
                       density_score * 0.2 + community_nmi * 0.2)
        else:
            overall = (degree_score * 0.4 + clustering_score * 0.35 + density_score * 0.25)

        return GraphQualityReport(
            real_n_nodes=real_n,
            real_n_edges=real_e,
            synthetic_n_nodes=synth_n,
            synthetic_n_edges=synth_e,
            degree_ks_statistic=ks_stat,
            degree_ks_pvalue=ks_pval,
            clustering_difference=clustering_diff,
            density_ratio=density_ratio,
            community_nmi=community_nmi,
            overall_score=overall,
        )


class EdgeDifferentialPrivacy:
    """Edge differential privacy for graph data.

    Protects against edge re-identification attacks using randomized response.
    """

    def __init__(self, epsilon: float = 1.0, seed: Optional[int] = None) -> None:
        """Initialize edge DP.

        Args:
            epsilon: Privacy budget (lower = more private)
            seed: Random seed
        """
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)

    def apply(self, graph: "nx.Graph") -> "nx.Graph":
        """Apply edge differential privacy to graph.

        Uses randomized response mechanism.

        Args:
            graph: Input graph

        Returns:
            Privatized graph
        """
        _check_networkx_available()

        # Probability of keeping true value
        p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))

        if graph.is_directed():
            result = nx.DiGraph()
        else:
            result = nx.Graph()

        # Copy nodes with attributes
        for node, data in graph.nodes(data=True):
            result.add_node(node, **data)

        nodes = list(result.nodes())
        n = len(nodes)

        # Apply randomized response to each potential edge
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:] if not graph.is_directed() else nodes:
                if u == v:
                    continue

                has_edge = graph.has_edge(u, v)

                # Randomized response
                if self._rng.random() < p:
                    # Report true value
                    if has_edge:
                        result.add_edge(u, v)
                else:
                    # Report flipped value
                    if not has_edge:
                        result.add_edge(u, v)

        return result

    def estimate_true_density(self, observed_density: float) -> float:
        """Estimate true density from observed (privatized) density.

        Args:
            observed_density: Density of privatized graph

        Returns:
            Estimated true density
        """
        p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
        # observed = p * true + (1-p) * (1 - true)
        # observed = p * true + (1-p) - (1-p) * true
        # observed = true * (2p - 1) + (1 - p)
        # true = (observed - (1-p)) / (2p - 1)
        return (observed_density - (1 - p)) / (2 * p - 1)


class FraudRingGenerator:
    """Generator for synthetic fraud ring networks.

    Creates realistic fraud networks with suspicious patterns like
    circular transactions, burst activity, and account clusters.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        _check_networkx_available()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def generate(
        self,
        n_legitimate: int = 900,
        n_fraudulent: int = 100,
        n_rings: int = 5,
        ring_size_range: Tuple[int, int] = (3, 10),
    ) -> "nx.DiGraph":
        """Generate a transaction network with fraud rings.

        Args:
            n_legitimate: Number of legitimate accounts
            n_fraudulent: Number of fraudulent accounts
            n_rings: Number of fraud rings
            ring_size_range: (min, max) accounts per ring

        Returns:
            DiGraph with transaction edges and fraud labels
        """
        G = nx.DiGraph()

        # Create legitimate accounts
        for i in range(n_legitimate):
            G.add_node(
                f"legit_{i}",
                type="account",
                is_fraud=False,
                account_age=int(self._rng.exponential(365)),
            )

        # Create fraud rings
        fraud_accounts = []
        for ring_id in range(n_rings):
            ring_size = self._rng.integers(ring_size_range[0], ring_size_range[1] + 1)
            ring_nodes = []

            for j in range(ring_size):
                node_id = f"fraud_{ring_id}_{j}"
                G.add_node(
                    node_id,
                    type="account",
                    is_fraud=True,
                    fraud_ring=ring_id,
                    account_age=int(self._rng.exponential(30)),  # Newer accounts
                )
                ring_nodes.append(node_id)
                fraud_accounts.append(node_id)

            # Create circular transactions within ring
            for j in range(len(ring_nodes)):
                source = ring_nodes[j]
                target = ring_nodes[(j + 1) % len(ring_nodes)]
                G.add_edge(
                    source,
                    target,
                    amount=self._rng.exponential(1000),
                    type="circular",
                    timestamp=self._rng.integers(0, 86400),  # Within 24h
                )

        # Add remaining fraudulent accounts not in rings
        remaining_fraud = n_fraudulent - len(fraud_accounts)
        for i in range(remaining_fraud):
            G.add_node(
                f"fraud_solo_{i}",
                type="account",
                is_fraud=True,
                account_age=int(self._rng.exponential(60)),
            )
            fraud_accounts.append(f"fraud_solo_{i}")

        # Create legitimate transactions
        legit_nodes = [n for n in G.nodes() if not G.nodes[n].get("is_fraud", False)]
        n_legit_transactions = n_legitimate * 3

        for _ in range(n_legit_transactions):
            source = self._rng.choice(legit_nodes)
            target = self._rng.choice(legit_nodes)
            if source != target:
                G.add_edge(
                    source,
                    target,
                    amount=self._rng.exponential(200),
                    type="normal",
                    timestamp=self._rng.integers(0, 30 * 86400),
                )

        # Create suspicious connections (fraud to legitimate)
        for fraud_node in fraud_accounts:
            n_targets = self._rng.integers(1, 5)
            targets = self._rng.choice(legit_nodes, n_targets, replace=False)
            for target in targets:
                G.add_edge(
                    fraud_node,
                    target,
                    amount=self._rng.exponential(500),
                    type="suspicious",
                    timestamp=self._rng.integers(0, 7 * 86400),
                )

        return G


__all__ = [
    # Core classes
    "GraphGenerator",
    "KnowledgeGraphGenerator",
    "SocialNetworkGenerator",
    "LearnedGraphGenerator",
    "FraudRingGenerator",
    # Quality evaluation
    "GraphQualityReport",
    "GraphQualityEvaluator",
    # Privacy
    "EdgeDifferentialPrivacy",
    # Configuration
    "GraphConfig",
    "GraphSchema",
    "NodeSchema",
    "EdgeSchema",
    # Enums
    "GraphType",
    "NodeType",
    "RelationshipType",
    # Utilities
    "NodeAttributeGenerator",
    # Convenience functions
    "generate_graph",
    "generate_knowledge_graph",
    "generate_social_network",
    # Constants
    "NETWORKX_AVAILABLE",
]
