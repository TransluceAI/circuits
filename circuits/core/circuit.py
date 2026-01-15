"""
Utilities for representing and visualizing CLSO circuits.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from circuits.core.utils import Order
from pydantic import BaseModel


class CircuitNode(BaseModel):
    layer: int
    position: int
    neuron_idx: int
    label: str
    activations: list[float]
    attribution: list[list[float]]
    polarity: str

    def __hash__(self) -> int:
        """Make CircuitNode hashable based on its unique identifier."""
        return hash((self.layer, self.position, self.neuron_idx))

    def __eq__(self, other) -> bool:
        """Define equality based on unique identifier."""
        if not isinstance(other, CircuitNode):
            return False
        return (self.layer, self.position, self.neuron_idx) == (
            other.layer,
            other.position,
            other.neuron_idx,
        )


class CircuitEdge(BaseModel):
    source: CircuitNode
    target: CircuitNode
    weight: float | list[float]
    order: Order
    absolute_weight: float | list[float] | None = None


class CircuitGraph:
    """
    A graph data structure for storing and analyzing traced circuits.

    Nodes are represented as CircuitNode objects.
    Edges are represented as CircuitEdge objects.
    """

    def __init__(self, db=None, neuron_label_cache=None, token_strings=None, focus_tokens=None):
        self.nodes: Set[CircuitNode] = set()
        self.edges: List[CircuitEdge] = []
        self.db = db  # Database connection for querying neuron labels
        self.neuron_label_cache = neuron_label_cache or {}  # Pre-fetched neuron labels
        self.token_strings = token_strings  # Token strings for each node
        self.focus_tokens = focus_tokens

        # For efficient lookups - using node keys for compatibility with existing code
        self._node_by_key: Dict[Tuple[int, int, int], CircuitNode] = {}
        self._incoming_edges: Dict[CircuitNode, List[CircuitEdge]] = defaultdict(list)
        self._outgoing_edges: Dict[CircuitNode, List[CircuitEdge]] = defaultdict(list)
        self._built = False

    def _get_node_key(self, node: CircuitNode) -> Tuple[int, int, int]:
        """Get the tuple key for a node."""
        return (node.layer, node.position, node.neuron_idx)

    def add_node(
        self,
        node: Tuple[int, int, int] | CircuitNode,
        activations: list[float] | None = None,
        attribution: list[float] | None = None,
        polarity: str | None = None,
    ) -> None:
        """Add a node to the graph. Can accept either a tuple or CircuitNode."""
        if isinstance(node, tuple):
            layer, position, neuron_idx = node

            # Get label from cache if available
            cache_key = (layer, neuron_idx)
            label = self.neuron_label_cache.get(cache_key, "N.A.")

            circuit_node = CircuitNode(
                layer=layer,
                position=position,
                neuron_idx=neuron_idx,
                label=label,
                activations=activations or [],
                attribution=attribution or [],
                polarity=polarity or "N.A.",
            )
        else:
            circuit_node = node

        node_key = self._get_node_key(circuit_node)
        if node_key not in self._node_by_key:
            self.nodes.add(circuit_node)
            self._node_by_key[node_key] = circuit_node
            self._built = False

    def add_edge(
        self,
        source: Tuple[int, int, int] | CircuitNode,
        target: Tuple[int, int, int] | CircuitNode,
        weight: float,
        order: Order,
        absolute_weight: float | None = None,
    ) -> None:
        """Add a weighted edge from source to target node."""
        # Convert tuples to CircuitNode if needed
        if isinstance(source, tuple):
            source_key = source
            if source_key not in self._node_by_key:
                self.add_node(source)
            source_node = self._node_by_key[source_key]
        else:
            source_node = source
            source_key = self._get_node_key(source_node)
            if source_key not in self._node_by_key:
                self.add_node(source_node)

        if isinstance(target, tuple):
            target_key = target
            if target_key not in self._node_by_key:
                self.add_node(target)
            target_node = self._node_by_key[target_key]
        else:
            target_node = target
            target_key = self._get_node_key(target_node)
            if target_key not in self._node_by_key:
                self.add_node(target_node)

        edge = CircuitEdge(
            source=source_node,
            target=target_node,
            weight=weight,
            order=order,
            absolute_weight=absolute_weight,
        )

        self.edges.append(edge)
        self._built = False

    def _build_adjacency_maps(self) -> None:
        """Build internal adjacency maps for efficient queries."""
        if self._built:
            return

        self._incoming_edges.clear()
        self._outgoing_edges.clear()

        for edge in self.edges:
            self._incoming_edges[edge.target].append(edge)
            self._outgoing_edges[edge.source].append(edge)

        self._built = True

    def get_incoming_edges(self, node: Tuple[int, int, int] | CircuitNode) -> List[CircuitEdge]:
        """Get all incoming edges to a node."""
        self._build_adjacency_maps()
        if isinstance(node, tuple):
            node = self._node_by_key[node]
        return self._incoming_edges[node].copy()

    def get_outgoing_edges(self, node: Tuple[int, int, int] | CircuitNode) -> List[CircuitEdge]:
        """Get all outgoing edges from a node."""
        self._build_adjacency_maps()
        if isinstance(node, tuple):
            node = self._node_by_key[node]
        return self._outgoing_edges[node].copy()

    def get_parents(
        self, node: Tuple[int, int, int] | CircuitNode
    ) -> List[Tuple[CircuitNode, float]]:
        """Get all parent nodes with edge weights."""
        incoming_edges = self.get_incoming_edges(node)
        return [(edge.source, edge.weight) for edge in incoming_edges]

    def get_children(
        self, node: Tuple[int, int, int] | CircuitNode
    ) -> List[Tuple[CircuitNode, float]]:
        """Get all child nodes with edge weights."""
        outgoing_edges = self.get_outgoing_edges(node)
        return [(edge.target, edge.weight) for edge in outgoing_edges]

    def get_nodes_by_layer(self, layer: int) -> List[CircuitNode]:
        """Get all nodes in a specific layer."""
        return [node for node in self.nodes if node.layer == layer]

    def get_nodes_by_position(self, position: int) -> List[CircuitNode]:
        """Get all nodes at a specific position."""
        return [node for node in self.nodes if node.position == position]

    def get_leaves(self) -> List[CircuitNode]:
        """Get all nodes that have no parents."""
        return [node for node in self.nodes if len(self.get_parents(node)) == 0]

    def get_layers(self) -> List[int]:
        """Get all unique layers in the graph, sorted."""
        return sorted(list(set(node.layer for node in self.nodes)))

    def get_positions(self) -> List[int]:
        """Get all unique positions in the graph, sorted."""
        return sorted(list(set(node.position for node in self.nodes)))

    def get_node_info(self, node: Tuple[int, int, int] | CircuitNode) -> Dict:
        """Get comprehensive information about a node."""
        if isinstance(node, tuple):
            circuit_node = self._node_by_key[node]
        else:
            circuit_node = node

        parents = self.get_parents(circuit_node)
        children = self.get_children(circuit_node)

        return {
            "layer": circuit_node.layer,
            "position": circuit_node.position,
            "neuron_idx": circuit_node.neuron_idx,
            "label": circuit_node.label,
            "num_parents": len(parents),
            "num_children": len(children),
            "total_incoming_weight": np.sum([weight for _, weight in parents]),
            "total_outgoing_weight": np.sum([weight for _, weight in children]),
            "parents": parents,
            "children": children,
            "polarity": circuit_node.polarity,
        }

    def get_node_label(self, node: Tuple[int, int, int] | CircuitNode) -> str:
        """Get the label for a specific node."""
        if isinstance(node, tuple):
            circuit_node = self._node_by_key[node]
        else:
            circuit_node = node
        return circuit_node.label

    def get_strongest_paths(
        self, start_node: Tuple[int, int, int] | CircuitNode, max_depth: int = 3
    ) -> List[List[CircuitNode]]:
        """Get the strongest paths starting from a node (by absolute weight)."""
        if isinstance(start_node, tuple):
            start_node = self._node_by_key[start_node]

        paths = []

        def dfs(current_path: List[CircuitNode], current_node: CircuitNode, depth: int):
            if depth >= max_depth:
                return

            children = self.get_children(current_node)
            if not children:
                if len(current_path) > 1:  # Only add paths with more than one node
                    paths.append(current_path.copy())
                return

            # Sort children by absolute weight (strongest first)
            children.sort(key=lambda x: abs(x[1]), reverse=True)

            for child_node, weight in children[:3]:  # Take top 3 strongest children
                if child_node not in current_path:  # Avoid cycles
                    current_path.append(child_node)
                    dfs(current_path, child_node, depth + 1)
                    current_path.pop()

        dfs([start_node], start_node, 0)
        return paths

    def list_all_nodes(self) -> List[CircuitNode]:
        """List all nodes in the graph, sorted by layer, then position, then neuron_idx."""
        return sorted(list(self.nodes), key=lambda n: (n.layer, n.position, n.neuron_idx))

    def _node_to_id(self, node: CircuitNode) -> str:
        """Convert a node to a string ID for JSON serialization."""
        return f"{node.layer}_{node.position}_{node.neuron_idx}"

    def _id_to_node_key(self, node_id: str) -> Tuple[int, int, int]:
        """Convert a string ID back to a node key tuple for JSON deserialization."""
        parts = node_id.split("_")
        return (int(parts[0]), int(parts[1]), int(parts[2]))

    def serialize_to_json(self, filepath: str | Path) -> None:
        """
        Serialize the circuit graph to a JSON file.

        Args:
            filepath: Path where to save the JSON file
        """
        # Convert nodes to serializable format
        nodes_data = []
        for node in self.list_all_nodes():
            node_id = self._node_to_id(node)
            node_data = {
                "id": node_id,
                "layer": node.layer,
                "position": node.position,
                "neuron_idx": node.neuron_idx,
                "label": node.label,
                "activations": node.activations,
                "attribution": node.attribution,
                "polarity": node.polarity,
            }
            nodes_data.append(node_data)

        # Convert edges to serializable format
        edges_data = []
        for edge in self.edges:
            edge_data = {
                "source": self._node_to_id(edge.source),
                "target": self._node_to_id(edge.target),
                "weight": edge.weight,
                "order": edge.order.value,
                "absolute_weight": edge.absolute_weight,
            }
            edges_data.append(edge_data)

        # Create the complete graph data structure
        graph_data = {
            "metadata": {
                "num_nodes": len(self.nodes),
                "num_edges": len(self.edges),
                "layers": self.get_layers(),
                "positions": self.get_positions(),
            },
            "nodes": nodes_data,
            "edges": edges_data,
            "token_strings": self.token_strings,
        }

        # Write to JSON file
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            json.dump(graph_data, f, indent=2)

    @classmethod
    def deserialize_from_json(
        cls, filepath: str | Path, db=None, neuron_label_cache=None
    ) -> "CircuitGraph":
        """
        Deserialize a circuit graph from a JSON file.

        Args:
            filepath: Path to the JSON file
            db: Optional database connection for querying neuron labels

        Returns:
            CircuitGraph object reconstructed from the JSON file
        """
        filepath = Path(filepath)

        with open(filepath, "r") as f:
            graph_data = json.load(f)

        # Create new CircuitGraph instance
        circuit = cls(
            db=db, neuron_label_cache=neuron_label_cache, token_strings=graph_data["token_strings"]
        )

        # Reconstruct nodes
        for node_data in graph_data["nodes"]:
            node = CircuitNode(
                layer=node_data["layer"],
                position=node_data["position"],
                neuron_idx=node_data["neuron_idx"],
                label=node_data["label"],
                activations=node_data.get("activations", []),
                attribution=node_data.get("attribution", []),
                polarity=node_data.get("polarity", "N.A."),
            )
            circuit.add_node(node)

        # Reconstruct edges
        for edge_data in graph_data["edges"]:
            source_key = circuit._id_to_node_key(edge_data["source"])
            target_key = circuit._id_to_node_key(edge_data["target"])
            source_node = circuit._node_by_key[source_key]
            target_node = circuit._node_by_key[target_key]
            weight = edge_data["weight"]
            order = Order(edge_data["order"])
            absolute_weight = edge_data.get("absolute_weight")
            circuit.add_edge(source_node, target_node, weight, order, absolute_weight)

        return circuit

    def summary(self) -> Dict:
        """Get summary statistics of the graph."""
        layers = self.get_layers()
        positions = self.get_positions()

        # Calculate degree statistics
        in_degrees = [len(self.get_parents(node)) for node in self.nodes]
        out_degrees = [len(self.get_children(node)) for node in self.nodes]

        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "num_layers": len(layers),
            "layer_range": (min(layers), max(layers)) if layers else (None, None),
            "num_positions": len(positions),
            "position_range": (min(positions), max(positions)) if positions else (None, None),
            "avg_in_degree": sum(in_degrees) / len(in_degrees) if in_degrees else 0,
            "avg_out_degree": sum(out_degrees) / len(out_degrees) if out_degrees else 0,
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
        }

    def __len__(self) -> int:
        """Return number of nodes."""
        return len(self.nodes)

    def __str__(self) -> str:
        """String representation of the graph."""
        summary = self.summary()
        return (
            f"CircuitGraph({summary['num_nodes']} nodes, "
            f"{summary['num_edges']} edges, layers {summary['layer_range']})"
        )


def generate_circuit_html(
    circuit: CircuitGraph,
    output_path: str | Path = "circuit_visualization.html",
    title: str = "Circuit Visualization",
    batch_labels: List[str] = [],
) -> str:
    """
    Generate an interactive HTML visualization of a CircuitGraph with right-aligned node positioning.
    """

    # Prepare data for visualization
    nodes_data = []
    edges_data = []

    # Get all nodes and their information
    for node in circuit.list_all_nodes():
        node_info = circuit.get_node_info(node)

        # Get token string for this position if available
        token_text = ""
        if circuit.token_strings and node.position < len(circuit.token_strings):
            token_text = circuit.token_strings[node.position]

        nodes_data.append(
            {
                "id": f"{node.layer}_{node.position}_{node.neuron_idx}",
                "layer": node.layer,
                "position": node.position,
                "neuron_idx": node.neuron_idx,
                "label": node.label,
                "token_text": token_text,
                "num_parents": node_info["num_parents"],
                "num_children": node_info["num_children"],
                "total_incoming_weight": node_info["total_incoming_weight"],
                "total_outgoing_weight": node_info["total_outgoing_weight"],
                "activations": node.activations,
                "attribution": node.attribution,
                "polarity": node.polarity,
            }
        )

    # Process edges and normalize weights for opacity
    edge_weights_by_target = {}

    # First pass: collect all incoming weights for each target node
    for edge in circuit.edges:
        target_id = f"{edge.target.layer}_{edge.target.position}_{edge.target.neuron_idx}"
        if target_id not in edge_weights_by_target:
            edge_weights_by_target[target_id] = []
        edge_weights_by_target[target_id].append(np.mean(np.abs(edge.weight)))

    # Normalize weights for each target node
    normalized_weights = {}
    for target_id, weights in edge_weights_by_target.items():
        if len(weights) > 0:
            # Store normalized weights
            normalized_weights[target_id] = []
            for w in weights:
                # if weight_range > 0:
                #     norm_w = (w - min_weight) / weight_range
                # else:
                #     norm_w = 1.0
                # print(w)
                normalized_weights[target_id].append(min(1.0, np.mean(np.abs(w))))

    # Second pass: create edge data with normalized opacity
    edge_idx = 0
    for edge in circuit.edges:
        source_id = f"{edge.source.layer}_{edge.source.position}_{edge.source.neuron_idx}"
        target_id = f"{edge.target.layer}_{edge.target.position}_{edge.target.neuron_idx}"

        # Get normalized opacity
        opacity = 0.0  # default
        if target_id in normalized_weights:
            target_weights = edge_weights_by_target[target_id]
            target_idx = -1
            for i, w in enumerate(target_weights):
                if abs(w - np.mean(np.abs(edge.weight))) < 1e-10:  # floating point comparison
                    target_idx = i
                    break
            if target_idx >= 0:
                opacity = 0.0 + 1.0 * normalized_weights[target_id][target_idx]

        edges_data.append(
            {
                "source": source_id,
                "target": target_id,
                "weight": edge.weight,
                "opacity": opacity,
                "order": edge.order.value,
                "absolute_weight": edge.absolute_weight,
            }
        )
        edge_idx += 1

    # Get token positions for x-axis
    token_positions = sorted(list(set(node.position for node in circuit.nodes)))
    token_labels = []
    for pos in token_positions:
        if circuit.token_strings and pos < len(circuit.token_strings):
            token_labels.append(circuit.token_strings[pos])
        else:
            token_labels.append(f"Pos_{pos}")

    # Read the HTML template
    template_path = Path(__file__).parent / "circuit_template.html"
    with open(template_path, "r", encoding="utf-8") as f:
        html_template = f.read()

    # Format the template with the data
    mapping: Dict[str, str] = {
        "title": title,
        "nodes_data": json.dumps(nodes_data),
        "edges_data": json.dumps(edges_data),
        "token_positions": json.dumps(token_positions),
        "token_labels": json.dumps(token_labels),
        "focus_tokens": json.dumps(circuit.focus_tokens or []),
        "batch_labels": json.dumps(batch_labels),
    }
    for key, value in mapping.items():
        html_template = html_template.replace("{{ " + key + " }}", value)
    html_content = html_template

    # Write HTML file
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Circuit visualization saved to: {output_path}")
    return str(output_path)
