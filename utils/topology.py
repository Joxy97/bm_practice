"""
Topology utilities for creating Boltzmann Machine graph structures.

New naming scheme (Phase 1):
- model_type: "fvbm" (Fully Visible BM) | "rbm" (Restricted BM)
- connectivity: "dense" | "sparse"
- connectivity_density: 0.0-1.0 (for sparse only)
"""

import numpy as np
from typing import List, Tuple, Literal


def create_topology(
    n_visible: int,
    n_hidden: int,
    model_type: Literal["fvbm", "rbm"],
    connectivity: Literal["dense", "sparse"] = "dense",
    connectivity_density: float = 0.5,
    seed: int = 42
) -> Tuple[List[int], List[Tuple[int, int]], List[int]]:
    """
    Create a Boltzmann Machine topology using the new naming scheme.

    Model Types:
    - "fvbm": Fully Visible Boltzmann Machine (n_hidden must be 0)
              Edges between visible nodes only
    - "rbm": Restricted Boltzmann Machine (n_hidden must be > 0)
             Bipartite graph - only visible-to-hidden edges

    Connectivity:
    - "dense": All allowed edges exist
    - "sparse": Random subset of allowed edges (controlled by connectivity_density)

    Args:
        n_visible: Number of visible units
        n_hidden: Number of hidden units
        model_type: Type of Boltzmann Machine ("fvbm" or "rbm")
        connectivity: Connectivity pattern ("dense" or "sparse")
        connectivity_density: Fraction of edges to include for sparse (0.0-1.0)
        seed: Random seed for reproducibility (used for sparse only)

    Returns:
        Tuple of:
        - nodes: List of all node identifiers
        - edges: List of edge tuples (u, v)
        - hidden_nodes: List of hidden node identifiers (empty for FVBM)

    Raises:
        ValueError: If model_type and n_hidden are inconsistent
    """
    # Validate model_type and n_hidden consistency
    if model_type == "fvbm" and n_hidden != 0:
        raise ValueError(f"FVBM requires n_hidden=0, got n_hidden={n_hidden}")
    if model_type == "rbm" and n_hidden == 0:
        raise ValueError(f"RBM requires n_hidden>0, got n_hidden={n_hidden}")

    # Validate connectivity_density
    if connectivity == "sparse" and not (0.0 <= connectivity_density <= 1.0):
        raise ValueError(f"connectivity_density must be in [0, 1], got {connectivity_density}")

    # Create node lists
    visible_nodes = list(range(n_visible))
    hidden_nodes = list(range(n_visible, n_visible + n_hidden))
    all_nodes = visible_nodes + hidden_nodes

    # Generate edges based on model_type and connectivity
    if connectivity == "dense":
        edges = _create_dense_edges(visible_nodes, hidden_nodes, model_type)
    else:  # sparse
        edges = _create_sparse_edges(
            visible_nodes, hidden_nodes, model_type, connectivity_density, seed
        )

    return all_nodes, edges, hidden_nodes


def _create_dense_edges(
    visible_nodes: List[int],
    hidden_nodes: List[int],
    model_type: str
) -> List[Tuple[int, int]]:
    """Create all allowed edges for the given model type."""
    edges = []

    if model_type == "fvbm":
        # Dense FVBM: all visible-visible edges
        for i in range(len(visible_nodes)):
            for j in range(i + 1, len(visible_nodes)):
                edges.append((visible_nodes[i], visible_nodes[j]))

    elif model_type == "rbm":
        # Dense RBM: all visible-hidden edges (bipartite)
        for v in visible_nodes:
            for h in hidden_nodes:
                edges.append((v, h))

    return edges


def _create_sparse_edges(
    visible_nodes: List[int],
    hidden_nodes: List[int],
    model_type: str,
    connectivity_density: float,
    seed: int
) -> List[Tuple[int, int]]:
    """Create sparse edges by randomly sampling from allowed edges."""
    rng = np.random.RandomState(seed)
    edges = []

    if model_type == "fvbm":
        # Sparse FVBM: random subset of visible-visible edges
        for i in range(len(visible_nodes)):
            for j in range(i + 1, len(visible_nodes)):
                if rng.random() < connectivity_density:
                    edges.append((visible_nodes[i], visible_nodes[j]))

    elif model_type == "rbm":
        # Sparse RBM: random subset of visible-hidden edges
        for v in visible_nodes:
            for h in hidden_nodes:
                if rng.random() < connectivity_density:
                    edges.append((v, h))

    # Ensure connectivity: every node has at least one edge
    all_nodes = visible_nodes + hidden_nodes
    node_degrees = {node: 0 for node in all_nodes}
    for u, v in edges:
        node_degrees[u] += 1
        node_degrees[v] += 1

    for node in all_nodes:
        if node_degrees[node] == 0:
            # Add edge to ensure connectivity
            if model_type == "fvbm":
                # Connect to random other visible node
                partner = rng.choice([n for n in visible_nodes if n != node])
                edges.append(tuple(sorted([node, partner])))
            else:  # rbm
                # Connect to random node from opposite layer
                if node in visible_nodes:
                    partner = rng.choice(hidden_nodes)
                else:
                    partner = rng.choice(visible_nodes)
                edges.append((node, partner))

    return edges


# Legacy functions (deprecated - kept for backward compatibility during migration)
def create_fully_connected_topology(
    n_visible: int,
    n_hidden: int = 0
) -> Tuple[List[int], List[Tuple[int, int]], List[int]]:
    """
    DEPRECATED: Use create_topology() with model_type and connectivity instead.

    Legacy function for backward compatibility.
    """
    model_type = "fvbm" if n_hidden == 0 else "rbm"
    return create_topology(n_visible, n_hidden, model_type, "dense")


def create_restricted_topology(
    n_visible: int,
    n_hidden: int = 0,
    connectivity: float = 0.5,
    seed: int = 42
) -> Tuple[List[int], List[Tuple[int, int]], List[int]]:
    """
    DEPRECATED: Use create_topology() with connectivity="sparse" instead.

    Legacy function for backward compatibility.
    """
    model_type = "fvbm" if n_hidden == 0 else "rbm"
    return create_topology(n_visible, n_hidden, model_type, "sparse", connectivity, seed)
