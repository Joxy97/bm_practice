"""
Topology utilities for creating Boltzmann Machine graph structures.
"""

import numpy as np
from typing import List, Tuple


def create_fully_connected_topology(
    n_visible: int,
    n_hidden: int = 0
) -> Tuple[List[int], List[Tuple[int, int]], List[int]]:
    """
    Create a fully-connected topology.

    For fully visible BM (n_hidden=0): All visible nodes connected to each other.
    For RBM (n_hidden>0): Bipartite graph - visible to hidden connections only.

    Args:
        n_visible: Number of visible units
        n_hidden: Number of hidden units (0 for fully visible BM)

    Returns:
        nodes: List of all nodes
        edges: List of edges
        hidden_nodes: List of hidden node identifiers
    """
    visible_nodes = list(range(n_visible))
    hidden_nodes = list(range(n_visible, n_visible + n_hidden))
    all_nodes = visible_nodes + hidden_nodes

    edges = []

    if n_hidden == 0:
        # Fully visible: connect all visible nodes to each other
        for i in range(n_visible):
            for j in range(i + 1, n_visible):
                edges.append((i, j))
    else:
        # RBM: bipartite - only visible-to-hidden connections
        for v in visible_nodes:
            for h in hidden_nodes:
                edges.append((v, h))

    return all_nodes, edges, hidden_nodes


def create_restricted_topology(
    n_visible: int,
    n_hidden: int = 0,
    connectivity: float = 0.5,
    seed: int = 42
) -> Tuple[List[int], List[Tuple[int, int]], List[int]]:
    """
    Create a restricted (sparse) topology.

    Args:
        n_visible: Number of visible units
        n_hidden: Number of hidden units (0 for visible-only sparse graph)
        connectivity: Fraction of possible edges to include (0 to 1)
        seed: Random seed for reproducibility

    Returns:
        nodes: List of all nodes
        edges: List of edges
        hidden_nodes: List of hidden node identifiers
    """
    rng = np.random.RandomState(seed)

    visible_nodes = list(range(n_visible))
    hidden_nodes = list(range(n_visible, n_visible + n_hidden))
    all_nodes = visible_nodes + hidden_nodes

    edges = []

    if n_hidden == 0:
        # Sparse visible graph
        for i in range(n_visible):
            for j in range(i + 1, n_visible):
                if rng.random() < connectivity:
                    edges.append((i, j))
    else:
        # Sparse bipartite graph (visible to hidden only)
        for v in visible_nodes:
            for h in hidden_nodes:
                if rng.random() < connectivity:
                    edges.append((v, h))

    # Ensure graph is connected (at least one edge per node)
    node_degrees = {node: 0 for node in all_nodes}
    for u, v in edges:
        node_degrees[u] += 1
        node_degrees[v] += 1

    for node in all_nodes:
        if node_degrees[node] == 0:
            # Connect to random node from opposite set
            if n_hidden > 0:
                if node in visible_nodes:
                    partner = rng.choice(hidden_nodes)
                else:
                    partner = rng.choice(visible_nodes)
                edges.append((node, partner))
            else:
                # For fully visible, connect to any other node
                partner = rng.choice([n for n in all_nodes if n != node])
                edges.append(tuple(sorted([node, partner])))

    return all_nodes, edges, hidden_nodes
