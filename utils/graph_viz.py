"""
Graph visualization utilities for Boltzmann Machine topologies.

Provides functions to visualize BM graph structures using matplotlib and networkx.
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple, Optional
import os


def visualize_bm_graph(
    nodes: List[int],
    edges: List[Tuple[int, int]],
    hidden_nodes: List[int],
    model_type: str,
    connectivity: str = "dense",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize a Boltzmann Machine graph structure.

    Creates a network diagram showing:
    - Visible nodes (blue circles)
    - Hidden nodes (red circles, if present)
    - Edges connecting them

    Args:
        nodes: List of all node identifiers
        edges: List of edge tuples (u, v)
        hidden_nodes: List of hidden node identifiers
        model_type: Type of BM ("fvbm", "rbm", or "sbm")
        connectivity: Connectivity pattern ("dense" or "sparse")
        save_path: Optional path to save the figure
        show: Whether to display the figure
        figsize: Figure size (width, height)
    """
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Separate visible and hidden nodes
    visible_nodes = [n for n in nodes if n not in hidden_nodes]
    n_visible = len(visible_nodes)
    n_hidden = len(hidden_nodes)

    # Create layout based on model type
    pos = _create_layout(visible_nodes, hidden_nodes, model_type)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color='gray',
        alpha=0.3,
        width=1.5,
        ax=ax
    )

    # Draw visible nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=visible_nodes,
        node_color='lightblue',
        node_size=800,
        edgecolors='darkblue',
        linewidths=2,
        ax=ax,
        label='Visible'
    )

    # Draw hidden nodes if present
    if hidden_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=hidden_nodes,
            node_color='lightcoral',
            node_size=800,
            edgecolors='darkred',
            linewidths=2,
            ax=ax,
            label='Hidden'
        )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_weight='bold',
        ax=ax
    )

    # Title and statistics
    model_type_full = {
        'fvbm': 'Fully Visible BM',
        'rbm': 'Restricted BM',
        'sbm': 'Standard BM'
    }
    title = f"{model_type_full.get(model_type, model_type.upper())} - {connectivity.capitalize()} Connectivity"

    # Calculate edge type breakdown for SBM
    edge_info = f"\nNodes: {n_visible} visible"
    if n_hidden > 0:
        edge_info += f", {n_hidden} hidden"
    edge_info += f"\nEdges: {len(edges)} total"

    if model_type == "sbm" and len(edges) > 0:
        # Count edge types
        vv_edges = sum(1 for u, v in edges if u in visible_nodes and v in visible_nodes)
        vh_edges = sum(1 for u, v in edges if (u in visible_nodes and v in hidden_nodes) or
                                                 (u in hidden_nodes and v in visible_nodes))
        hh_edges = sum(1 for u, v in edges if u in hidden_nodes and v in hidden_nodes)
        edge_info += f"\n  ({vv_edges} v-v, {vh_edges} v-h, {hh_edges} h-h)"

    ax.set_title(title + edge_info, fontsize=14, fontweight='bold', pad=20)

    # Legend
    ax.legend(loc='upper right', fontsize=12)

    # Remove axes
    ax.axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Graph visualization saved to: {save_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def _create_layout(
    visible_nodes: List[int],
    hidden_nodes: List[int],
    model_type: str
) -> dict:
    """
    Create node positions based on model type.

    Args:
        visible_nodes: List of visible node identifiers
        hidden_nodes: List of hidden node identifiers
        model_type: Type of BM ("fvbm", "rbm", or "sbm")

    Returns:
        Dictionary mapping node IDs to (x, y) positions
    """
    pos = {}
    n_visible = len(visible_nodes)
    n_hidden = len(hidden_nodes)

    if model_type == "fvbm":
        # Circular layout for visible nodes only
        import numpy as np
        angles = np.linspace(0, 2 * np.pi, n_visible, endpoint=False)
        for i, node in enumerate(visible_nodes):
            pos[node] = (np.cos(angles[i]), np.sin(angles[i]))

    elif model_type in ["rbm", "sbm"]:
        # Bipartite-style layout with two rows
        # Visible nodes on bottom, hidden nodes on top

        # Visible nodes (bottom row)
        visible_spacing = 2.0 / (n_visible + 1) if n_visible > 1 else 1.0
        for i, node in enumerate(visible_nodes):
            x = -1.0 + visible_spacing * (i + 1)
            pos[node] = (x, -0.5)

        # Hidden nodes (top row)
        hidden_spacing = 2.0 / (n_hidden + 1) if n_hidden > 1 else 1.0
        for i, node in enumerate(hidden_nodes):
            x = -1.0 + hidden_spacing * (i + 1)
            pos[node] = (x, 0.5)

    return pos


def visualize_topology_from_config(
    config: dict,
    model_key: str = 'true_model',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Visualize a BM topology directly from a configuration dictionary.

    Args:
        config: Configuration dictionary containing model specifications
        model_key: Key in config for model parameters ('true_model' or 'learned_model')
        save_path: Optional path to save the figure
        show: Whether to display the figure
    """
    from .topology import create_topology

    model_config = config[model_key]

    # Create topology
    nodes, edges, hidden_nodes = create_topology(
        n_visible=model_config['n_visible'],
        n_hidden=model_config['n_hidden'],
        model_type=model_config['model_type'],
        connectivity=model_config['connectivity'],
        connectivity_density=model_config.get('connectivity_density', 0.5),
        seed=config.get('seed', 42)
    )

    # Visualize
    visualize_bm_graph(
        nodes=nodes,
        edges=edges,
        hidden_nodes=hidden_nodes,
        model_type=model_config['model_type'],
        connectivity=model_config['connectivity'],
        save_path=save_path,
        show=show
    )


def compare_topologies(
    configs: List[Tuple[dict, str, str]],
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (18, 6)
) -> None:
    """
    Compare multiple BM topologies side-by-side.

    Args:
        configs: List of tuples (config_dict, model_key, title)
        save_path: Optional path to save the figure
        show: Whether to display the figure
        figsize: Figure size (width, height)
    """
    from .topology import create_topology

    n_configs = len(configs)
    fig, axes = plt.subplots(1, n_configs, figsize=figsize)

    if n_configs == 1:
        axes = [axes]

    for ax, (config, model_key, title) in zip(axes, configs):
        model_config = config[model_key]

        # Create topology
        nodes, edges, hidden_nodes = create_topology(
            n_visible=model_config['n_visible'],
            n_hidden=model_config['n_hidden'],
            model_type=model_config['model_type'],
            connectivity=model_config['connectivity'],
            connectivity_density=model_config.get('connectivity_density', 0.5),
            seed=config.get('seed', 42)
        )

        # Separate visible and hidden
        visible_nodes = [n for n in nodes if n not in hidden_nodes]

        # Create graph
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        # Layout
        pos = _create_layout(visible_nodes, hidden_nodes, model_config['model_type'])

        # Draw
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, width=1.5, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=visible_nodes, node_color='lightblue',
                               node_size=600, edgecolors='darkblue', linewidths=2, ax=ax)

        if hidden_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_color='lightcoral',
                                   node_size=600, edgecolors='darkred', linewidths=2, ax=ax)

        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)

        # Title with stats
        model_type_full = {
            'fvbm': 'FVBM',
            'rbm': 'RBM',
            'sbm': 'SBM'
        }
        stats = f"{model_type_full.get(model_config['model_type'], 'BM')}"
        stats += f"\n{len(visible_nodes)}v"
        if hidden_nodes:
            stats += f", {len(hidden_nodes)}h"
        stats += f", {len(edges)}e"

        ax.set_title(f"{title}\n{stats}", fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
