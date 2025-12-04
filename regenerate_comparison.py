"""
Regenerate architecture comparison visualization with curved edges.
"""

from utils import compare_topologies, load_config

# Load base config
base_config = load_config('configs/config.yaml')

# Create 6 configurations: Dense and Sparse for each model type
configs = []

# Dense FVBM
config_fvbm_dense = {
    'seed': 42,
    'true_model': {
        'n_visible': 10,
        'n_hidden': 0,
        'model_type': 'fvbm',
        'connectivity': 'dense'
    }
}
configs.append((config_fvbm_dense, 'true_model', 'Dense FVBM'))

# Dense RBM
config_rbm_dense = {
    'seed': 42,
    'true_model': {
        'n_visible': 6,
        'n_hidden': 4,
        'model_type': 'rbm',
        'connectivity': 'dense'
    }
}
configs.append((config_rbm_dense, 'true_model', 'Dense RBM'))

# Dense SBM
config_sbm_dense = {
    'seed': 42,
    'true_model': {
        'n_visible': 6,
        'n_hidden': 4,
        'model_type': 'sbm',
        'connectivity': 'dense'
    }
}
configs.append((config_sbm_dense, 'true_model', 'Dense SBM'))

# Sparse FVBM
config_fvbm_sparse = {
    'seed': 42,
    'true_model': {
        'n_visible': 10,
        'n_hidden': 0,
        'model_type': 'fvbm',
        'connectivity': 'sparse',
        'connectivity_density': 0.4
    }
}
configs.append((config_fvbm_sparse, 'true_model', 'Sparse FVBM'))

# Sparse RBM
config_rbm_sparse = {
    'seed': 42,
    'true_model': {
        'n_visible': 6,
        'n_hidden': 4,
        'model_type': 'rbm',
        'connectivity': 'sparse',
        'connectivity_density': 0.4
    }
}
configs.append((config_rbm_sparse, 'true_model', 'Sparse RBM'))

# Sparse SBM
config_sbm_sparse = {
    'seed': 42,
    'true_model': {
        'n_visible': 6,
        'n_hidden': 4,
        'model_type': 'sbm',
        'connectivity': 'sparse',
        'connectivity_density': 0.4
    }
}
configs.append((config_sbm_sparse, 'true_model', 'Sparse SBM'))

# Create comparison in 2x3 grid
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20, 12))

# Top row: Dense architectures
for i, config_tuple in enumerate(configs[:3]):
    ax = plt.subplot(2, 3, i + 1)

    from utils.topology import create_topology
    import networkx as nx
    from utils.graph_viz import _create_layout, _draw_bipartite_edges

    config, model_key, title = config_tuple
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

    if model_config['model_type'] in ["rbm", "sbm"]:
        _draw_bipartite_edges(ax, G, pos, visible_nodes, hidden_nodes)
    else:
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, width=1.5, ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=visible_nodes, node_color='lightblue',
                          node_size=600, edgecolors='darkblue', linewidths=2, ax=ax)

    if hidden_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_color='lightcoral',
                              node_size=600, edgecolors='darkred', linewidths=2, ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)

    # Title with stats
    model_type_full = {'fvbm': 'FVBM', 'rbm': 'RBM', 'sbm': 'SBM'}
    stats = f"{model_type_full.get(model_config['model_type'], 'BM')}"
    stats += f"\n{len(visible_nodes)}v"
    if hidden_nodes:
        stats += f", {len(hidden_nodes)}h"
    stats += f", {len(edges)}e"

    # Add edge type breakdown for SBM
    if model_config['model_type'] == 'sbm' and len(edges) > 0:
        vv_edges = sum(1 for u, v in edges if u in visible_nodes and v in visible_nodes)
        vh_edges = sum(1 for u, v in edges if (u in visible_nodes and v in hidden_nodes) or
                                                 (u in hidden_nodes and v in visible_nodes))
        hh_edges = sum(1 for u, v in edges if u in hidden_nodes and v in hidden_nodes)
        stats += f"\n({vv_edges}v-v, {vh_edges}v-h, {hh_edges}h-h)"

    ax.set_title(f"{title}\n{stats}", fontsize=12, fontweight='bold')
    ax.axis('off')

# Bottom row: Sparse architectures
for i, config_tuple in enumerate(configs[3:]):
    ax = plt.subplot(2, 3, i + 4)

    config, model_key, title = config_tuple
    model_config = config[model_key]

    from utils.topology import create_topology
    import networkx as nx
    from utils.graph_viz import _create_layout, _draw_bipartite_edges

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

    if model_config['model_type'] in ["rbm", "sbm"]:
        _draw_bipartite_edges(ax, G, pos, visible_nodes, hidden_nodes)
    else:
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, width=1.5, ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=visible_nodes, node_color='lightblue',
                          node_size=600, edgecolors='darkblue', linewidths=2, ax=ax)

    if hidden_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_color='lightcoral',
                              node_size=600, edgecolors='darkred', linewidths=2, ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)

    # Title with stats
    model_type_full = {'fvbm': 'FVBM', 'rbm': 'RBM', 'sbm': 'SBM'}
    stats = f"{model_type_full.get(model_config['model_type'], 'BM')}"
    stats += f"\n{len(visible_nodes)}v"
    if hidden_nodes:
        stats += f", {len(hidden_nodes)}h"
    stats += f", {len(edges)}e"

    # Add edge type breakdown for SBM
    if model_config['model_type'] == 'sbm' and len(edges) > 0:
        vv_edges = sum(1 for u, v in edges if u in visible_nodes and v in visible_nodes)
        vh_edges = sum(1 for u, v in edges if (u in visible_nodes and v in hidden_nodes) or
                                                 (u in hidden_nodes and v in visible_nodes))
        hh_edges = sum(1 for u, v in edges if u in hidden_nodes and v in hidden_nodes)
        stats += f"\n({vv_edges}v-v, {vh_edges}v-h, {hh_edges}h-h)"

    ax.set_title(f"{title}\n{stats}", fontsize=12, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('docs/architecture_comparison.png', dpi=150, bbox_inches='tight')
print("Updated architecture comparison saved to: docs/architecture_comparison.png")
print("\nCurved edges now show all connections clearly in Dense RBM and Dense SBM!")
