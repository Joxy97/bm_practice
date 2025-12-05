"""
Visualization utilities for Boltzmann Machines.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import os

from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine as GRBM


def plot_model_parameters(
    grbm: GRBM,
    title: str = "Model Parameters",
    save_path: Optional[str] = None
):
    """
    Visualize the biases and weights of a GRBM model.

    Args:
        grbm: GraphRestrictedBoltzmannMachine instance
        title: Title for the plot
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract parameters
    linear = grbm.linear.detach().cpu().numpy()
    quadratic = grbm.quadratic.detach().cpu().numpy()

    n_visible = len(grbm.visible_idx)
    n_hidden = len(grbm.hidden_idx)

    # Plot biases
    visible_biases = linear[grbm.visible_idx.cpu().numpy()]
    colors = ['steelblue'] * n_visible
    labels = [f'v{i}' for i in range(n_visible)]

    if n_hidden > 0:
        hidden_biases = linear[grbm.hidden_idx.cpu().numpy()]
        colors += ['coral'] * n_hidden
        labels += [f'h{i}' for i in range(n_hidden)]
        all_biases = np.concatenate([visible_biases, hidden_biases])
    else:
        all_biases = visible_biases

    x_pos = np.arange(len(all_biases))
    axes[0].bar(x_pos, all_biases, color=colors, alpha=0.7)
    axes[0].set_xlabel('Unit', fontsize=11)
    axes[0].set_ylabel('Bias Value', fontsize=11)
    axes[0].set_title('Linear Biases', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, rotation=45 if len(labels) > 10 else 0)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    if n_hidden > 0:
        axes[0].legend(['Visible', 'Hidden'], loc='upper right')

    # Plot weights as heatmap
    edge_idx_i = grbm.edge_idx_i.cpu().numpy()
    edge_idx_j = grbm.edge_idx_j.cpu().numpy()
    n_nodes = len(grbm.nodes)

    # Create weight matrix
    weight_matrix = np.zeros((n_nodes, n_nodes))
    for idx, (i, j) in enumerate(zip(edge_idx_i, edge_idx_j)):
        weight_matrix[i, j] = quadratic[idx]
        weight_matrix[j, i] = quadratic[idx]  # Symmetric

    max_weight = np.abs(quadratic).max() if len(quadratic) > 0 else 1.0

    im = axes[1].imshow(weight_matrix, cmap='RdBu_r', vmin=-max_weight, vmax=max_weight,
                        aspect='auto', interpolation='nearest')

    axes[1].set_xlabel('Node j', fontsize=11)
    axes[1].set_ylabel('Node i', fontsize=11)
    axes[1].set_title(f'Quadratic Weights ({len(quadratic)} edges)', fontsize=12, fontweight='bold')

    # Set ticks
    axes[1].set_xticks(range(n_nodes))
    axes[1].set_yticks(range(n_nodes))

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('Weight Value', fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[str] = None
):
    """
    Plot training history.

    Args:
        history: Dictionary with training metrics
        save_path: Path to save the figure (optional)
    """
    has_val = 'val_loss' in history and len(history['val_loss']) > 0
    has_beta = 'beta' in history and len(history['beta']) > 0

    n_plots = 2 + int(has_beta)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    if n_plots == 1:
        axes = [axes]

    # Loss curve
    ax_idx = 0
    axes[ax_idx].plot(history['train_loss'], linewidth=2, color='steelblue', label='Train')
    if has_val:
        axes[ax_idx].plot(history['val_loss'], linewidth=2, color='coral', label='Val', linestyle='--')
        axes[ax_idx].legend()
    axes[ax_idx].set_xlabel('Epoch', fontsize=11)
    axes[ax_idx].set_ylabel('Loss', fontsize=11)
    axes[ax_idx].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[ax_idx].grid(True, alpha=0.3)

    # Gradient norm
    ax_idx += 1
    axes[ax_idx].plot(history['grad_norm'], linewidth=2, color='green')
    axes[ax_idx].set_xlabel('Epoch', fontsize=11)
    axes[ax_idx].set_ylabel('Gradient Norm', fontsize=11)
    axes[ax_idx].set_title('Average Gradient Magnitude', fontsize=12, fontweight='bold')
    axes[ax_idx].grid(True, alpha=0.3)
    axes[ax_idx].set_yscale('log')

    # Beta (if available)
    if has_beta:
        ax_idx += 1
        axes[ax_idx].plot(history['beta'], linewidth=2, color='purple')
        axes[ax_idx].set_xlabel('Epoch', fontsize=11)
        axes[ax_idx].set_ylabel('β (Inverse Temperature)', fontsize=11)
        axes[ax_idx].set_title('Estimated Temperature', fontsize=12, fontweight='bold')
        axes[ax_idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_model_comparison(
    true_grbm: GRBM,
    learned_grbm: GRBM,
    save_path: Optional[str] = None
):
    """
    Compare true vs learned models side-by-side.

    Args:
        true_grbm: True model
        learned_grbm: Learned model
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Extract parameters
    true_linear = true_grbm.linear.detach().cpu().numpy()
    learned_linear = learned_grbm.linear.detach().cpu().numpy()
    true_quadratic = true_grbm.quadratic.detach().cpu().numpy()
    learned_quadratic = learned_grbm.quadratic.detach().cpu().numpy()

    n_nodes = len(true_linear)

    # True biases
    axes[0, 0].bar(range(n_nodes), true_linear, color='green', alpha=0.7)
    axes[0, 0].set_title('TRUE Linear Biases', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Value', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Learned biases
    axes[0, 1].bar(range(n_nodes), learned_linear, color='blue', alpha=0.7)
    axes[0, 1].set_title('LEARNED Linear Biases', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Bias comparison
    x = np.arange(n_nodes)
    width = 0.35
    axes[0, 2].bar(x - width/2, true_linear, width, label='True', color='green', alpha=0.7)
    axes[0, 2].bar(x + width/2, learned_linear, width, label='Learned', color='blue', alpha=0.7)
    axes[0, 2].set_title('Bias Comparison', fontsize=12, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    axes[0, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Create weight matrices
    edge_idx_i = true_grbm.edge_idx_i.cpu().numpy()
    edge_idx_j = true_grbm.edge_idx_j.cpu().numpy()
    n_nodes = len(true_grbm.nodes)
    max_weight = max(np.abs(true_quadratic).max(), np.abs(learned_quadratic).max())

    # True weight matrix
    true_weight_matrix = np.zeros((n_nodes, n_nodes))
    for idx, (i, j) in enumerate(zip(edge_idx_i, edge_idx_j)):
        true_weight_matrix[i, j] = true_quadratic[idx]
        true_weight_matrix[j, i] = true_quadratic[idx]

    im1 = axes[1, 0].imshow(true_weight_matrix, cmap='RdBu_r', vmin=-max_weight, vmax=max_weight,
                            aspect='auto', interpolation='nearest')
    axes[1, 0].set_title('TRUE Quadratic Weights', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Node j', fontsize=11)
    axes[1, 0].set_ylabel('Node i', fontsize=11)
    axes[1, 0].set_xticks(range(n_nodes))
    axes[1, 0].set_yticks(range(n_nodes))
    plt.colorbar(im1, ax=axes[1, 0])

    # Learned weight matrix
    learned_weight_matrix = np.zeros((n_nodes, n_nodes))
    for idx, (i, j) in enumerate(zip(edge_idx_i, edge_idx_j)):
        learned_weight_matrix[i, j] = learned_quadratic[idx]
        learned_weight_matrix[j, i] = learned_quadratic[idx]

    im2 = axes[1, 1].imshow(learned_weight_matrix, cmap='RdBu_r', vmin=-max_weight, vmax=max_weight,
                            aspect='auto', interpolation='nearest')
    axes[1, 1].set_title('LEARNED Quadratic Weights', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Node j', fontsize=11)
    axes[1, 1].set_xticks(range(n_nodes))
    axes[1, 1].set_yticks(range(n_nodes))
    plt.colorbar(im2, ax=axes[1, 1])

    # Weight difference matrix
    diff_matrix = learned_weight_matrix - true_weight_matrix

    im3 = axes[1, 2].imshow(diff_matrix, cmap='RdBu_r', vmin=-max_weight, vmax=max_weight,
                            aspect='auto', interpolation='nearest')
    axes[1, 2].set_title('Difference (Learned - True)', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Node j', fontsize=11)
    axes[1, 2].set_xticks(range(n_nodes))
    axes[1, 2].set_yticks(range(n_nodes))
    plt.colorbar(im3, ax=axes[1, 2])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    # Calculate errors
    bias_mae = np.mean(np.abs(true_linear - learned_linear))
    weight_mae = np.mean(np.abs(true_quadratic - learned_quadratic))

    print(f"\nModel Comparison Metrics:")
    print(f"  Linear Bias MAE:      {bias_mae:.4f}")
    print(f"  Quadratic Weight MAE: {weight_mae:.4f}")

    return fig, bias_mae, weight_mae
