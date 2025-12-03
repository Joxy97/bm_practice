"""
Boltzmann Machine Tutorial: Using D-Wave's GraphRestrictedBoltzmannMachine

This tutorial demonstrates:
1. Using D-Wave's official BoltzmannMachine implementation
2. Flexible architecture: Fully-Connected or Restricted (with hidden units)
3. Training a model to reverse-engineer true parameters
4. Proper MCMC sampling and PyTorch-based training
5. Comprehensive visualization and comparison

Key features:
- User-configurable number of visible and hidden units
- Choice between Fully-Connected and Restricted architectures
- Automatic parameter generation with fixed random seed for reproducibility
- Integration with PyTorch optimizers and D-Wave samplers
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Literal, Optional
import warnings

# Import D-Wave modules
try:
    from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine as GRBM
    from dwave.samplers import SimulatedAnnealingSampler
    import dimod
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    warnings.warn("D-Wave not available. Install with: pip install dwave-ocean-sdk dwave-pytorch-plugin")


def create_fully_connected_topology(n_visible: int, n_hidden: int = 0):
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


def create_restricted_topology(n_visible: int, n_hidden: int = 0, connectivity: float = 0.5, seed: int = 42):
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
    # Add minimum edges if needed
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


def generate_random_parameters(nodes, edges, seed: int = 42,
                              linear_scale: float = 1.0,
                              quadratic_scale: float = 1.0):
    """
    Generate random linear and quadratic parameters.

    Args:
        nodes: List of node identifiers
        edges: List of edge tuples
        seed: Random seed for reproducibility
        linear_scale: Scale for linear biases
        quadratic_scale: Scale for quadratic biases

    Returns:
        linear: Dictionary {node: bias}
        quadratic: Dictionary {(node1, node2): weight}
    """
    rng = np.random.RandomState(seed)

    # Generate linear biases: uniform in [-linear_scale, linear_scale]
    linear = {node: linear_scale * (2 * rng.random() - 1) for node in nodes}

    # Generate quadratic biases: uniform in [-quadratic_scale, quadratic_scale]
    quadratic = {edge: quadratic_scale * (2 * rng.random() - 1) for edge in edges}

    return linear, quadratic


def visualize_model_parameters(grbm: GRBM, title: str = "Model Parameters"):
    """
    Visualize the biases and weights of a GRBM model.

    Args:
        grbm: GraphRestrictedBoltzmannMachine instance
        title: Title for the plot
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
    axes[0].set_xlabel('Unit')
    axes[0].set_ylabel('Bias Value')
    axes[0].set_title('Linear Biases (Blue=Visible, Orange=Hidden)')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, rotation=45 if len(labels) > 10 else 0)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Plot weights as scatter
    edge_idx_i = grbm.edge_idx_i.cpu().numpy()
    edge_idx_j = grbm.edge_idx_j.cpu().numpy()

    axes[1].scatter(edge_idx_i, edge_idx_j, c=quadratic, cmap='RdBu_r',
                   s=100, vmin=-max(abs(quadratic)), vmax=max(abs(quadratic)))
    axes[1].scatter(edge_idx_j, edge_idx_i, c=quadratic, cmap='RdBu_r',
                   s=100, vmin=-max(abs(quadratic)), vmax=max(abs(quadratic)))

    axes[1].set_xlabel('Node i')
    axes[1].set_ylabel('Node j')
    axes[1].set_title(f'Interaction Weights ({len(quadratic)} edges)')
    axes[1].grid(True, alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdBu_r',
                               norm=plt.Normalize(vmin=-max(abs(quadratic)),
                                                 vmax=max(abs(quadratic))))
    sm.set_array([])
    plt.colorbar(sm, ax=axes[1], label='Weight Value')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def visualize_training_history(history: dict):
    """
    Visualize training progress.

    Args:
        history: Dictionary with keys 'loss', 'grad_norm', 'beta'
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curve
    axes[0].plot(history['loss'], linewidth=2, color='steelblue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Quasi-Objective Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    # Gradient norm
    axes[1].plot(history['grad_norm'], linewidth=2, color='coral')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Average Gradient Magnitude')
    axes[1].set_title('Gradient Norm')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    # Estimated beta
    if 'beta' in history and len(history['beta']) > 0:
        axes[2].plot(history['beta'], linewidth=2, color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Estimated β (Inverse Temperature)')
        axes[2].set_title('Model Temperature')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'Beta estimation\nnot available',
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].axis('off')

    plt.tight_layout()
    return fig


def compare_models(true_grbm: GRBM, learned_grbm: GRBM):
    """
    Side-by-side comparison of true vs learned models.

    Args:
        true_grbm: True model
        learned_grbm: Learned model
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Extract parameters
    true_linear = true_grbm.linear.detach().cpu().numpy()
    learned_linear = learned_grbm.linear.detach().cpu().numpy()
    true_quadratic = true_grbm.quadratic.detach().cpu().numpy()
    learned_quadratic = learned_grbm.quadratic.detach().cpu().numpy()

    n_nodes = len(true_linear)
    n_visible = len(true_grbm.visible_idx)

    # True biases
    axes[0, 0].bar(range(n_nodes), true_linear, color='green', alpha=0.7)
    axes[0, 0].set_title('TRUE Linear Biases')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Learned biases
    axes[0, 1].bar(range(n_nodes), learned_linear, color='blue', alpha=0.7)
    axes[0, 1].set_title('LEARNED Linear Biases')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Bias comparison
    x = np.arange(n_nodes)
    width = 0.35
    axes[0, 2].bar(x - width/2, true_linear, width, label='True', color='green', alpha=0.7)
    axes[0, 2].bar(x + width/2, learned_linear, width, label='Learned', color='blue', alpha=0.7)
    axes[0, 2].set_title('Bias Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    axes[0, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # True weights scatter
    edge_idx_i = true_grbm.edge_idx_i.cpu().numpy()
    edge_idx_j = true_grbm.edge_idx_j.cpu().numpy()
    max_weight = max(abs(true_quadratic).max(), abs(learned_quadratic).max())

    sc1 = axes[1, 0].scatter(edge_idx_i, edge_idx_j, c=true_quadratic, cmap='RdBu_r',
                             s=50, vmin=-max_weight, vmax=max_weight)
    axes[1, 0].set_title('TRUE Interaction Weights')
    axes[1, 0].set_xlabel('Node i')
    axes[1, 0].set_ylabel('Node j')
    plt.colorbar(sc1, ax=axes[1, 0])

    # Learned weights scatter
    sc2 = axes[1, 1].scatter(edge_idx_i, edge_idx_j, c=learned_quadratic, cmap='RdBu_r',
                             s=50, vmin=-max_weight, vmax=max_weight)
    axes[1, 1].set_title('LEARNED Interaction Weights')
    axes[1, 1].set_xlabel('Node i')
    plt.colorbar(sc2, ax=axes[1, 1])

    # Weight difference
    diff = learned_quadratic - true_quadratic
    sc3 = axes[1, 2].scatter(edge_idx_i, edge_idx_j, c=diff, cmap='RdBu_r',
                             s=50, vmin=-max_weight, vmax=max_weight)
    axes[1, 2].set_title('Difference (Learned - True)')
    axes[1, 2].set_xlabel('Node i')
    plt.colorbar(sc3, ax=axes[1, 2])

    plt.tight_layout()

    # Calculate errors
    bias_mae = np.mean(np.abs(true_linear - learned_linear))
    weight_mae = np.mean(np.abs(true_quadratic - learned_quadratic))

    print(f"\nMean Absolute Error:")
    print(f"  Linear Biases:       {bias_mae:.4f}")
    print(f"  Quadratic Weights:   {weight_mae:.4f}")

    return fig, bias_mae, weight_mae


def train_boltzmann_machine(
    grbm: GRBM,
    data: torch.Tensor,
    sampler,
    n_epochs: int = 100,
    learning_rate: float = 0.1,
    model_sample_size: int = 100,
    prefactor: float = 1.0,
    kind: Optional[Literal["sampling", "exact-disc"]] = None,
    sample_params: Optional[dict] = None,
    verbose: bool = True
):
    """
    Train a Boltzmann Machine using maximum likelihood estimation.

    Args:
        grbm: GraphRestrictedBoltzmannMachine instance
        data: Training data, shape (n_samples, n_visible)
        sampler: D-Wave sampler instance
        n_epochs: Number of training epochs
        learning_rate: Learning rate for SGD
        model_sample_size: Number of samples from model per epoch
        prefactor: Prefactor for energy scaling
        kind: How to handle hidden units (None, "sampling", or "exact-disc")
        sample_params: Additional parameters for sampler
        verbose: Print progress

    Returns:
        history: Dictionary with training metrics
    """
    if sample_params is None:
        sample_params = {
            'num_reads': model_sample_size,
            'beta_range': [1.0, 1.0],
            'proposal_acceptance_criteria': 'Gibbs'
        }

    optimizer = torch.optim.SGD(grbm.parameters(), lr=learning_rate)

    history = {
        'loss': [],
        'grad_norm': [],
        'beta': []
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"Training Boltzmann Machine")
        print(f"{'='*70}")
        print(f"Epochs:              {n_epochs}")
        print(f"Learning rate:       {learning_rate}")
        print(f"Model samples/epoch: {model_sample_size}")
        print(f"Training data:       {len(data)} samples")
        print(f"Architecture:        {len(grbm.visible_idx)} visible, {len(grbm.hidden_idx)} hidden")
        print(f"Edges:               {len(grbm.edges)}")
        print(f"{'='*70}\n")

    for epoch in range(n_epochs):
        # Sample from current model
        model_samples = grbm.sample(
            sampler,
            prefactor=prefactor,
            sample_params=sample_params,
            as_tensor=True
        )

        # Estimate temperature
        try:
            beta = grbm.estimate_beta(model_samples)
            history['beta'].append(beta)
        except:
            beta = None

        # Compute quasi-objective (negative log-likelihood approximation)
        optimizer.zero_grad()
        loss = grbm.quasi_objective(
            data,
            model_samples,
            kind=kind,
            prefactor=prefactor if kind is not None else None,
            sampler=sampler if kind == "sampling" else None,
            sample_kwargs=sample_params if kind == "sampling" else None
        )

        # Backpropagate
        loss.backward()

        # Compute gradient norm
        grad_norm = (grbm._linear.grad.abs().mean().item() +
                    grbm._quadratic.grad.abs().mean().item()) / 2

        # Update parameters
        optimizer.step()

        # Record history
        history['loss'].append(loss.item())
        history['grad_norm'].append(grad_norm)

        # Print progress
        if verbose and epoch % 10 == 0:
            beta_str = f", β={beta:.4f}" if beta is not None else ""
            print(f"Epoch {epoch:3d}: Loss={loss.item():8.4f}, Grad={grad_norm:.4f}{beta_str}")

    if verbose:
        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}\n")

    return history


def main():
    """Main tutorial execution."""
    print("=" * 70)
    print("BOLTZMANN MACHINE TUTORIAL: D-Wave Implementation")
    print("=" * 70)

    if not DWAVE_AVAILABLE:
        print("\nERROR: D-Wave not available!")
        print("Please install: pip install dwave-ocean-sdk dwave-pytorch-plugin")
        return

    # ========================================================================
    # HYPERPARAMETERS - ADJUST THESE
    # ========================================================================

    # Architecture
    N_VISIBLE = 4                       # Number of visible units
    N_HIDDEN = 0                        # Number of hidden units (0 for fully visible)
    ARCHITECTURE = "fully-connected"    # "fully-connected" or "restricted"
    CONNECTIVITY = 0.7                  # Only for restricted architecture (0 to 1)

    # Parameter generation
    RANDOM_SEED = 42                # For reproducibility
    LINEAR_BIAS_SCALE = 1.0         # Scale for random linear biases
    QUADRATIC_WEIGHT_SCALE = 1.5    # Scale for random quadratic weights

    # Data generation
    N_TRAINING_SAMPLES = 10000  # Number of training samples
    SAMPLE_NUM_READS = 1000    # Generated samples per D-Wave API call

    # Training parameters
    LEARNING_RATE = 0.005      # Learning rate
    N_EPOCHS = 100             # Number of epochs
    MODEL_SAMPLE_SIZE = 1000   # Samples from model per epoch
    PREFACTOR = 1.0            # Energy scaling factor

    # Hidden unit handling (only relevant if N_HIDDEN > 0)
    HIDDEN_KIND = "exact-disc"  # "exact-disc" or "sampling" or None

    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"Architecture:        {ARCHITECTURE}")
    print(f"Visible units:       {N_VISIBLE}")
    print(f"Hidden units:        {N_HIDDEN}")
    if ARCHITECTURE == "restricted" and N_HIDDEN == 0:
        print(f"Connectivity:        {CONNECTIVITY:.1%}")
    print(f"Random seed:         {RANDOM_SEED}")
    print(f"Training samples:    {N_TRAINING_SAMPLES}")
    print(f"Learning rate:       {LEARNING_RATE}")
    print(f"Epochs:              {N_EPOCHS}")
    print(f"Model samples:       {MODEL_SAMPLE_SIZE}")

    # ========================================================================
    # STEP 1: Create TRUE model topology
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Creating TRUE Model Topology")
    print("=" * 70)

    if ARCHITECTURE == "fully-connected":
        nodes, edges, hidden_nodes = create_fully_connected_topology(N_VISIBLE, N_HIDDEN)
    elif ARCHITECTURE == "restricted":
        nodes, edges, hidden_nodes = create_restricted_topology(
            N_VISIBLE, N_HIDDEN, CONNECTIVITY, RANDOM_SEED
        )
    else:
        raise ValueError(f"Unknown architecture: {ARCHITECTURE}")

    print(f"\nTopology:")
    print(f"  Total nodes:  {len(nodes)}")
    print(f"  Visible:      {N_VISIBLE}")
    print(f"  Hidden:       {N_HIDDEN}")
    print(f"  Total edges:  {len(edges)}")
    print(f"  Density:      {len(edges) / (len(nodes) * (len(nodes) - 1) / 2):.1%}")

    # ========================================================================
    # STEP 2: Initialize TRUE model with random parameters
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Initializing TRUE Model Parameters")
    print("=" * 70)

    true_linear, true_quadratic = generate_random_parameters(
        nodes, edges,
        seed=RANDOM_SEED,
        linear_scale=LINEAR_BIAS_SCALE,
        quadratic_scale=QUADRATIC_WEIGHT_SCALE
    )

    true_grbm = GRBM(
        nodes=nodes,
        edges=edges,
        hidden_nodes=hidden_nodes if N_HIDDEN > 0 else None,
        linear=true_linear,
        quadratic=true_quadratic
    )

    print(f"\nTrue model initialized:")
    print(f"  Linear bias range:    [{true_grbm.linear.min():.3f}, {true_grbm.linear.max():.3f}]")
    print(f"  Quadratic weight range: [{true_grbm.quadratic.min():.3f}, {true_grbm.quadratic.max():.3f}]")

    fig1 = visualize_model_parameters(true_grbm, "TRUE Boltzmann Machine")
    plt.savefig('dwave_true_bm_parameters.png', dpi=150, bbox_inches='tight')
    print("\nSaved: dwave_true_bm_parameters.png")

    # ========================================================================
    # STEP 3: Sample training data from TRUE model
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Sampling Training Data from TRUE Model")
    print("=" * 70)

    sampler = SimulatedAnnealingSampler()

    print(f"\nGenerating {N_TRAINING_SAMPLES} samples using MCMC...")

    training_samples = true_grbm.sample(
        sampler,
        prefactor=PREFACTOR,
        sample_params={
            'num_reads': SAMPLE_NUM_READS,
            'beta_range': [1.0, 1.0],
            'proposal_acceptance_criteria': 'Gibbs'
        },
        as_tensor=True
    )

    # Extract only visible units for training
    training_data = training_samples[:N_TRAINING_SAMPLES, true_grbm.visible_idx]

    print(f"Generated {len(training_data)} training samples")
    print(f"Data shape: {training_data.shape}")
    print(f"Data mean: {training_data.mean():.3f} (should be near 0 for {-1,+1} encoding)")

    # Visualize data statistics
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))

    data_np = training_data.cpu().numpy()

    # Variable activation rates
    activation_rates = (data_np == 1).mean(axis=0)
    axes[0].bar(range(N_VISIBLE), activation_rates, color='steelblue')
    axes[0].set_xlabel('Visible Unit')
    axes[0].set_ylabel('P(unit = +1)')
    axes[0].set_title('Training Data: Activation Rates')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Pairwise correlations
    corr_matrix = np.corrcoef(data_np.T)
    im = axes[1].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_xlabel('Visible Unit')
    axes[1].set_ylabel('Visible Unit')
    axes[1].set_title('Training Data: Correlations')
    plt.colorbar(im, ax=axes[1])

    for i in range(min(N_VISIBLE, 10)):  # Limit annotations for large graphs
        for j in range(min(N_VISIBLE, 10)):
            if N_VISIBLE <= 10:
                text = axes[1].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()
    plt.savefig('dwave_training_data_statistics.png', dpi=150, bbox_inches='tight')
    print("Saved: dwave_training_data_statistics.png")

    # ========================================================================
    # STEP 4: Initialize LEARNED model with random parameters
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Initializing LEARNED Model (Random)")
    print("=" * 70)

    # Use different seed for learned model initialization
    learned_linear, learned_quadratic = generate_random_parameters(
        nodes, edges,
        seed=RANDOM_SEED + 1000,  # Different seed
        linear_scale=0.1,  # Smaller initial scale
        quadratic_scale=0.1
    )

    learned_grbm = GRBM(
        nodes=nodes,
        edges=edges,
        hidden_nodes=hidden_nodes if N_HIDDEN > 0 else None,
        linear=learned_linear,
        quadratic=learned_quadratic
    )

    print(f"\nLearned model initialized with random parameters:")
    print(f"  Linear bias range:    [{learned_grbm.linear.min():.3f}, {learned_grbm.linear.max():.3f}]")
    print(f"  Quadratic weight range: [{learned_grbm.quadratic.min():.3f}, {learned_grbm.quadratic.max():.3f}]")

    # ========================================================================
    # STEP 5: Train the LEARNED model
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Training LEARNED Model")
    print("=" * 70)

    history = train_boltzmann_machine(
        grbm=learned_grbm,
        data=training_data,
        sampler=sampler,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        model_sample_size=MODEL_SAMPLE_SIZE,
        prefactor=PREFACTOR,
        kind=HIDDEN_KIND if N_HIDDEN > 0 else None,
        verbose=True
    )

    print(f"\nFinal learned parameters:")
    print(f"  Linear bias range:    [{learned_grbm.linear.min():.3f}, {learned_grbm.linear.max():.3f}]")
    print(f"  Quadratic weight range: [{learned_grbm.quadratic.min():.3f}, {learned_grbm.quadratic.max():.3f}]")

    # ========================================================================
    # STEP 6: Visualize results
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Visualizing Results")
    print("=" * 70)

    fig3 = visualize_model_parameters(learned_grbm, "LEARNED Boltzmann Machine")
    plt.savefig('dwave_learned_bm_parameters.png', dpi=150, bbox_inches='tight')
    print("\nSaved: dwave_learned_bm_parameters.png")

    fig4 = visualize_training_history(history)
    plt.savefig('dwave_training_history.png', dpi=150, bbox_inches='tight')
    print("Saved: dwave_training_history.png")

    fig5, bias_mae, weight_mae = compare_models(true_grbm, learned_grbm)
    plt.savefig('dwave_true_vs_learned_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: dwave_true_vs_learned_comparison.png")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("TUTORIAL COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. dwave_true_bm_parameters.png")
    print("  2. dwave_training_data_statistics.png")
    print("  3. dwave_learned_bm_parameters.png")
    print("  4. dwave_training_history.png")
    print("  5. dwave_true_vs_learned_comparison.png")

    print(f"\nFinal Metrics:")
    print(f"  Linear Bias MAE:      {bias_mae:.4f}")
    print(f"  Quadratic Weight MAE: {weight_mae:.4f}")
    print(f"  Final Loss:           {history['loss'][-1]:.4f}")
    print(f"  Final Gradient Norm:  {history['grad_norm'][-1]:.6f}")

    print("\n" + "=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
