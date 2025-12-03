"""
Boltzmann Machine Tutorial: Reverse Engineering a Quadratic Model

This tutorial demonstrates:
1. Manually defining a "true" BM with 4 binary variables
2. Sampling training data from the true model using D-Wave with PROPER MCMC sampling
3. Training a BM to reverse-engineer the original parameters
4. Visualizing the learning process

This is a pedagogical example with full transparency and control.

KEY FIX: Now uses proper MCMC sampling parameters (beta_range=[1,1], Gibbs proposal)
to sample from the Boltzmann distribution instead of finding ground states.

HYPERPARAMETERS: See lines 348-370 in main() function for all adjustable parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import torch
import torch.nn as nn

# Try to import D-Wave's PyTorch integration
try:
    from dwave.samplers import SimulatedAnnealingSampler
    import dimod
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    print("WARNING: D-Wave not available. Install with: pip install dwave-ocean-sdk")


class BoltzmannMachine:
    """
    A simple Boltzmann Machine with 4 binary variables.

    Energy function: E(v) = -sum_i(b_i * v_i) - sum_{i<j}(W_ij * v_i * v_j)

    Where:
    - v_i are binary variables {0, 1} or {-1, +1}
    - b_i are biases
    - W_ij are interaction weights
    """

    def __init__(self, n_visible: int = 4, binary_encoding: str = '0-1'):
        """
        Initialize a Boltzmann Machine.

        Args:
            n_visible: Number of visible variables
            binary_encoding: '0-1' for {0,1} or 'spin' for {-1,+1}
        """
        self.n_visible = n_visible
        self.binary_encoding = binary_encoding

        # Initialize parameters
        self.biases = np.zeros(n_visible)
        # Upper triangular matrix for weights (symmetric, no self-connections)
        self.weights = np.zeros((n_visible, n_visible))

    def set_parameters(self, biases: np.ndarray, weights: np.ndarray):
        """Manually set the BM parameters."""
        self.biases = biases.copy()
        self.weights = weights.copy()
        # Ensure weights are symmetric and diagonal is zero
        self.weights = (self.weights + self.weights.T) / 2
        np.fill_diagonal(self.weights, 0)

    def energy(self, state: np.ndarray) -> float:
        """
        Calculate energy of a given state.

        Args:
            state: Binary state vector of shape (n_visible,)

        Returns:
            Energy value
        """
        # Linear term (biases)
        linear_energy = -np.dot(self.biases, state)

        # Quadratic term (interactions)
        quadratic_energy = -0.5 * state @ self.weights @ state

        return linear_energy + quadratic_energy

    def to_qubo(self) -> Dict[Tuple[int, int], float]:
        """
        Convert BM parameters to QUBO (Quadratic Unconstrained Binary Optimization) format.

        QUBO format: minimize sum_i(Q_ii * x_i) + sum_{i<j}(Q_ij * x_i * x_j)

        Our energy (to minimize): E = -sum_i(b_i * x_i) - sum_{i<j}(W_ij * x_i * x_j)
        So Q_ii = -b_i and Q_ij = -W_ij
        """
        Q = {}

        # Linear terms (diagonal)
        for i in range(self.n_visible):
            Q[(i, i)] = -self.biases[i]

        # Quadratic terms (upper triangular)
        for i in range(self.n_visible):
            for j in range(i + 1, self.n_visible):
                if self.weights[i, j] != 0:
                    Q[(i, j)] = -self.weights[i, j]

        return Q

    def sample_dwave(self, num_samples: int = 1000, num_reads: int = 100) -> np.ndarray:
        """
        Sample from the BM using D-Wave's simulated annealing with PROPER MCMC parameters.

        CRITICAL: This uses MCMC sampling mode (thermal sampling), NOT optimization mode!
        The key parameters ensure sampling from the Boltzmann distribution at fixed temperature.

        Args:
            num_samples: Number of samples to generate
            num_reads: Number of reads per sampling run

        Returns:
            Array of samples, shape (num_samples, n_visible)
        """
        if not DWAVE_AVAILABLE:
            raise ImportError("D-Wave not available. Install with: pip install dwave-ocean-sdk")

        Q = self.to_qubo()
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

        sampler = SimulatedAnnealingSampler()

        # CRITICAL FIX: Proper MCMC sampling parameters
        # These make the sampler operate in thermal sampling mode instead of optimization mode
        mcmc_params = {
            'beta_range': [1.0, 1.0],  # Fixed inverse temperature (no annealing!)
            'proposal_acceptance_criteria': 'Gibbs',  # Use Gibbs sampling (proper MCMC)
            'randomize_order': True,  # Randomize variable update order for better mixing
        }

        samples_list = []

        # Sample in batches
        num_batches = (num_samples + num_reads - 1) // num_reads
        for _ in range(num_batches):
            sampleset = sampler.sample(
                bqm,
                num_reads=min(num_reads, num_samples - len(samples_list)),
                **mcmc_params  # Apply MCMC parameters
            )

            for sample, energy, num_occurrences in sampleset.data(['sample', 'energy', 'num_occurrences']):
                # Convert OrderedDict to array
                state = np.array([sample[i] for i in range(self.n_visible)])
                samples_list.append(state)

                if len(samples_list) >= num_samples:
                    break

            if len(samples_list) >= num_samples:
                break

        return np.array(samples_list[:num_samples])

    def visualize_parameters(self, title: str = "BM Parameters"):
        """Visualize the biases and weights."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot biases
        axes[0].bar(range(self.n_visible), self.biases, color='steelblue')
        axes[0].set_xlabel('Variable Index')
        axes[0].set_ylabel('Bias Value')
        axes[0].set_title('Biases')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(range(self.n_visible))

        # Plot weights as heatmap
        im = axes[1].imshow(self.weights, cmap='RdBu_r', vmin=-2, vmax=2)
        axes[1].set_xlabel('Variable j')
        axes[1].set_ylabel('Variable i')
        axes[1].set_title('Interaction Weights')
        axes[1].set_xticks(range(self.n_visible))
        axes[1].set_yticks(range(self.n_visible))

        # Add values to heatmap
        for i in range(self.n_visible):
            for j in range(self.n_visible):
                text = axes[1].text(j, i, f'{self.weights[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=10)

        plt.colorbar(im, ax=axes[1])
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig


class BoltzmannMachineTrainer:
    """
    Train a Boltzmann Machine using maximum likelihood estimation.
    """

    def __init__(self, bm: BoltzmannMachine):
        self.bm = bm
        self.training_history = {
            'biases': [],
            'weights': [],
            'loss': []
        }

    def train(self, data: np.ndarray, learning_rate: float = 0.1,
              num_epochs: int = 100, sample_size: int = 100):
        """
        Train the BM using contrastive divergence.

        For a fully visible BM (no hidden units), we can use simpler maximum likelihood:
        - Data statistics: <v_i>_data, <v_i * v_j>_data
        - Model statistics: <v_i>_model, <v_i * v_j>_model (from sampling)

        Gradient: dE/db_i = <v_i>_data - <v_i>_model
                  dE/dW_ij = <v_i * v_j>_data - <v_i * v_j>_model

        Args:
            data: Training data, shape (n_samples, n_visible)
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            sample_size: Number of samples for model statistics
        """
        print(f"Training BM for {num_epochs} epochs...")
        print(f"Training data: {len(data)} samples")

        # Calculate data statistics (empirical expectations)
        data_mean = data.mean(axis=0)  # <v_i>_data
        data_corr = (data.T @ data) / len(data)  # <v_i * v_j>_data

        print(f"\nData statistics:")
        print(f"Mean: {data_mean}")
        print(f"Correlations:\n{data_corr}")

        for epoch in range(num_epochs):
            # Sample from current model
            try:
                model_samples = self.bm.sample_dwave(num_samples=sample_size)
            except Exception as e:
                print(f"Error sampling from model: {e}")
                print("Using Gibbs sampling fallback...")
                model_samples = self._gibbs_sampling_fallback(sample_size)

            # Calculate model statistics
            model_mean = model_samples.mean(axis=0)
            model_corr = (model_samples.T @ model_samples) / len(model_samples)

            # Compute gradients
            bias_gradient = data_mean - model_mean
            weight_gradient = data_corr - model_corr
            np.fill_diagonal(weight_gradient, 0)  # No self-connections

            # Update parameters
            self.bm.biases += learning_rate * bias_gradient
            self.bm.weights += learning_rate * weight_gradient

            # Make weights symmetric
            self.bm.weights = (self.bm.weights + self.bm.weights.T) / 2
            np.fill_diagonal(self.bm.weights, 0)

            # Calculate loss (KL divergence approximation)
            loss = np.sum((data_mean - model_mean)**2) + \
                   np.sum((data_corr - model_corr)**2)

            # Store history
            self.training_history['biases'].append(self.bm.biases.copy())
            self.training_history['weights'].append(self.bm.weights.copy())
            self.training_history['loss'].append(loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Loss = {loss:.6f}")

        print("\nTraining complete!")

    def _gibbs_sampling_fallback(self, num_samples: int) -> np.ndarray:
        """Simple Gibbs sampling as fallback."""
        samples = []
        state = np.random.randint(0, 2, self.bm.n_visible)

        for _ in range(num_samples * 10):  # Burn-in
            for i in range(self.bm.n_visible):
                # Calculate activation
                activation = self.bm.biases[i] + np.dot(self.bm.weights[i], state)
                prob = 1 / (1 + np.exp(-activation))
                state[i] = 1 if np.random.random() < prob else 0

        for _ in range(num_samples):
            for i in range(self.bm.n_visible):
                activation = self.bm.biases[i] + np.dot(self.bm.weights[i], state)
                prob = 1 / (1 + np.exp(-activation))
                state[i] = 1 if np.random.random() < prob else 0
            samples.append(state.copy())

        return np.array(samples)

    def plot_training_history(self):
        """Visualize training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss curve
        axes[0, 0].plot(self.training_history['loss'], linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # Bias evolution
        biases_history = np.array(self.training_history['biases'])
        for i in range(self.bm.n_visible):
            axes[0, 1].plot(biases_history[:, i], label=f'b_{i}', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Bias Value')
        axes[0, 1].set_title('Bias Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Weight evolution (upper triangular only)
        weights_history = np.array(self.training_history['weights'])
        idx = 0
        for i in range(self.bm.n_visible):
            for j in range(i + 1, self.bm.n_visible):
                axes[1, 0].plot(weights_history[:, i, j],
                              label=f'W_{i}{j}', linewidth=2)
                idx += 1
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Weight Value')
        axes[1, 0].set_title('Weight Evolution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Final parameters comparison (to be filled externally)
        axes[1, 1].text(0.5, 0.5, 'Compare with true model\nusing separate visualization',
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12)
        axes[1, 1].axis('off')

        plt.tight_layout()
        return fig


def main():
    """Main tutorial execution."""
    print("=" * 70)
    print("BOLTZMANN MACHINE TUTORIAL: REVERSE ENGINEERING A QUADRATIC MODEL")
    print("=" * 70)

    # ========================================================================
    # HYPERPARAMETERS - ADJUST THESE FOR TESTING
    # ========================================================================
    # Model architecture
    N_VARIABLES = 4

    # True model parameters (what we're trying to learn)
    TRUE_BIASES = np.array([0.5, -0.3, 0.8, -0.6])
    TRUE_WEIGHTS = np.array([
        [0.0,  1.2, -0.8,  0.4],
        [1.2,  0.0,  0.6, -0.9],
        [-0.8, 0.6,  0.0,  1.1],
        [0.4, -0.9,  1.1,  0.0]
    ])

    # Data generation
    N_TRAINING_SAMPLES = 5000  # Number of training samples (try: 1000, 3000, 5000)
    SAMPLE_NUM_READS = 5000     # Samples per batch

    # Training parameters
    LEARNING_RATE = 0.2       # Learning rate (try: 0.01, 0.05, 0.1)
    N_EPOCHS = 200            # Number of epochs (try: 100, 200, 500)
    MODEL_SAMPLE_SIZE = 5000    # Samples from model per epoch (try: 200, 500, 1000)

    print("\nHyperparameters:")
    print(f"  N_VARIABLES: {N_VARIABLES}")
    print(f"  N_TRAINING_SAMPLES: {N_TRAINING_SAMPLES}")
    print(f"  LEARNING_RATE: {LEARNING_RATE}")
    print(f"  N_EPOCHS: {N_EPOCHS}")
    print(f"  MODEL_SAMPLE_SIZE: {MODEL_SAMPLE_SIZE}")

    # ========================================================================
    # STEP 1: Define the TRUE model
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Defining the TRUE Boltzmann Machine")
    print("=" * 70)

    true_bm = BoltzmannMachine(n_visible=N_VARIABLES, binary_encoding='0-1')

    # Manually set biases and weights for pedagogical clarity
    true_biases = TRUE_BIASES

    true_weights = TRUE_WEIGHTS

    true_bm.set_parameters(true_biases, true_weights)

    print("\nTrue Model Parameters:")
    print(f"Biases: {true_biases}")
    print(f"Weights:\n{true_weights}")

    fig1 = true_bm.visualize_parameters("TRUE Boltzmann Machine")
    plt.savefig('true_bm_parameters.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization: true_bm_parameters.png")

    # ========================================================================
    # STEP 2: Sample training data from the TRUE model
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Sampling Training Data from TRUE Model")
    print("=" * 70)

    if not DWAVE_AVAILABLE:
        print("\nERROR: D-Wave not available!")
        print("Please install: pip install dwave-ocean-sdk")
        return

    print(f"\nGenerating {N_TRAINING_SAMPLES} samples using D-Wave simulated annealing...")

    training_data = true_bm.sample_dwave(num_samples=N_TRAINING_SAMPLES, num_reads=SAMPLE_NUM_READS)

    print(f"\nGenerated {len(training_data)} training samples")
    print(f"Sample shape: {training_data.shape}")
    print(f"\nFirst 10 samples:")
    print(training_data[:10])

    # Visualize data distribution
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Variable frequencies
    var_freq = training_data.mean(axis=0)
    axes[0].bar(range(4), var_freq, color='steelblue')
    axes[0].set_xlabel('Variable')
    axes[0].set_ylabel('P(variable = 1)')
    axes[0].set_title('Training Data: Variable Frequencies')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3)

    # Pairwise correlations
    corr_matrix = np.corrcoef(training_data.T)
    im = axes[1].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_xlabel('Variable')
    axes[1].set_ylabel('Variable')
    axes[1].set_title('Training Data: Correlations')
    plt.colorbar(im, ax=axes[1])

    for i in range(4):
        for j in range(4):
            text = axes[1].text(j, i, f'{corr_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=10)

    plt.tight_layout()
    plt.savefig('training_data_statistics.png', dpi=150, bbox_inches='tight')
    print("Saved visualization: training_data_statistics.png")

    # ========================================================================
    # STEP 3: Train a NEW model to reverse-engineer the TRUE model
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Training a NEW BM to Reverse-Engineer TRUE Model")
    print("=" * 70)

    # Initialize a new BM with random parameters
    learned_bm = BoltzmannMachine(n_visible=N_VARIABLES, binary_encoding='0-1')
    learned_bm.set_parameters(
        biases=np.random.randn(N_VARIABLES) * 0.1,
        weights=np.random.randn(N_VARIABLES, N_VARIABLES) * 0.1
    )

    print("\nInitial (random) parameters:")
    print(f"Biases: {learned_bm.biases}")
    print(f"Weights:\n{learned_bm.weights}")

    # Train the model
    trainer = BoltzmannMachineTrainer(learned_bm)
    trainer.train(
        data=training_data,
        learning_rate=LEARNING_RATE,
        num_epochs=N_EPOCHS,
        sample_size=MODEL_SAMPLE_SIZE
    )

    print("\nLearned parameters:")
    print(f"Biases: {learned_bm.biases}")
    print(f"Weights:\n{learned_bm.weights}")

    # ========================================================================
    # STEP 4: Compare TRUE vs LEARNED models
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Comparing TRUE vs LEARNED Models")
    print("=" * 70)

    fig3 = learned_bm.visualize_parameters("LEARNED Boltzmann Machine")
    plt.savefig('learned_bm_parameters.png', dpi=150, bbox_inches='tight')
    print("Saved visualization: learned_bm_parameters.png")

    # Plot training history
    fig4 = trainer.plot_training_history()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("Saved visualization: training_history.png")

    # Side-by-side comparison
    fig5, axes = plt.subplots(2, 3, figsize=(15, 8))

    # True biases
    axes[0, 0].bar(range(4), true_bm.biases, color='green', alpha=0.7)
    axes[0, 0].set_title('TRUE Biases')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)

    # Learned biases
    axes[0, 1].bar(range(4), learned_bm.biases, color='blue', alpha=0.7)
    axes[0, 1].set_title('LEARNED Biases')
    axes[0, 1].grid(True, alpha=0.3)

    # Bias comparison
    x = np.arange(4)
    width = 0.35
    axes[0, 2].bar(x - width/2, true_bm.biases, width, label='True', color='green', alpha=0.7)
    axes[0, 2].bar(x + width/2, learned_bm.biases, width, label='Learned', color='blue', alpha=0.7)
    axes[0, 2].set_title('Bias Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # True weights
    im1 = axes[1, 0].imshow(true_bm.weights, cmap='RdBu_r', vmin=-2, vmax=2)
    axes[1, 0].set_title('TRUE Weights')
    axes[1, 0].set_ylabel('Variable i')
    axes[1, 0].set_xlabel('Variable j')
    plt.colorbar(im1, ax=axes[1, 0])

    # Learned weights
    im2 = axes[1, 1].imshow(learned_bm.weights, cmap='RdBu_r', vmin=-2, vmax=2)
    axes[1, 1].set_title('LEARNED Weights')
    axes[1, 1].set_xlabel('Variable j')
    plt.colorbar(im2, ax=axes[1, 1])

    # Weight difference
    diff = learned_bm.weights - true_bm.weights
    im3 = axes[1, 2].imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 2].set_title('Difference (Learned - True)')
    axes[1, 2].set_xlabel('Variable j')
    plt.colorbar(im3, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig('true_vs_learned_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization: true_vs_learned_comparison.png")

    # Calculate errors
    bias_error = np.mean(np.abs(true_bm.biases - learned_bm.biases))
    weight_error = np.mean(np.abs(true_bm.weights - learned_bm.weights))

    print(f"\nMean Absolute Error:")
    print(f"  Biases:  {bias_error:.4f}")
    print(f"  Weights: {weight_error:.4f}")

    print("\n" + "=" * 70)
    print("TUTORIAL COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. true_bm_parameters.png - Visualization of true model")
    print("  2. training_data_statistics.png - Training data analysis")
    print("  3. learned_bm_parameters.png - Visualization of learned model")
    print("  4. training_history.png - Training progress")
    print("  5. true_vs_learned_comparison.png - Side-by-side comparison")

    plt.show()


if __name__ == "__main__":
    main()
