"""
Synthetic Data Generator for Boltzmann Machines.

Generates synthetic data by sampling from a true BM model with known parameters.
Useful for testing, validation, and benchmarking.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bm_core.models import BoltzmannMachine
from bm_core.utils import create_topology, generate_random_parameters


class SyntheticDataGenerator:
    """
    Generate synthetic training data by sampling from a true BM.

    This creates a "ground truth" BM with known parameters and samples
    from it to generate training data. Useful for:
    - Testing training algorithms
    - Benchmarking samplers
    - Validating implementations
    - Debugging
    """

    def __init__(self, config: Dict[str, Any], sampler_dict: Dict[str, Any]):
        """
        Initialize data generator.

        Args:
            config: Generator configuration from YAML
            sampler_dict: Dictionary of available samplers from factory
        """
        self.config = config
        self.sampler_dict = sampler_dict
        self.seed = config.get('seed', 42)

        # Set random seeds
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create true model
        self.true_model = self._create_true_model()

        print(f"\nSynthetic Data Generator initialized")
        print(f"  Model: {self.config['true_model']['model_type'].upper()}")
        print(f"  Visible units: {self.config['true_model']['n_visible']}")
        print(f"  Hidden units: {self.config['true_model']['n_hidden']}")

    def _create_true_model(self) -> BoltzmannMachine:
        """Create true model for data generation."""
        true_model_config = self.config['true_model']

        # Create topology
        nodes, edges, hidden_nodes = create_topology(
            n_visible=true_model_config['n_visible'],
            n_hidden=true_model_config['n_hidden'],
            model_type=true_model_config['model_type'],
            connectivity=true_model_config.get('connectivity', 'dense'),
            connectivity_density=true_model_config.get('connectivity_density', 0.5),
            seed=self.seed
        )

        # Generate random parameters
        linear, quadratic = generate_random_parameters(
            nodes,
            edges,
            seed=self.seed,
            linear_scale=true_model_config['linear_bias_scale'],
            quadratic_scale=true_model_config['quadratic_weight_scale']
        )

        # Create BoltzmannMachine
        true_model = BoltzmannMachine(
            nodes=nodes,
            edges=edges,
            hidden_nodes=hidden_nodes if true_model_config['n_hidden'] > 0 else None,
            linear=linear,
            quadratic=quadratic,
            sampler_dict=self.sampler_dict
        )

        print(f"\nTrue model parameters:")
        linear_tensor, quadratic_tensor = true_model.get_parameters()
        print(f"  Linear bias range: [{linear_tensor.min():.3f}, {linear_tensor.max():.3f}]")
        print(f"  Quadratic weight range: [{quadratic_tensor.min():.3f}, {quadratic_tensor.max():.3f}]")

        return true_model

    def generate(self, save_dir: str) -> pd.DataFrame:
        """
        Generate synthetic data and save to CSV.

        Args:
            save_dir: Directory to save the dataset

        Returns:
            DataFrame containing generated samples
        """
        data_config = self.config['data']

        print(f"\n{'='*70}")
        print("GENERATING SYNTHETIC DATA")
        print(f"{'='*70}")

        n_samples = data_config['n_samples']
        sampler_type = data_config['sampler_type']
        sampler_params = data_config.get('sampler_params', {})

        print(f"\nConfiguration:")
        print(f"  Samples: {n_samples}")
        print(f"  Sampler: {sampler_type}")

        # Sample from true model
        print(f"\nSampling from true model...")
        samples = self.true_model.sample(
            sampler_name=sampler_type,
            prefactor=data_config.get('prefactor', 1.0),
            sample_params={
                'num_reads': n_samples,
                **sampler_params
            },
            as_tensor=True
        )

        # Extract visible units only
        visible_samples = samples[:n_samples, self.true_model.visible_idx]

        # Convert to numpy
        data_np = visible_samples.cpu().numpy()

        # Create DataFrame
        n_visible = self.config['true_model']['n_visible']
        columns = [f'v{i}' for i in range(n_visible)]
        df = pd.DataFrame(data_np, columns=columns)

        # Add metadata
        df['sample_id'] = range(len(df))

        print(f"\nGenerated {len(df)} samples")
        print(f"Shape: {df.shape}")
        print(f"\nFirst 5 samples:")
        print(df.head())

        # Data statistics
        print(f"\nData statistics:")
        print(df[columns].describe())

        # Save to CSV
        os.makedirs(save_dir, exist_ok=True)
        dataset_name = data_config['dataset_name']
        filepath = os.path.join(save_dir, f"{dataset_name}.csv")
        df.to_csv(filepath, index=False)

        print(f"\n✓ Dataset saved to: {filepath}")
        print(f"{'='*70}")

        return df

    def get_true_model(self) -> BoltzmannMachine:
        """Return the true model for comparison."""
        return self.true_model

    def get_topology(self):
        """Return the topology (nodes, edges, hidden_nodes)."""
        return (
            self.true_model.nodes,
            self.true_model.edges,
            self.true_model.hidden_nodes
        )
