"""
Data generator for sampling from a true Boltzmann Machine.
"""

import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Dict, Any

from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine as GRBM
from dwave.samplers import SimulatedAnnealingSampler

from utils.topology import create_topology
from utils.parameters import generate_random_parameters


class DataGenerator:
    """
    Generate training data by sampling from a true Boltzmann Machine.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data generator.

        Args:
            config: Configuration dictionary containing true_model, data, and seed
        """
        self.config = config
        self.seed = config['seed']
        self.true_model_config = config['true_model']
        self.data_config = config['data']

        # Set random seeds
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create topology
        self.nodes, self.edges, self.hidden_nodes = self._create_topology()

        # Initialize true model
        self.true_model = self._initialize_true_model()

        # Sampler
        self.sampler = SimulatedAnnealingSampler()

    def _create_topology(self):
        """Create the graph topology based on config."""
        n_visible = self.true_model_config['n_visible']
        n_hidden = self.true_model_config['n_hidden']
        model_type = self.true_model_config['model_type']
        connectivity = self.true_model_config['connectivity']
        connectivity_density = self.true_model_config.get('connectivity_density', 0.5)

        nodes, edges, hidden_nodes = create_topology(
            n_visible=n_visible,
            n_hidden=n_hidden,
            model_type=model_type,
            connectivity=connectivity,
            connectivity_density=connectivity_density,
            seed=self.seed
        )

        print(f"Topology created:")
        print(f"  Model Type: {model_type.upper()}")
        print(f"  Connectivity: {connectivity}")
        if connectivity == "sparse":
            print(f"  Connectivity Density: {connectivity_density:.1%}")
        print(f"  Nodes: {len(nodes)} ({n_visible} visible, {n_hidden} hidden)")
        print(f"  Edges: {len(edges)}")

        return nodes, edges, hidden_nodes

    def _initialize_true_model(self) -> GRBM:
        """Initialize the true Boltzmann Machine with random parameters."""
        linear, quadratic = generate_random_parameters(
            self.nodes,
            self.edges,
            seed=self.seed,
            linear_scale=self.true_model_config['linear_bias_scale'],
            quadratic_scale=self.true_model_config['quadratic_weight_scale']
        )

        grbm = GRBM(
            nodes=self.nodes,
            edges=self.edges,
            hidden_nodes=self.hidden_nodes if self.true_model_config['n_hidden'] > 0 else None,
            linear=linear,
            quadratic=quadratic
        )

        print(f"\nTrue model initialized:")
        print(f"  Linear bias range: [{grbm.linear.min():.3f}, {grbm.linear.max():.3f}]")
        print(f"  Quadratic weight range: [{grbm.quadratic.min():.3f}, {grbm.quadratic.max():.3f}]")

        return grbm

    def generate(self, save_dir: str = "data") -> pd.DataFrame:
        """
        Generate samples from the true model and save to CSV.

        Args:
            save_dir: Directory to save the dataset

        Returns:
            DataFrame containing the generated samples
        """
        print(f"\n{'='*70}")
        print("GENERATING DATA FROM TRUE MODEL")
        print(f"{'='*70}")

        n_samples = self.data_config['n_samples']
        num_reads = self.data_config['num_reads']
        prefactor = self.data_config['prefactor']
        sampler_params = self.data_config['sampler_params']

        print(f"\nSampling {n_samples} samples using MCMC...")

        # Sample from true model
        samples = self.true_model.sample(
            self.sampler,
            prefactor=prefactor,
            sample_params={
                'num_reads': num_reads,
                **sampler_params
            },
            as_tensor=True
        )

        # Extract visible units only
        visible_samples = samples[:n_samples, self.true_model.visible_idx]

        # Convert to numpy
        data_np = visible_samples.cpu().numpy()

        # Create DataFrame
        n_visible = self.true_model_config['n_visible']
        columns = [f'v{i}' for i in range(n_visible)]
        df = pd.DataFrame(data_np, columns=columns)

        # Add metadata
        df['sample_id'] = range(len(df))

        print(f"\nGenerated {len(df)} samples")
        print(f"Shape: {df.shape}")
        print(f"\nFirst 5 samples:")
        print(df.head())
        print(f"\nData statistics:")
        print(df[columns].describe())

        # Save to CSV
        os.makedirs(save_dir, exist_ok=True)
        dataset_name = self.data_config['dataset_name']
        filepath = os.path.join(save_dir, f"{dataset_name}.csv")
        df.to_csv(filepath, index=False)

        print(f"\nDataset saved to: {filepath}")

        return df

    def get_true_model(self) -> GRBM:
        """Return the true model for comparison."""
        return self.true_model

    def get_topology(self):
        """Return the topology (nodes, edges, hidden_nodes)."""
        return self.nodes, self.edges, self.hidden_nodes
