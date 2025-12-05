"""
Project Manager - CLI tool for creating and managing BM projects.

Usage:
    python project_manager.py create --name my_project
    python project_manager.py list
"""

import os
import argparse
from pathlib import Path


class ProjectManager:
    """Manages BM project lifecycle."""

    def __init__(self, projects_dir: str = "projects"):
        self.projects_dir = Path(projects_dir)

    def create_project(self, project_name: str):
        """
        Create new project with all necessary files.

        Args:
            project_name: Name for the new project
        """
        project_path = self.projects_dir / project_name

        if project_path.exists():
            print(f"Error: Project '{project_name}' already exists at {project_path}")
            return

        # Create project directory structure
        print(f"Creating project '{project_name}'...")
        project_path.mkdir(parents=True, exist_ok=True)
        (project_path / "data").mkdir(exist_ok=True)
        (project_path / "outputs").mkdir(exist_ok=True)
        (project_path / "outputs" / "checkpoints").mkdir(exist_ok=True)
        (project_path / "outputs" / "plots").mkdir(exist_ok=True)

        # Create project_config.py
        self._create_config_file(project_path)

        # Create dataset.py
        self._create_dataset_file(project_path)

        # Create __init__.py
        (project_path / "__init__.py").write_text("")

        print(f"\nProject '{project_name}' created successfully!")
        print(f"\nProject location: {project_path}")
        print(f"\nNext steps:")
        print(f"  1. Edit configuration:")
        print(f"     {project_path}/project_config.py")
        print(f"  2. Implement custom dataset:")
        print(f"     {project_path}/dataset.py")
        print(f"  3. Place your CSV data files in:")
        print(f"     {project_path}/data/")
        print(f"  4. Run training:")
        print(f"     python -m bm_core.bm --mode train --config {project_path}/project_config.py --dataset {project_path}/data/train.csv")

    def _create_config_file(self, project_path: Path):
        """Generate project_config.py with all configuration options."""
        config_content = '''"""
Project Configuration

Customize all settings for your BM training pipeline.
All configuration options from bm_config_template.py are exposed here.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from bm_core.config.bm_config_template import (
    BMConfig,
    GRBMConfig,
    TrainingConfig,
    OptimizerConfig,
    GradientClippingConfig,
    RegularizationConfig,
    LRSchedulerConfig,
    EarlyStoppingConfig,
    PCDConfig,
    DataConfig,
    LoggingConfig
)


# =============================================================================
# GRBM Architecture Configuration
# =============================================================================

grbm_config = GRBMConfig(
    n_visible=10,           # Number of visible units (adjust to your data dimension)
    n_hidden=0,             # Number of hidden units (0 for FVBM, >0 for RBM/SBM)
    sparsity=None,          # Sparsity: None for dense, 0.0-1.0 for sparse topology
    model_type="fvbm",      # Model type: "fvbm", "rbm", or "sbm"
    init_linear_scale=0.1,  # Initial scale for linear biases
    init_quadratic_scale=0.1  # Initial scale for quadratic weights
)


# =============================================================================
# Optimizer Configuration
# =============================================================================

optimizer_config = OptimizerConfig(
    optimizer="adam",       # Optimizer: "adam" or "sgd"
    learning_rate=0.01,     # Learning rate
    betas=[0.85, 0.999],    # Adam beta parameters [beta1, beta2]
    eps=1.0e-7,             # Adam epsilon for numerical stability
    weight_decay=0.0        # Weight decay (prefer regularization instead)
)


# =============================================================================
# Gradient Clipping Configuration
# =============================================================================

gradient_clipping_config = GradientClippingConfig(
    enabled=True,           # Enable gradient clipping
    method="norm",          # Clipping method: "norm" or "value"
    max_norm=1.0,           # Maximum gradient norm (for method="norm")
    max_value=0.5           # Maximum gradient value (for method="value")
)


# =============================================================================
# Regularization Configuration
# =============================================================================

regularization_config = RegularizationConfig(
    linear_l2=0.001,        # L2 penalty on linear biases
    quadratic_l2=0.01,      # L2 penalty on quadratic weights
    quadratic_l1=0.0        # L1 penalty for sparsity (optional)
)


# =============================================================================
# Learning Rate Scheduler Configuration
# =============================================================================

lr_scheduler_config = LRSchedulerConfig(
    enabled=True,           # Enable learning rate scheduling
    type="plateau",         # Scheduler type: "plateau", "step", "cosine", "exponential"

    # Plateau-specific parameters
    factor=0.5,             # Reduce LR by this factor
    patience=15,            # Epochs without improvement before reducing
    min_lr=1.0e-5,          # Minimum learning rate
    monitor="val_loss",     # Metric to monitor

    # Step-specific parameters
    step_size=100,          # Reduce LR every N epochs
    gamma=0.5,              # Multiplicative factor

    # Cosine-specific parameters
    T_max=50,               # Maximum iterations
    eta_min=1.0e-5          # Minimum learning rate
)


# =============================================================================
# Early Stopping Configuration
# =============================================================================

early_stopping_config = EarlyStoppingConfig(
    enabled=False,          # Enable early stopping
    patience=20,            # Epochs without improvement before stopping
    min_delta=0.0001,       # Minimum change to qualify as improvement
    metric="val_loss",      # Metric to monitor: "val_loss", "train_loss", "grad_norm"
    mode="min",             # "min" for loss, "max" for accuracy-like metrics
    restore_best_weights=True  # Restore best model weights after stopping
)


# =============================================================================
# PCD (Persistent Contrastive Divergence) Configuration
# =============================================================================

pcd_config = PCDConfig(
    num_chains=100,         # Number of persistent chains
    k_steps=10,             # MCMC steps per parameter update
    initialize_from="random"  # Initialize chains from "random" or "data"
)


# =============================================================================
# Sampler Configuration
# =============================================================================
# Configure which sampler to use and its parameters.
# Available samplers (from SamplerFactory):
#
# Classical MCMC Samplers:
#   - "gibbs": Gibbs sampling (single-spin flip)
#   - "metropolis": Metropolis-Hastings sampler
#   - "parallel_tempering": Parallel tempering (replica exchange)
#   - "simulated_annealing": Simulated annealing
#
# GPU Accelerated Samplers:
#   - "gibbs_gpu": GPU-accelerated Gibbs sampler (multiple chains in parallel)
#   - "metropolis_gpu": GPU-accelerated Metropolis sampler
#   - "parallel_tempering_gpu": GPU-accelerated parallel tempering
#   - "simulated_annealing_gpu": GPU-accelerated simulated annealing
#   - "population_annealing_gpu": GPU population annealing
#
# Exact Samplers (small systems only):
#   - "exact": Exact sampling via enumeration (max ~20 variables)
#   - "gumbel_max": Gumbel-max trick sampling (max ~20 variables)
#
# Optimization/Local Search:
#   - "steepest_descent": Steepest descent local search
#   - "tabu": Tabu search
#   - "greedy": Greedy descent
#
# Baseline:
#   - "random": Random sampling (baseline)

sampler_name = "gibbs"  # Choose sampler from list above

# Sampler-specific parameters
# Different samplers accept different parameters. Adjust based on your chosen sampler.

# Common parameters for MCMC samplers (gibbs, metropolis, parallel_tempering)
sampler_params_gibbs = {
    'num_sweeps': 1000,      # Number of MCMC sweeps
    'burn_in': 100,          # Burn-in sweeps before collecting samples
    'thinning': 1,           # Take every Nth sample (1 = no thinning)
    'randomize_order': True  # Randomize spin update order (Gibbs only)
}

# Metropolis sampler parameters
sampler_params_metropolis = {
    'temperature': 1.0,      # Sampling temperature
    'num_sweeps': 1000,
    'burn_in': 100,
    'thinning': 1
}

# Parallel Tempering parameters
sampler_params_parallel_tempering = {
    'num_replicas': 8,       # Number of temperature replicas
    'T_min': 1.0,            # Minimum temperature
    'T_max': 4.0,            # Maximum temperature
    'swap_interval': 10,     # Swap attempt interval
    'num_sweeps': 1000,
    'burn_in': 100,
    'thinning': 1
}

# GPU sampler parameters (adds num_chains for parallel execution)
sampler_params_gibbs_gpu = {
    'num_sweeps': 1000,
    'burn_in': 100,
    'thinning': 1,
    'randomize_order': True,
    'num_chains': 32,        # Number of parallel chains on GPU
    'use_cuda': True         # Use CUDA if available
}

sampler_params_metropolis_gpu = {
    'temperature': 1.0,
    'num_sweeps': 1000,
    'burn_in': 100,
    'thinning': 1,
    'num_chains': 32,
    'use_cuda': True
}

sampler_params_parallel_tempering_gpu = {
    'num_replicas': 8,
    'T_min': 1.0,
    'T_max': 4.0,
    'swap_interval': 10,
    'num_sweeps': 1000,
    'burn_in': 100,
    'thinning': 1,
    'use_cuda': True
}

sampler_params_simulated_annealing_gpu = {
    'beta_range': (1.0, 10.0),  # Inverse temperature range (start, end)
    'proposal_acceptance_criteria': "Metropolis",
    'num_sweeps': 1000,
    'num_chains': 32,
    'use_cuda': True
}

sampler_params_population_annealing_gpu = {
    'population_size': 1000,  # Number of replicas in population
    'num_sweeps': 100,        # Sweeps between resampling
    'beta_min': 0.1,          # Minimum inverse temperature
    'beta_max': 10.0,         # Maximum inverse temperature
    'resample_threshold': 0.5,  # Resampling threshold
    'use_cuda': True
}

# Select parameters based on your chosen sampler
# Uncomment the one matching your sampler_name above
sampler_params = sampler_params_gibbs
# sampler_params = sampler_params_metropolis
# sampler_params = sampler_params_parallel_tempering
# sampler_params = sampler_params_gibbs_gpu
# sampler_params = sampler_params_metropolis_gpu
# sampler_params = sampler_params_parallel_tempering_gpu
# sampler_params = sampler_params_simulated_annealing_gpu
# sampler_params = sampler_params_population_annealing_gpu


# =============================================================================
# Training Configuration
# =============================================================================

training_config = TrainingConfig(
    # Basic training parameters
    batch_size=5000,        # Batch size for training
    n_epochs=100,           # Number of training epochs

    # Training mode
    mode="pcd",             # Training mode: "cd" or "pcd"
    cd_k=1,                 # Number of Gibbs steps for CD-k (when mode="cd")
    pcd=pcd_config,         # PCD configuration (when mode="pcd")

    # Sampler configuration (use variables defined above)
    sampler_name=sampler_name,      # Sampler to use (see list above)
    sampler_params=sampler_params,  # Sampler parameters (see configurations above)

    model_sample_size=100,  # Number of samples to draw from model during training
    prefactor=1.0,          # Temperature scaling factor (inverse temperature / beta)

    # Sub-configurations
    optimizer=optimizer_config,
    gradient_clipping=gradient_clipping_config,
    regularization=regularization_config,
    lr_scheduler=lr_scheduler_config,
    early_stopping=early_stopping_config,

    # Hidden unit handling
    hidden_kind=None,       # None for FVBM, "exact-disc" or "sampling" for RBM/SBM

    # Checkpointing
    save_best_model=True,   # Save best model during training
    checkpoint_dir="outputs/checkpoints"  # Directory to save checkpoints
)


# =============================================================================
# Data Configuration
# =============================================================================

data_config = DataConfig(
    train_ratio=0.7,        # Training set ratio
    val_ratio=0.15,         # Validation set ratio
    test_ratio=0.15         # Test set ratio
)


# =============================================================================
# Logging Configuration
# =============================================================================

logging_config = LoggingConfig(
    log_interval=1,         # Log every N epochs
    save_plots=True,        # Save training plots
    plot_dir="outputs/plots",  # Directory to save plots
    track_metrics=[         # Metrics to track during training
        "loss",
        "grad_norm",
        "beta",
        "val_loss"
    ]
)


# =============================================================================
# Main Configuration
# =============================================================================

config = BMConfig(
    seed=42,                # Random seed for reproducibility

    grbm=grbm_config,
    training=training_config,
    data=data_config,
    logging=logging_config,

    # Device configuration
    device={
        'use_cuda': 'auto'  # 'auto', 'cuda', 'cpu', or specific device like 'cuda:0'
    },

    # Paths configuration
    paths={
        'data_dir': 'data',
        'model_dir': 'outputs/models',
        'log_dir': 'outputs/logs',
        'plot_dir': 'outputs/plots'
    }
)


# Validate configuration on import
config.validate()
'''
        (project_path / "project_config.py").write_text(config_content)

    def _create_dataset_file(self, project_path: Path):
        """Generate dataset.py template."""
        dataset_content = '''"""
Custom Dataset Implementation

This module defines a custom PyTorch Dataset class for your project.
Extend BMDataset and override load_data() to implement custom preprocessing
and data loading logic for your CSV files.

The BM training pipeline uses this dataset with PyTorch DataLoader for
batching and efficient data loading.

Dependencies:
    - bm_core.models.dataset.BMDataset: Base dataset class with PyTorch integration
    - This provides __len__, __getitem__, and DataLoader compatibility
    - Training pipeline uses create_dataloaders() which handles train/val/test splits
"""

import pandas as pd
import numpy as np
from bm_core.models.dataset import BMDataset


class CustomDataset(BMDataset):
    """
    Custom dataset for this project.

    Extends BMDataset to provide custom data loading and preprocessing.
    The base class (BMDataset) is a PyTorch Dataset that:
    - Implements __len__ and __getitem__ for DataLoader compatibility
    - Returns torch.Tensor samples for batch processing
    - Handles train/val/test splitting via create_dataloaders()

    Usage in training:
        from bm_core.models.dataset import create_dataloaders
        from projects.your_project.dataset import CustomDataset

        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_path="data/your_data.csv",
            dataset_class=CustomDataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=128
        )
    """

    def load_data(self, csv_path: str) -> np.ndarray:
        """
        Load and preprocess data from CSV file.

        This method is called by BMDataset.__init__ to load your data.
        Override this to implement custom preprocessing logic.

        Args:
            csv_path: Path to CSV file

        Returns:
            Numpy array of shape (n_samples, n_visible) with dtype float32
            - Each row is one sample
            - Each column is one visible unit (binary/continuous value)

        Example CSV formats:
            1. Visible units as columns (default):
               v0,v1,v2,v3,...
               1,0,1,0,...
               0,1,0,1,...

            2. Custom column names:
               feature_1,feature_2,...,label
               0.5,0.8,...,1
               0.3,0.6,...,0

            3. Multiple data sources (combine multiple CSVs):
               You can load and concatenate data from multiple files here
        """
        # Load CSV
        df = pd.read_csv(csv_path)

        # =================================================================
        # CUSTOMIZE THIS SECTION FOR YOUR DATA FORMAT
        # =================================================================

        # Option 1: Default - Extract columns starting with 'v' (visible units)
        # This matches the default BM data format: v0, v1, v2, ...
        visible_cols = [col for col in df.columns if col.startswith('v')]

        if not visible_cols:
            # Option 2: Use all numeric columns (if no 'v' columns found)
            visible_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            # Option 3: Specify exact column names for your data
            # visible_cols = ['feature_1', 'feature_2', 'feature_3', ...]

            # Option 4: Exclude certain columns (e.g., labels, metadata)
            # visible_cols = [col for col in df.columns if col not in ['label', 'id', 'timestamp']]

            if not visible_cols:
                raise ValueError(
                    f"No valid data columns found in {csv_path}. "
                    f"Available columns: {df.columns.tolist()}"
                )

        # Extract data
        data = df[visible_cols].values

        # =================================================================
        # PREPROCESSING (CUSTOMIZE AS NEEDED)
        # =================================================================

        # Example 1: Normalize to [0, 1] range
        # data_min = data.min(axis=0)
        # data_max = data.max(axis=0)
        # data_range = data_max - data_min
        # data_range[data_range == 0] = 1.0  # Avoid division by zero
        # data = (data - data_min) / data_range

        # Example 2: Standardize (zero mean, unit variance)
        # data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

        # Example 3: Binarize (threshold at 0.5)
        # data = (data > 0.5).astype(np.float32)

        # Example 4: Clip to valid range
        # data = np.clip(data, -1, 1)

        # Example 5: Handle missing values
        # data = np.nan_to_num(data, nan=0.0)

        # =================================================================
        # RETURN AS FLOAT32 (REQUIRED)
        # =================================================================

        # Convert to float32 for PyTorch compatibility
        data = data.astype(np.float32)

        print(f"Loaded dataset from {csv_path}")
        print(f"  Shape: {data.shape}")
        print(f"  Columns used: {visible_cols}")
        print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")

        return data


# =============================================================================
# Helper Functions (Optional)
# =============================================================================

def load_multiple_csvs(csv_paths: list[str]) -> np.ndarray:
    """
    Load and concatenate data from multiple CSV files.

    Useful if your dataset is split across multiple files.

    Args:
        csv_paths: List of paths to CSV files

    Returns:
        Combined numpy array with shape (total_samples, n_visible)

    Example:
        data = load_multiple_csvs([
            'data/batch1.csv',
            'data/batch2.csv',
            'data/batch3.csv'
        ])
    """
    datasets = []
    for path in csv_paths:
        df = pd.read_csv(path)
        # Apply consistent column selection
        visible_cols = [col for col in df.columns if col.startswith('v')]
        datasets.append(df[visible_cols].values)

    combined = np.vstack(datasets).astype(np.float32)
    return combined


def apply_feature_engineering(data: np.ndarray) -> np.ndarray:
    """
    Apply feature engineering transformations.

    Args:
        data: Raw data array

    Returns:
        Transformed data array

    Example transformations:
        - Polynomial features
        - Interaction terms
        - Log transforms
        - PCA dimensionality reduction
    """
    # Example: Add squared features
    # squared_features = data ** 2
    # data = np.hstack([data, squared_features])

    # Example: Add interaction terms (product of pairs)
    # interactions = data[:, :-1] * data[:, 1:]
    # data = np.hstack([data, interactions])

    return data
'''
        (project_path / "dataset.py").write_text(dataset_content)

    def list_projects(self):
        """List all projects."""
        projects = [
            d.name for d in self.projects_dir.iterdir()
            if d.is_dir() and d.name not in ['template', '__pycache__']
        ]

        if not projects:
            print("No projects found.")
            print(f"\nCreate a project with:")
            print(f"  python project_manager.py create --name my_project")
        else:
            print(f"\nAvailable projects ({len(projects)}):")
            for p in sorted(projects):
                print(f"  - {p}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BM Project Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'action',
        choices=['create', 'list'],
        help='Action to perform'
    )

    parser.add_argument(
        '--name',
        type=str,
        help='Project name (for create action)'
    )

    args = parser.parse_args()

    manager = ProjectManager()

    if args.action == 'create':
        if not args.name:
            print("Error: --name is required for create action")
            return
        manager.create_project(args.name)

    elif args.action == 'list':
        manager.list_projects()


if __name__ == '__main__':
    main()
