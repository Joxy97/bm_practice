"""
Project Configuration Template

Copy this file to your project directory and customize for your use-case.

Usage:
    python -m bm_core.bm --mode train --config projects/my_project/project_config.py --dataset projects/my_project/data/train.csv
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bm_core.config import BMConfig, GRBMConfig, TrainingConfig

# Main configuration
config = BMConfig(
    seed=42,

    # GRBM Architecture
    grbm=GRBMConfig(
        n_visible=10,          # TODO: Set based on your data
        n_hidden=0,            # 0 for FVBM, >0 for RBM/SBM
        model_type="fvbm",     # "fvbm", "rbm", or "sbm"
        sparsity=None,         # None for dense, 0.0-1.0 for sparse
        init_linear_scale=0.1,
        init_quadratic_scale=0.1
    ),

    # Training Configuration
    training=TrainingConfig(
        batch_size=5000,
        n_epochs=100,
        mode="pcd",            # "cd" or "pcd"
        sampler_name="gibbs_gpu",  # "gibbs", "gibbs_gpu", "metropolis", etc.
    ),

    # Paths (relative to this file's location)
    paths={
        'data_dir': 'data',
        'model_dir': 'outputs/models',
        'log_dir': 'outputs/logs',
        'plot_dir': 'outputs/plots'
    }
)

# Validate configuration
config.validate()
