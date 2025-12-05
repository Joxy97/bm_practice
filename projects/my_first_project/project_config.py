"""
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
