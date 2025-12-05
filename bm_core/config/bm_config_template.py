"""
Python configuration template for BM Core pipeline.

This module provides type-safe configuration using Python dataclasses instead of YAML.
Users copy this template to their project and customize for their use-case.

Benefits over YAML:
- Type safety and validation
- IDE autocomplete and IntelliSense
- Easier programmatic manipulation
- Clear structure with documentation
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any


@dataclass
class GRBMConfig:
    """
    GRBM architecture configuration.

    For v1, only simple sparsity is supported. In v2, full graph constructor
    will be available.
    """
    n_visible: int = 10
    """Number of visible units"""

    n_hidden: int = 0
    """Number of hidden units (0 for FVBM)"""

    sparsity: Optional[float] = None
    """
    Sparsity parameter for v1 (0.0-1.0).
    If None, creates dense topology.
    If specified, creates sparse topology with this density.
    """

    model_type: Literal["fvbm", "rbm", "sbm"] = "fvbm"
    """
    Model type:
    - "fvbm": Fully Visible BM (n_hidden must be 0)
    - "rbm": Restricted BM (bipartite, n_hidden > 0)
    - "sbm": Standard/General BM (all connections, n_hidden > 0)
    """

    init_linear_scale: float = 0.1
    """Initial scale for linear biases"""

    init_quadratic_scale: float = 0.1
    """Initial scale for quadratic weights"""


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    optimizer: Literal["adam", "sgd"] = "adam"
    """Optimizer type"""

    learning_rate: float = 0.01
    """Learning rate"""

    betas: List[float] = field(default_factory=lambda: [0.85, 0.999])
    """Adam beta parameters [beta1, beta2]"""

    eps: float = 1.0e-7
    """Adam epsilon for numerical stability"""

    weight_decay: float = 0.0
    """Weight decay (use regularization instead)"""


@dataclass
class GradientClippingConfig:
    """Gradient clipping configuration to prevent divergence."""

    enabled: bool = True
    """Enable gradient clipping"""

    method: Literal["norm", "value"] = "norm"
    """Clipping method: 'norm' (clip by norm) or 'value' (clip by value)"""

    max_norm: float = 1.0
    """Maximum gradient norm (for method='norm')"""

    max_value: float = 0.5
    """Maximum gradient value (for method='value')"""


@dataclass
class RegularizationConfig:
    """Regularization configuration to prevent overfitting."""

    linear_l2: float = 0.001
    """L2 penalty on linear biases"""

    quadratic_l2: float = 0.01
    """L2 penalty on quadratic weights"""

    quadratic_l1: float = 0.0
    """L1 penalty for sparsity (optional)"""


@dataclass
class LRSchedulerConfig:
    """Learning rate scheduler configuration."""

    enabled: bool = True
    """Enable learning rate scheduling"""

    type: Literal["plateau", "step", "cosine", "exponential"] = "plateau"
    """Scheduler type"""

    # Plateau-specific
    factor: float = 0.5
    """Reduce LR by this factor (plateau)"""

    patience: int = 15
    """Epochs without improvement before reducing (plateau)"""

    min_lr: float = 1.0e-5
    """Minimum learning rate"""

    monitor: str = "val_loss"
    """Metric to monitor (plateau)"""

    # Step-specific
    step_size: int = 100
    """Reduce LR every N epochs (step)"""

    gamma: float = 0.5
    """Multiplicative factor (step/exponential)"""

    # Cosine-specific
    T_max: int = 50
    """Maximum iterations (cosine)"""

    eta_min: float = 1.0e-5
    """Minimum learning rate (cosine)"""


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""

    enabled: bool = False
    """Enable early stopping"""

    patience: int = 20
    """Number of epochs without improvement before stopping"""

    min_delta: float = 0.0001
    """Minimum change to qualify as improvement"""

    metric: str = "val_loss"
    """Metric to monitor: 'val_loss', 'train_loss', 'grad_norm'"""

    mode: Literal["min", "max"] = "min"
    """'min' for loss, 'max' for accuracy-like metrics"""

    restore_best_weights: bool = True
    """Restore best model weights after early stopping"""


@dataclass
class PCDConfig:
    """Persistent Contrastive Divergence configuration."""

    num_chains: int = 100
    """Number of persistent chains"""

    k_steps: int = 10
    """MCMC steps per parameter update"""

    initialize_from: Literal["random", "data"] = "random"
    """Initialize chains from 'random' or 'data'"""


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Basic training parameters
    batch_size: int = 5000
    """Batch size for training"""

    n_epochs: int = 100
    """Number of training epochs"""

    # Training mode
    mode: Literal["cd", "pcd"] = "pcd"
    """Training mode: 'cd' (Contrastive Divergence) or 'pcd' (Persistent CD)"""

    cd_k: int = 1
    """Number of Gibbs steps for CD-k (when mode='cd')"""

    pcd: PCDConfig = field(default_factory=PCDConfig)
    """PCD configuration (when mode='pcd')"""

    # Sampler configuration
    sampler_name: str = "gibbs"
    """
    Sampler to use for training.
    Common options: 'gibbs', 'gibbs_gpu', 'metropolis', 'metropolis_gpu'
    """

    sampler_params: Dict[str, Any] = field(default_factory=lambda: {
        'num_sweeps': 1000,
        'burn_in': 100,
        'thinning': 1,
        'randomize_order': True
    })
    """Sampler-specific parameters"""

    model_sample_size: int = 100
    """Number of samples to draw from model during training"""

    prefactor: float = 1.0
    """Temperature scaling factor (inverse temperature / beta)"""

    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """Optimizer configuration"""

    # Gradient clipping
    gradient_clipping: GradientClippingConfig = field(default_factory=GradientClippingConfig)
    """Gradient clipping configuration"""

    # Regularization
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    """Regularization configuration"""

    # Learning rate scheduler
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    """Learning rate scheduler configuration"""

    # Early stopping
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    """Early stopping configuration"""

    # Hidden unit handling
    hidden_kind: Optional[Literal["exact-disc", "sampling"]] = None
    """Hidden unit treatment (None for FVBM, 'exact-disc' or 'sampling' for RBM/SBM)"""

    # Checkpointing
    save_best_model: bool = True
    """Save best model during training"""

    checkpoint_dir: str = "outputs/checkpoints"
    """Directory to save checkpoints"""


@dataclass
class DataConfig:
    """Data configuration."""

    train_ratio: float = 0.7
    """Training set ratio"""

    val_ratio: float = 0.15
    """Validation set ratio"""

    test_ratio: float = 0.15
    """Test set ratio"""


@dataclass
class LoggingConfig:
    """Logging and visualization configuration."""

    log_interval: int = 1
    """Log every N epochs"""

    save_plots: bool = True
    """Save training plots"""

    plot_dir: str = "outputs/plots"
    """Directory to save plots"""

    track_metrics: List[str] = field(default_factory=lambda: [
        "loss", "grad_norm", "beta", "val_loss"
    ])
    """Metrics to track during training"""


@dataclass
class BMConfig:
    """
    Main BM pipeline configuration.

    This is the top-level configuration object that users customize
    for their project.

    Example usage:
        from bm_core.config import BMConfig, GRBMConfig, TrainingConfig

        config = BMConfig(
            seed=42,
            grbm=GRBMConfig(
                n_visible=784,  # MNIST
                n_hidden=100,
                sparsity=0.1
            ),
            training=TrainingConfig(
                batch_size=128,
                n_epochs=50,
                mode='pcd'
            )
        )
    """

    seed: int = 42
    """Random seed for reproducibility"""

    # Model configuration
    grbm: GRBMConfig = field(default_factory=GRBMConfig)
    """GRBM architecture configuration"""

    # Training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)
    """Training configuration"""

    # Data configuration
    data: DataConfig = field(default_factory=DataConfig)
    """Data split configuration"""

    # Logging configuration
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    """Logging and visualization configuration"""

    # Device configuration
    device: Dict[str, Any] = field(default_factory=lambda: {
        'use_cuda': 'auto'  # 'auto', 'cuda', 'cpu', or specific device like 'cuda:0'
    })
    """Device configuration (CPU/CUDA)"""

    # Paths configuration
    paths: Dict[str, str] = field(default_factory=lambda: {
        'data_dir': 'data',
        'model_dir': 'outputs/models',
        'log_dir': 'outputs/logs',
        'plot_dir': 'outputs/plots'
    })
    """File paths configuration"""

    def validate(self) -> None:
        """
        Validate configuration consistency.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate model_type and n_hidden consistency
        if self.grbm.model_type == "fvbm" and self.grbm.n_hidden != 0:
            raise ValueError(
                f"FVBM requires n_hidden=0, got n_hidden={self.grbm.n_hidden}"
            )

        if self.grbm.model_type in ["rbm", "sbm"] and self.grbm.n_hidden == 0:
            raise ValueError(
                f"{self.grbm.model_type.upper()} requires n_hidden>0, "
                f"got n_hidden={self.grbm.n_hidden}"
            )

        # Validate sparsity
        if self.grbm.sparsity is not None:
            if not (0.0 <= self.grbm.sparsity <= 1.0):
                raise ValueError(
                    f"Sparsity must be in [0, 1], got {self.grbm.sparsity}"
                )

        # Validate data splits
        total_ratio = self.data.train_ratio + self.data.val_ratio + self.data.test_ratio
        if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
            raise ValueError(
                f"Data ratios must sum to 1.0, got {total_ratio}"
            )

        # Validate hidden_kind consistency
        if self.grbm.n_hidden == 0 and self.training.hidden_kind is not None:
            raise ValueError(
                "hidden_kind must be None for FVBM (n_hidden=0)"
            )

        if self.grbm.n_hidden > 0 and self.training.hidden_kind is None:
            print(
                "Warning: n_hidden > 0 but hidden_kind is None. "
                "Consider setting hidden_kind='exact-disc' or 'sampling'"
            )


# Example configuration instances for common use-cases

def create_fvbm_config(n_visible: int = 10) -> BMConfig:
    """
    Create configuration for Fully Visible BM.

    Args:
        n_visible: Number of visible units

    Returns:
        BMConfig instance for FVBM
    """
    return BMConfig(
        grbm=GRBMConfig(
            n_visible=n_visible,
            n_hidden=0,
            model_type="fvbm",
            sparsity=None  # Dense
        ),
        training=TrainingConfig(
            mode="cd",
            cd_k=1,
            sampler_name="gibbs"
        )
    )


def create_rbm_config(n_visible: int = 784, n_hidden: int = 100) -> BMConfig:
    """
    Create configuration for Restricted BM.

    Args:
        n_visible: Number of visible units
        n_hidden: Number of hidden units

    Returns:
        BMConfig instance for RBM
    """
    return BMConfig(
        grbm=GRBMConfig(
            n_visible=n_visible,
            n_hidden=n_hidden,
            model_type="rbm",
            sparsity=None  # Dense
        ),
        training=TrainingConfig(
            mode="pcd",
            pcd=PCDConfig(num_chains=100, k_steps=10),
            sampler_name="gibbs",
            hidden_kind="exact-disc"
        )
    )


def create_sparse_fvbm_config(
    n_visible: int = 1000,
    sparsity: float = 0.1
) -> BMConfig:
    """
    Create configuration for sparse Fully Visible BM.

    Args:
        n_visible: Number of visible units
        sparsity: Connection density (0.0-1.0)

    Returns:
        BMConfig instance for sparse FVBM
    """
    return BMConfig(
        grbm=GRBMConfig(
            n_visible=n_visible,
            n_hidden=0,
            model_type="fvbm",
            sparsity=sparsity
        ),
        training=TrainingConfig(
            mode="pcd",
            pcd=PCDConfig(num_chains=200, k_steps=10),
            sampler_name="gibbs_gpu"  # Use GPU for large sparse models
        )
    )
