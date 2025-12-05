# API Reference

## Core Modules

### bm_core.models.BoltzmannMachine

```python
class BoltzmannMachine:
    """
    Boltzmann Machine abstraction layer over D-Wave GRBM.

    This class provides a clean API for interacting with Boltzmann Machines,
    decoupling the core pipeline from specific implementations.
    """

    def __init__(
        self,
        nodes: List[int],
        edges: List[Tuple[int, int]],
        hidden_nodes: Optional[List[int]] = None,
        linear: Optional[Dict[int, float]] = None,
        quadratic: Optional[Dict[Tuple[int, int], float]] = None,
        sampler_dict: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Boltzmann Machine.

        Args:
            nodes: List of all node identifiers
            edges: List of edge tuples (u, v)
            hidden_nodes: List of hidden node identifiers (None for FVBM)
            linear: Dictionary of linear biases {node: bias}
            quadratic: Dictionary of quadratic weights {(u,v): weight}
            sampler_dict: Dictionary of available samplers {name: sampler}
        """

    def sample(
        self,
        sampler_name: str,
        prefactor: float = 1.0,
        sample_params: Optional[Dict[str, Any]] = None,
        as_tensor: bool = True
    ) -> torch.Tensor:
        """
        Sample from the Boltzmann Machine.

        Args:
            sampler_name: Name of sampler from sampler_dict
            prefactor: Temperature scaling factor (inverse temperature)
            sample_params: Sampler-specific parameters
            as_tensor: Return PyTorch tensor (True) or numpy array (False)

        Returns:
            Samples tensor of shape (num_samples, n_visible + n_hidden)
        """

    def quasi_objective(
        self,
        data_samples: torch.Tensor,
        model_samples: torch.Tensor,
        kind: Optional[str] = None,
        prefactor: float = 1.0,
        sampler_name: Optional[str] = None
    ) -> torch.Tensor:
        """
        Compute quasi-objective (loss) for training.

        Args:
            data_samples: Samples from data distribution
            model_samples: Samples from model distribution
            kind: Hidden unit treatment: "exact-disc", "sampling", or None
            prefactor: Temperature scaling factor
            sampler_name: Sampler to use if kind="sampling"

        Returns:
            Loss tensor (scalar)
        """

    def get_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get model parameters (linear, quadratic)."""

    def set_parameters(self, linear: torch.Tensor, quadratic: torch.Tensor):
        """Set model parameters."""

    def to(self, device: torch.device) -> 'BoltzmannMachine':
        """Move model to specified device."""

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int = 0,
        optimizer_state: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """Save model checkpoint."""

    @classmethod
    def load_checkpoint(
        cls,
        filepath: str,
        sampler_dict: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ) -> 'BoltzmannMachine':
        """Load model from checkpoint."""
```

### bm_core.models.BMDataset

```python
class BMDataset(Dataset):
    """
    Base PyTorch Dataset for Boltzmann Machine samples.

    Users extend this class and override load_data() for custom loading.
    """

    def __init__(self, csv_path: str, **kwargs):
        """
        Initialize dataset.

        Args:
            csv_path: Path to CSV file
            **kwargs: Additional arguments for custom implementations
        """

    def load_data(self, csv_path: str) -> np.ndarray:
        """
        Load data from CSV file (override this method).

        Args:
            csv_path: Path to CSV file

        Returns:
            Numpy array of shape (n_samples, n_visible) with float32 dtype
        """

    def __len__(self) -> int:
        """Return number of samples."""

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample."""

    def get_n_visible(self) -> int:
        """Return number of visible units."""
```

### bm_core.trainers.BoltzmannMachineTrainer

```python
class BoltzmannMachineTrainer:
    """
    Trainer for Boltzmann Machines with CD/PCD support.
    """

    def __init__(
        self,
        model: BoltzmannMachine,
        config: Dict[str, Any],
        device: torch.device,
        sampler_name: str = 'gibbs'
    ):
        """
        Initialize trainer.

        Args:
            model: BoltzmannMachine instance
            config: Training configuration
            device: PyTorch device
            sampler_name: Name of sampler to use
        """

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        n_epochs: int = 100
    ):
        """
        Train model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            n_epochs: Number of epochs
        """

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch and return metrics."""

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model and return metrics."""

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """Test model and return metrics."""

    def save_checkpoint(self, filepath: str, epoch: int):
        """Save training checkpoint."""

    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
```

### bm_core.config

```python
@dataclass
class GRBMConfig:
    """GRBM architecture configuration."""
    n_visible: int = 10
    n_hidden: int = 0
    model_type: Literal["fvbm", "rbm", "sbm"] = "fvbm"
    sparsity: Optional[float] = None
    init_linear_scale: float = 0.1
    init_quadratic_scale: float = 0.1

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 5000
    n_epochs: int = 100
    mode: Literal["cd", "pcd"] = "pcd"
    cd_k: int = 1
    pcd: PCDConfig = field(default_factory=PCDConfig)
    sampler_name: str = "gibbs"
    sampler_params: Dict[str, Any] = field(default_factory=dict)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    gradient_clipping: GradientClippingConfig = field(default_factory=GradientClippingConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

@dataclass
class BMConfig:
    """Main BM pipeline configuration."""
    seed: int = 42
    grbm: GRBMConfig = field(default_factory=GRBMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    device: Dict[str, Any] = field(default_factory=lambda: {'use_cuda': 'auto'})
    paths: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration consistency."""
```

### bm_core.utils

```python
def create_topology(
    n_visible: int,
    n_hidden: int,
    model_type: Literal["fvbm", "rbm", "sbm"],
    connectivity: Literal["dense", "sparse"] = "dense",
    connectivity_density: float = 0.5,
    seed: int = 42
) -> Tuple[List[int], List[Tuple[int, int]], List[int]]:
    """
    Create Boltzmann Machine topology.

    Args:
        n_visible: Number of visible units
        n_hidden: Number of hidden units
        model_type: Type of BM ("fvbm", "rbm", "sbm")
        connectivity: "dense" or "sparse"
        connectivity_density: Fraction of edges for sparse (0.0-1.0)
        seed: Random seed

    Returns:
        Tuple of (nodes, edges, hidden_nodes)
    """

def generate_random_parameters(
    nodes: List,
    edges: List[Tuple],
    seed: int = 42,
    linear_scale: float = 1.0,
    quadratic_scale: float = 1.0
) -> Tuple[Dict, Dict]:
    """
    Generate random parameters.

    Args:
        nodes: List of node identifiers
        edges: List of edge tuples
        seed: Random seed
        linear_scale: Scale for linear biases
        quadratic_scale: Scale for quadratic weights

    Returns:
        Tuple of (linear_dict, quadratic_dict)
    """

def get_device(device_config: Dict) -> torch.device:
    """
    Get PyTorch device based on configuration.

    Args:
        device_config: Device configuration dict
            {'use_cuda': 'auto'|'cuda'|'cpu'|'cuda:0'}

    Returns:
        PyTorch device
    """

def create_dataloaders(
    dataset_path: str,
    dataset_class: type = BMDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders from CSV.

    Args:
        dataset_path: Path to CSV file
        dataset_class: Dataset class to use
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        batch_size: Batch size
        shuffle: Shuffle training data
        seed: Random seed
        **dataset_kwargs: Additional arguments for dataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
```

## Plugin Modules

### plugins.sampler_factory.SamplerFactory

```python
class SamplerFactory:
    """
    Sampler/Solver factory for Boltzmann Machines.

    Provides centralized sampler management and registration.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize factory.

        Args:
            config_path: Path to sampler_factory_config.yaml (optional)
        """

    def get_sampler_dict(self) -> Dict[str, Sampler]:
        """
        Get dictionary of all registered samplers.

        Returns:
            Dictionary mapping sampler names to sampler instances
        """

    def get_sampler(self, name: str) -> Sampler:
        """
        Get specific sampler by name.

        Args:
            name: Sampler name

        Returns:
            Sampler instance

        Raises:
            ValueError: If sampler not found
        """

    def list_samplers(self) -> List[str]:
        """
        List all available sampler names.

        Returns:
            List of sampler names
        """

    def summary(self) -> str:
        """Get detailed summary of available samplers."""
```

## Project Management

### projects.project_manager.ProjectManager

```python
class ProjectManager:
    """Manages BM project lifecycle."""

    def __init__(self, projects_dir: str = "projects"):
        """
        Initialize project manager.

        Args:
            projects_dir: Directory containing projects
        """

    def create_project(self, project_name: str, template: str = "template"):
        """
        Create new project from template.

        Args:
            project_name: Name for the new project
            template: Template to use
        """

    def list_projects(self):
        """List all projects."""
```

## CLI Commands

### bm_core.bm

Main entry point for training and testing:

```bash
python -m bm_core.bm --mode <build|train|test> --config <config.py> [options]

Options:
  --mode {build,train,test}  Operation mode (required)
  --config PATH              Path to project_config.py (required)
  --dataset PATH             Path to dataset CSV file
  --checkpoint PATH          Path to model checkpoint (for test mode)
```

### projects.project_manager

Project management CLI:

```bash
python -m projects.project_manager <action> [options]

Actions:
  create                     Create new project
  list                       List all projects

Options (for create):
  --name NAME                Project name (required)
  --template TEMPLATE        Template to use (default: template)
```

## Usage Examples

### Basic Training

```python
from bm_core.models import BoltzmannMachine, create_dataloaders
from bm_core.trainers import BoltzmannMachineTrainer
from bm_core.config import BMConfig, GRBMConfig, TrainingConfig
from bm_core.utils import create_topology, generate_random_parameters, get_device
from plugins.sampler_factory import SamplerFactory
import torch

# Configuration
config = BMConfig(
    grbm=GRBMConfig(n_visible=10, n_hidden=0, model_type="fvbm"),
    training=TrainingConfig(batch_size=5000, n_epochs=100)
)

# Setup
device = get_device(config.device)
factory = SamplerFactory()
sampler_dict = factory.get_sampler_dict()

# Build model
nodes, edges, hidden_nodes = create_topology(
    n_visible=config.grbm.n_visible,
    n_hidden=config.grbm.n_hidden,
    model_type=config.grbm.model_type
)
linear, quadratic = generate_random_parameters(nodes, edges)

model = BoltzmannMachine(
    nodes=nodes,
    edges=edges,
    hidden_nodes=hidden_nodes,
    linear=linear,
    quadratic=quadratic,
    sampler_dict=sampler_dict
).to(device)

# Load data
train_loader, val_loader, test_loader = create_dataloaders(
    'data/train.csv',
    batch_size=config.training.batch_size
)

# Train
trainer = BoltzmannMachineTrainer(model, config, device, sampler_name='gibbs')
trainer.train(train_loader, val_loader, n_epochs=config.training.n_epochs)
```

### Custom Dataset

```python
from bm_core.models import BMDataset
import pandas as pd
import numpy as np

class MyDataset(BMDataset):
    def load_data(self, csv_path: str) -> np.ndarray:
        df = pd.read_csv(csv_path)
        # Custom preprocessing
        data = df[['feature1', 'feature2', 'feature3']].values
        data = (data - data.mean()) / data.std()
        return data.astype(np.float32)

# Use custom dataset
from bm_core.models import create_dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    'data/my_data.csv',
    dataset_class=MyDataset,
    batch_size=128
)
```

### Sampling

```python
# Sample from trained model
samples = model.sample(
    sampler_name='gibbs',
    prefactor=1.0,
    sample_params={
        'num_reads': 1000,
        'num_sweeps': 1000,
        'burn_in': 100
    }
)

print(f"Generated {samples.shape[0]} samples")
print(f"Shape: {samples.shape}")  # (1000, n_visible)
```

### Checkpoint Management

```python
# Save checkpoint
model.save_checkpoint(
    'checkpoints/model_epoch_50.pt',
    epoch=50,
    optimizer_state=trainer.optimizer.state_dict(),
    metadata={'val_loss': 1.234}
)

# Load checkpoint
loaded_model = BoltzmannMachine.load_checkpoint(
    'checkpoints/model_epoch_50.pt',
    sampler_dict=sampler_dict,
    device=device
)
```

## Type Hints

The codebase uses comprehensive type hints for better IDE support:

```python
from typing import List, Tuple, Dict, Optional, Literal, Any
import torch
import numpy as np

def my_function(
    data: np.ndarray,
    config: BMConfig,
    device: torch.device,
    mode: Literal["train", "test"]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ...
```

## Error Handling

Common exceptions and how to handle them:

```python
try:
    config.validate()
except ValueError as e:
    print(f"Configuration error: {e}")

try:
    sampler = factory.get_sampler('unknown_sampler')
except ValueError as e:
    print(f"Sampler not found: {e}")
    print(f"Available: {factory.list_samplers()}")

try:
    samples = model.sample('gibbs', ...)
except RuntimeError as e:
    print(f"Sampling error: {e}")
```

## Performance Tips

### GPU Acceleration

```python
# Use GPU samplers for large models
config = BMConfig(
    training=TrainingConfig(sampler_name='gibbs_gpu')
)

# Check CUDA availability
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
```

### Memory Management

```python
# Use sparse connectivity for large models
config = BMConfig(
    grbm=GRBMConfig(
        n_visible=10000,
        sparsity=0.01  # Only 1% connections
    )
)

# Adjust batch size based on available memory
config = BMConfig(
    training=TrainingConfig(
        batch_size=1000  # Reduce if OOM
    )
)
```

### Sampling Efficiency

```python
# Reduce burn-in for faster sampling (but less accurate)
samples = model.sample(
    'gibbs',
    sample_params={
        'num_sweeps': 100,  # Fewer sweeps
        'burn_in': 10       # Shorter burn-in
    }
)
```
