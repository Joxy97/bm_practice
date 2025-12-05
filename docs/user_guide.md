# User Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Creating a Project](#creating-a-project)
4. [Configuration](#configuration)
5. [Custom Dataset Implementation](#custom-dataset-implementation)
6. [Training](#training)
7. [Testing](#testing)
8. [Advanced Topics](#advanced-topics)
9. [Troubleshooting](#troubleshooting)

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- D-Wave Ocean SDK
- NumPy, pandas

### Install Dependencies

```bash
cd bm_practice
pip install -r requirements.txt
```

### Verify Installation

```python
import torch
from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine
from plugins.sampler_factory import SamplerFactory

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")

# Check samplers
factory = SamplerFactory()
print(f"Available samplers: {len(factory.list_samplers())}")
```

## Quick Start

### 1. Create Your First Project

```bash
python -m project_manager create --name my_first_bm
```

This creates:
```
projects/my_first_bm/
├── project_config.py      # Configuration
├── custom_dataset.py      # Data loading (implement this)
├── data/                  # Your CSV files go here
└── outputs/               # Training outputs
```

### 2. Prepare Your Data

Create `projects/my_first_bm/data/train.csv`:

```csv
v0,v1,v2,v3,v4,v5,v6,v7,v8,v9
-1,1,-1,1,1,-1,-1,1,1,-1
1,1,-1,-1,1,1,-1,-1,1,1
-1,-1,1,1,-1,-1,1,1,-1,-1
...
```

**Data format:**
- Columns named `v0`, `v1`, ..., `vN` for visible units
- Binary values: `-1` or `+1` (spin format)
- Each row is one sample

### 3. Configure (Optional - defaults work for toy example)

Edit `projects/my_first_bm/project_config.py` if needed:

```python
from bm_core.config import BMConfig, GRBMConfig, TrainingConfig

config = BMConfig(
    seed=42,
    grbm=GRBMConfig(
        n_visible=10,  # Match your data dimensions
        n_hidden=0,    # 0 for FVBM
        model_type="fvbm"
    ),
    training=TrainingConfig(
        batch_size=5000,
        n_epochs=100,
        sampler_name="gibbs"
    )
)
```

### 4. Train

```bash
python -m bm_core.bm --mode train \
  --config projects/my_first_bm/project_config.py \
  --dataset projects/my_first_bm/data/train.csv
```

### 5. View Results

Check `projects/my_first_bm/outputs/`:
- `checkpoints/best_model.pt` - Trained model
- `plots/` - Training curves
- `logs/` - Training logs

## Creating a Project

### Using Project Manager

```bash
# Create from default template
python project_manager.py create --name mnist_project

# List all projects
python project_manager.py list
```

### Manual Creation

1. Copy `projects/template/` to `projects/my_project/`
2. Edit `project_config.py`
3. Implement `custom_dataset.py`
4. Create `data/` directory and add CSV files

### Project Structure

```
my_project/
├── project_config.py      # BMConfig instance
├── custom_dataset.py      # Custom data loading
├── data/                  # Input data
│   ├── train.csv
│   ├── val.csv           # Optional (will split from train if not provided)
│   └── test.csv
└── outputs/               # Generated during training
    ├── checkpoints/
    ├── plots/
    ├── logs/
    └── models/
```

## Configuration

### Configuration Structure

```python
from bm_core.config import BMConfig, GRBMConfig, TrainingConfig

config = BMConfig(
    seed=42,                    # Random seed
    grbm=GRBMConfig(...),      # Model architecture
    training=TrainingConfig(...),  # Training parameters
    data=DataConfig(...),      # Data splits
    logging=LoggingConfig(...),    # Logging settings
    device={'use_cuda': 'auto'},   # GPU/CPU
    paths={...}                # Output directories
)
```

### GRBM Configuration

```python
GRBMConfig(
    n_visible=10,              # Number of visible units
    n_hidden=0,                # Number of hidden units (0 for FVBM)
    model_type="fvbm",         # "fvbm", "rbm", or "sbm"
    sparsity=None,             # None=dense, 0.0-1.0=sparse
    init_linear_scale=0.1,     # Initial bias scale
    init_quadratic_scale=0.1   # Initial weight scale
)
```

**Model types:**
- `fvbm`: Fully Visible BM (n_hidden must be 0)
- `rbm`: Restricted BM (bipartite, n_hidden > 0)
- `sbm`: Standard BM (fully connected, n_hidden > 0)

**Sparsity:**
- `None`: Dense connectivity (all allowed edges)
- `0.0-1.0`: Fraction of edges to include

### Training Configuration

```python
TrainingConfig(
    batch_size=5000,           # Samples per batch
    n_epochs=100,              # Number of epochs
    mode="pcd",                # "cd" or "pcd"
    cd_k=1,                    # CD-k steps (if mode='cd')
    pcd=PCDConfig(             # PCD settings (if mode='pcd')
        num_chains=100,
        k_steps=10,
        initialize_from="random"
    ),
    sampler_name="gibbs",      # Sampler to use
    sampler_params={           # Sampler-specific parameters
        'num_sweeps': 1000,
        'burn_in': 100
    },
    optimizer=OptimizerConfig(
        optimizer="adam",
        learning_rate=0.01
    ),
    gradient_clipping=GradientClippingConfig(
        enabled=True,
        max_norm=1.0
    ),
    regularization=RegularizationConfig(
        linear_l2=0.001,
        quadratic_l2=0.01
    )
)
```

### Available Samplers

**Classical MCMC:**
- `gibbs` - Gibbs sampling (recommended for small-medium models)
- `metropolis` - Metropolis-Hastings
- `parallel_tempering` - Parallel tempering (good for multimodal)
- `simulated_annealing` - Simulated annealing

**GPU Accelerated (for large models):**
- `gibbs_gpu` - GPU Gibbs (N > 1000)
- `metropolis_gpu` - GPU Metropolis
- `parallel_tempering_gpu` - GPU Parallel Tempering
- `simulated_annealing_gpu` - GPU Simulated Annealing
- `population_annealing_gpu` - GPU Population Annealing

**Exact (only N ≤ 20):**
- `exact` - Brute force enumeration
- `gumbel_max` - Gumbel-max sampling

**Optimization (for finding modes, not sampling):**
- `steepest_descent` - Local search
- `tabu` - Tabu search
- `greedy` - Greedy heuristic

**Baseline:**
- `random` - Random sampling

### Example Configurations

**Small Dense FVBM:**
```python
config = BMConfig(
    grbm=GRBMConfig(
        n_visible=10,
        n_hidden=0,
        model_type="fvbm"
    ),
    training=TrainingConfig(
        mode="cd",
        cd_k=1,
        sampler_name="gibbs"
    )
)
```

**Large Sparse FVBM with GPU:**
```python
config = BMConfig(
    grbm=GRBMConfig(
        n_visible=1000,
        n_hidden=0,
        model_type="fvbm",
        sparsity=0.05  # 5% connections
    ),
    training=TrainingConfig(
        mode="pcd",
        pcd=PCDConfig(num_chains=200, k_steps=10),
        sampler_name="gibbs_gpu",
        batch_size=10000
    )
)
```

**Restricted Boltzmann Machine:**
```python
config = BMConfig(
    grbm=GRBMConfig(
        n_visible=784,  # MNIST
        n_hidden=100,
        model_type="rbm"
    ),
    training=TrainingConfig(
        mode="pcd",
        sampler_name="gibbs",
        batch_size=128,
        n_epochs=50,
        hidden_kind="exact-disc"  # For RBM with hidden units
    )
)
```

## Custom Dataset Implementation

### Basic Implementation

```python
# projects/my_project/custom_dataset.py
import pandas as pd
import numpy as np
from bm_core.models import BMDataset

class MyDataset(BMDataset):
    def load_data(self, csv_path: str) -> np.ndarray:
        """
        Load and preprocess data.

        Must return numpy array of shape (n_samples, n_visible)
        with float32 dtype.
        """
        df = pd.read_csv(csv_path)

        # Extract visible columns
        visible_cols = [col for col in df.columns if col.startswith('v')]
        data = df[visible_cols].values

        return data.astype(np.float32)
```

### With Preprocessing

```python
class MyDataset(BMDataset):
    def load_data(self, csv_path: str) -> np.ndarray:
        df = pd.read_csv(csv_path)

        # Custom column extraction
        visible_cols = ['feature1', 'feature2', 'feature3']
        data = df[visible_cols].values

        # Normalization
        data = (data - data.mean(axis=0)) / data.std(axis=0)

        # Binarization to {-1, +1}
        data = np.where(data > 0, 1.0, -1.0)

        return data.astype(np.float32)
```

### With Data Augmentation

```python
class MyDataset(BMDataset):
    def __init__(self, csv_path: str, augment=False):
        self.augment = augment
        super().__init__(csv_path)

    def load_data(self, csv_path: str) -> np.ndarray:
        # Standard loading
        df = pd.read_csv(csv_path)
        data = df[[col for col in df.columns if col.startswith('v')]].values
        return data.astype(np.float32)

    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)

        if self.augment:
            # Add noise
            noise = torch.randn_like(sample) * 0.1
            sample = sample + noise

        return sample
```

### Image Data Example

```python
class ImageDataset(BMDataset):
    def load_data(self, csv_path: str) -> np.ndarray:
        # Load image paths and labels from CSV
        df = pd.read_csv(csv_path)

        data = []
        for img_path in df['image_path']:
            # Load image
            img = PIL.Image.open(img_path).convert('L')  # Grayscale
            img = img.resize((28, 28))

            # Convert to numpy and flatten
            img_array = np.array(img).flatten()

            # Normalize to [-1, 1]
            img_array = (img_array / 127.5) - 1.0

            data.append(img_array)

        return np.array(data, dtype=np.float32)
```

## Training

### Basic Training

```bash
python -m bm_core.bm --mode train \
  --config projects/my_project/project_config.py \
  --dataset projects/my_project/data/train.csv
```

### Training Output

```
Loading configuration from: projects/my_project/project_config.py
✓ Configuration validated

Using device: cuda:0

Initializing sampler factory...
  Registered 25 samplers

Building Boltzmann Machine
====================================================================

Topology:
  Model Type: FVBM
  Visible Units: 10
  Hidden Units: 0
  Total Nodes: 10
  Edges: 45
  Connectivity: dense

Model initialized:
...

Loading dataset from: projects/my_project/data/train.csv
  Total samples: 5000
  Visible units: 10

Dataset split:
  Train: 3500 (70.0%)
  Val:   750 (15.0%)
  Test:  750 (15.0%)

Starting training for 100 epochs...
====================================================================

Epoch 1/100: train_loss=2.345, val_loss=2.312, grad_norm=0.856
Epoch 2/100: train_loss=2.123, val_loss=2.098, grad_norm=0.742
...
```

### Monitoring Training

Training progress is saved to:
- **Checkpoints:** `outputs/checkpoints/best_model.pt`
- **Plots:** `outputs/plots/training_curves.png`
- **Logs:** `outputs/logs/training.log`

### Early Stopping

Enable in config:
```python
TrainingConfig(
    early_stopping=EarlyStoppingConfig(
        enabled=True,
        patience=20,
        metric="val_loss",
        mode="min"
    )
)
```

### Learning Rate Scheduling

```python
TrainingConfig(
    lr_scheduler=LRSchedulerConfig(
        enabled=True,
        type="plateau",  # "plateau", "step", "cosine", "exponential"
        factor=0.5,
        patience=15
    )
)
```

## Testing

### Test a Trained Model

```bash
python -m bm_core.bm --mode test \
  --config projects/my_project/project_config.py \
  --checkpoint projects/my_project/outputs/checkpoints/best_model.pt \
  --dataset projects/my_project/data/test.csv
```

### Load Model Programmatically

```python
from bm_core.models import BoltzmannMachine
from plugins.sampler_factory import SamplerFactory
import torch

# Initialize sampler factory
factory = SamplerFactory()
sampler_dict = factory.get_sampler_dict()

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BoltzmannMachine.load_checkpoint(
    'outputs/checkpoints/best_model.pt',
    sampler_dict=sampler_dict,
    device=device
)

# Sample from model
samples = model.sample('gibbs', prefactor=1.0, sample_params={'num_reads': 1000})
print(f"Generated {samples.shape[0]} samples")
```

## Advanced Topics

### Custom Training Loop

```python
from bm_core.models import BoltzmannMachine, create_dataloaders
from bm_core.trainers import BoltzmannMachineTrainer
from plugins.sampler_factory import SamplerFactory
import torch

# Setup
config = load_python_config('projects/my_project/project_config.py')
device = torch.device('cuda')
factory = SamplerFactory()
sampler_dict = factory.get_sampler_dict()

# Build model
model = build_model(config, sampler_dict, device)

# Load data
train_loader, val_loader, test_loader = create_dataloaders(
    'projects/my_project/data/train.csv',
    batch_size=config.training.batch_size
)

# Create trainer
trainer = BoltzmannMachineTrainer(model, config, device, sampler_name='gibbs')

# Train with custom logic
for epoch in range(config.training.n_epochs):
    train_metrics = trainer.train_epoch(train_loader)
    val_metrics = trainer.validate(val_loader)

    print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
          f"val_loss={val_metrics['loss']:.4f}")

    # Custom logic here
    if should_stop(val_metrics):
        break
```

### Hyperparameter Tuning

```python
from itertools import product

# Define hyperparameter grid
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [128, 512, 2048]
modes = ['cd', 'pcd']

results = []
for lr, bs, mode in product(learning_rates, batch_sizes, modes):
    config = BMConfig(
        training=TrainingConfig(
            learning_rate=lr,
            batch_size=bs,
            mode=mode
        )
    )

    # Train and evaluate
    val_loss = train_and_evaluate(config)
    results.append({'lr': lr, 'bs': bs, 'mode': mode, 'val_loss': val_loss})

# Find best
best = min(results, key=lambda x: x['val_loss'])
print(f"Best config: {best}")
```

### Ensemble Models

```python
# Train multiple models
models = []
for seed in [42, 43, 44, 45, 46]:
    config.seed = seed
    model = train_model(config)
    models.append(model)

# Sample from ensemble
def ensemble_sample(models, num_samples):
    all_samples = []
    for model in models:
        samples = model.sample('gibbs', sample_params={'num_reads': num_samples // len(models)})
        all_samples.append(samples)
    return torch.cat(all_samples, dim=0)

ensemble_samples = ensemble_sample(models, 1000)
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```
ModuleNotFoundError: No module named 'bm_core'
```
**Solution:** Run from project root:
```bash
cd c:\Users\jovan\Projects\bm_practice
python -m bm_core.bm ...
```

**2. Configuration Validation Errors**
```
ValueError: FVBM requires n_hidden=0, got n_hidden=10
```
**Solution:** Check model_type and n_hidden consistency:
- `fvbm` requires `n_hidden=0`
- `rbm`/`sbm` require `n_hidden>0`

**3. Sampler Not Found**
```
ValueError: Sampler 'gibbs_gpu' not found in sampler_dict
```
**Solution:** Check available samplers:
```python
from plugins.sampler_factory import SamplerFactory
factory = SamplerFactory()
print(factory.list_samplers())
```

**4. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce `batch_size`
- Reduce `num_chains` (for PCD)
- Use CPU sampler: `sampler_name="gibbs"`
- Use sparse connectivity: `sparsity=0.1`

**5. Slow Training**
**Solutions:**
- Use GPU sampler: `sampler_name="gibbs_gpu"`
- Increase `batch_size`
- Reduce `num_sweeps` in sampler_params
- Use CD instead of PCD for faster iterations

### Debugging Tips

**Enable verbose output:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check model parameters:**
```python
linear, quadratic = model.get_parameters()
print(f"Linear range: [{linear.min():.3f}, {linear.max():.3f}]")
print(f"Quadratic range: [{quadratic.min():.3f}, {quadratic.max():.3f}]")
```

**Visualize samples:**
```python
samples = model.sample('gibbs', sample_params={'num_reads': 100})
import matplotlib.pyplot as plt
plt.hist(samples.cpu().numpy().flatten())
plt.show()
```

**Profile training:**
```python
import time
start = time.time()
metrics = trainer.train_epoch(train_loader)
print(f"Epoch time: {time.time() - start:.2f}s")
```

## Next Steps

- Read [Architecture Guide](architecture.md) for system design
- Check [QUICKSTART.md](../QUICKSTART.md) for examples
- Explore example projects in `projects/`
- Experiment with different configurations
- Try different samplers and compare performance

## Support

For issues or questions:
1. Check this user guide
2. Review architecture documentation
3. Examine example projects
4. Check existing GitHub issues
5. Create a new issue with reproducible example
