# Quick Start Guide

## New Workflow with Refactored Structure

The BM practice project has been refactored into a clean, modular pipeline. Here's how to use it:

## 1. Create a New Project

```bash
python -m projects.project_manager create --name my_first_project
```

This creates:
```
projects/my_first_project/
├── project_config.py      # Your BM configuration (edit this)
├── custom_dataset.py      # Your data loading logic (edit this)
├── data/                  # Place your CSV files here
└── outputs/               # Training outputs go here
```

## 2. Configure Your BM

Edit `projects/my_first_project/project_config.py`:

```python
from bm_core.config import BMConfig, GRBMConfig, TrainingConfig

config = BMConfig(
    seed=42,

    # GRBM Architecture
    grbm=GRBMConfig(
        n_visible=10,          # Match your data dimensions
        n_hidden=0,            # 0 for FVBM, >0 for RBM
        model_type="fvbm",     # "fvbm", "rbm", or "sbm"
        sparsity=None,         # None=dense, 0.1=10% connections
    ),

    # Training
    training=TrainingConfig(
        batch_size=5000,
        n_epochs=100,
        mode="pcd",            # "cd" or "pcd"
        sampler_name="gibbs",  # "gibbs", "gibbs_gpu", "metropolis", etc.
    )
)
```

## 3. Implement Your Dataset (if custom data format)

Edit `projects/my_first_project/custom_dataset.py`:

```python
from bm_core.models import BMDataset
import pandas as pd
import numpy as np

class MyDataset(BMDataset):
    def load_data(self, csv_path: str) -> np.ndarray:
        df = pd.read_csv(csv_path)

        # Your custom preprocessing
        visible_cols = [col for col in df.columns if col.startswith('v')]
        data = df[visible_cols].values.astype(np.float32)

        return data
```

**Note:** If your CSV already has columns named `v0`, `v1`, ... `vN`, you can skip this step and use the default implementation.

## 4. Prepare Your Data

Place your CSV data files in `projects/my_first_project/data/`:

```
projects/my_first_project/data/train.csv
```

**CSV Format:**
```
v0,v1,v2,v3,v4,v5,v6,v7,v8,v9
-1,1,-1,1,1,-1,-1,1,1,-1
1,1,-1,-1,1,1,-1,-1,1,1
...
```

## 5. Train Your Model

```bash
python -m bm_core.bm --mode train \
  --config projects/my_first_project/project_config.py \
  --dataset projects/my_first_project/data/train.csv
```

Output:
- Model checkpoints: `projects/my_first_project/outputs/checkpoints/best_model.pt`
- Training logs: `projects/my_first_project/outputs/logs/`
- Plots: `projects/my_first_project/outputs/plots/`

## 6. Test Your Model

```bash
python -m bm_core.bm --mode test \
  --config projects/my_first_project/project_config.py \
  --checkpoint projects/my_first_project/outputs/checkpoints/best_model.pt \
  --dataset projects/my_first_project/data/test.csv
```

## Available Modes

### Build Mode (Initialize Only)
```bash
python -m bm_core.bm --mode build \
  --config projects/my_first_project/project_config.py
```
Creates and displays the model without training.

### Train Mode
```bash
python -m bm_core.bm --mode train \
  --config <config.py> \
  --dataset <train.csv>
```
Trains the model and saves checkpoints.

### Test Mode
```bash
python -m bm_core.bm --mode test \
  --config <config.py> \
  --checkpoint <best_model.pt> \
  --dataset <test.csv>
```
Tests a trained model on new data.

## Configuration Options

### Model Types
- `fvbm`: Fully Visible BM (n_hidden=0, all visible units connected)
- `rbm`: Restricted BM (bipartite graph, n_hidden>0)
- `sbm`: Standard BM (full graph, n_hidden>0)

### Training Modes
- `cd`: Contrastive Divergence (fast, good for small models)
- `pcd`: Persistent CD (better for large models, more accurate)

### Available Samplers
**Classical MCMC:**
- `gibbs` - Gibbs sampling (recommended)
- `metropolis` - Metropolis-Hastings
- `parallel_tempering` - Parallel tempering
- `simulated_annealing` - Simulated annealing

**GPU Accelerated:**
- `gibbs_gpu` - GPU Gibbs (for large models)
- `metropolis_gpu` - GPU Metropolis
- `parallel_tempering_gpu` - GPU Parallel Tempering
- `simulated_annealing_gpu` - GPU Simulated Annealing
- `population_annealing_gpu` - GPU Population Annealing

**Exact (N ≤ 20):**
- `exact` - Brute force enumeration
- `gumbel_max` - Gumbel-max sampling

**Optimization:**
- `steepest_descent` - Local search
- `tabu` - Tabu search
- `greedy` - Greedy heuristic

**Baseline:**
- `random` - Random sampling

## Example: MNIST-like Project

```python
# projects/mnist_project/project_config.py
config = BMConfig(
    grbm=GRBMConfig(
        n_visible=784,  # 28x28 images
        n_hidden=100,
        model_type="rbm",
        sparsity=0.1
    ),
    training=TrainingConfig(
        batch_size=128,
        n_epochs=50,
        mode="pcd",
        sampler_name="gibbs_gpu"  # Use GPU for large models
    )
)
```

## Example: Sparse FVBM Project

```python
# projects/sparse_project/project_config.py
config = BMConfig(
    grbm=GRBMConfig(
        n_visible=1000,
        n_hidden=0,
        model_type="fvbm",
        sparsity=0.05  # Only 5% connections
    ),
    training=TrainingConfig(
        batch_size=10000,
        n_epochs=200,
        mode="pcd",
        sampler_name="gibbs_gpu"
    )
)
```

## Project Management

### List all projects
```bash
python -m projects.project_manager list
```

### Create from custom template
```bash
python -m projects.project_manager create --name my_project --template custom_template
```

## Troubleshooting

### Import Errors
Make sure you're running from the `bm_practice` directory:
```bash
cd c:\Users\jovan\Projects\bm_practice
python -m bm_core.bm --mode train ...
```

### Config Validation Errors
Check that:
- `n_hidden=0` for FVBM
- `n_hidden>0` for RBM/SBM
- Data ratios sum to 1.0
- Sparsity is between 0.0 and 1.0

### Sampler Not Found
Check available samplers:
```python
from plugins.sampler_factory import SamplerFactory
factory = SamplerFactory()
print(factory.list_samplers())
```

### GPU Issues
If GPU samplers fail, check CUDA availability:
```python
import torch
print(torch.cuda.is_available())
```
Fall back to CPU samplers if needed: `sampler_name="gibbs"`

## Next Steps

1. Try the example workflow above
2. Experiment with different configurations
3. Create multiple projects for different use-cases
4. Read [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) for architecture details

## Key Files

- **Configuration:** Python dataclasses in `bm_core/config/bm_config_template.py`
- **Main Entry Point:** `bm_core/bm.py`
- **BM Abstraction:** `bm_core/models/bm_model.py`
- **Dataset Base Class:** `bm_core/models/dataset.py`
- **Sampler Factory:** `plugins/sampler_factory/sampler_factory.py`
- **Project Manager:** `projects/project_manager.py`

## Advantages of New Structure

1. **Type Safety:** Python configs with IDE autocomplete
2. **Modularity:** Clear separation of core, plugins, projects
3. **Extensibility:** Easy to add samplers, models, projects
4. **Team-Friendly:** Each project is self-contained
5. **Future-Proof:** BM abstraction allows backend swapping

Happy training! 🚀
