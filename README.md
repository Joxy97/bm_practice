# Boltzmann Machine Training Pipeline

A modular, production-ready pipeline for training Boltzmann Machines using D-Wave's PyTorch plugin.

## Overview

This project provides a complete end-to-end pipeline for:
- Generating training data from a "true" Boltzmann Machine
- Training a model to learn the true parameters
- Evaluating and comparing results
- Comprehensive visualization

## Quick Start

```bash
cd bm_pipeline
python main.py --mode full --config configs/config.yaml
```

See [QUICKSTART.md](QUICKSTART.md) for detailed usage instructions.

## Features

### Architecture Support
- **Fully-Connected**: All visible nodes connected (dense graph)
- **Restricted**: Sparse connectivity with configurable density
- **RBM**: Bipartite graphs with hidden units

### Training Features
- **PyTorch Integration**: Full integration with PyTorch optimizers
- **Data Management**: Automated CSV storage, train/val/test splits, batching
- **Early Stopping**: Automatic stopping based on validation loss
- **Checkpointing**: Saves best and final models
- **Validation**: Proper train/val/test separation

### Visualization
- Model parameters (biases and weights)
- Training curves (loss, gradients, temperature)
- Side-by-side true vs learned comparison
- Data statistics and correlations

### Code Quality
- Modular design with clean separation of concerns
- Configuration-driven (YAML)
- Reproducible (fixed random seeds)
- Well-documented
- Type hints throughout

## Project Structure

```
bm_pipeline/
├── configs/
│   └── config.yaml           # All hyperparameters
│
├── models/
│   ├── data_generator.py     # DataGenerator class
│   └── dataset.py            # PyTorch Dataset/DataLoader
│
├── trainers/
│   └── bm_trainer.py         # BoltzmannMachineTrainer class
│
├── utils/
│   ├── topology.py           # Graph topology creation
│   ├── parameters.py         # Parameter generation
│   ├── visualization.py      # Plotting functions
│   └── config_loader.py      # Config management
│
├── outputs/                  # Generated files (git-ignored)
│   ├── data/                 # CSV datasets
│   ├── plots/                # PNG visualizations
│   ├── models/               # Model checkpoints (.pt)
│   └── checkpoints/          # Best model checkpoints
│
├── main.py                   # Main entry point
├── run.py                    # Alternative runner
├── .gitignore                # Excludes outputs
├── README.md                 # This file
├── QUICKSTART.md             # Quick start guide
└── requirements.txt          # Dependencies
```

## Installation

```bash
# Navigate to project
cd bm_practice/bm_pipeline

# Install dependencies
pip install -r requirements.txt
```

Required packages:
- `numpy`, `pandas`, `torch`, `matplotlib`, `seaborn`, `pyyaml`
- `dwave-ocean-sdk`, `dwave-pytorch-plugin`, `dimod`

## Usage

### Full Pipeline

```bash
python main.py --mode full --config configs/config.yaml
```

Runs:
1. Data generation from true model
2. Model training with validation
3. Testing and comparison

### Individual Steps

**Generate data only:**
```bash
python main.py --mode generate --config configs/config.yaml
```

**Train model only** (requires existing dataset):
```bash
python main.py --mode train --config configs/config.yaml
```

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Model architecture
true_model:
  n_visible: 4
  n_hidden: 0
  architecture: "fully-connected"

# Data generation
data:
  n_samples: 5000
  train_ratio: 0.7
  val_ratio: 0.15

# Training
training:
  batch_size: 128
  n_epochs: 100
  learning_rate: 0.1
  optimizer: "sgd"
```

## Examples

### Example 1: Fully Visible BM (Default)
```yaml
true_model:
  n_visible: 4
  n_hidden: 0
  architecture: "fully-connected"
```

### Example 2: Restricted Boltzmann Machine
```yaml
true_model:
  n_visible: 6
  n_hidden: 3
  architecture: "fully-connected"  # Creates bipartite graph

training:
  hidden_kind: "exact-disc"  # Exact marginalization
```

### Example 3: Large Sparse Network
```yaml
true_model:
  n_visible: 10
  n_hidden: 0
  architecture: "restricted"
  connectivity: 0.3
```

## Output Files

All outputs are saved in `outputs/`:

### Data
- `outputs/data/bm_dataset_v1.csv` - Generated samples with metadata

### Plots
- `outputs/plots/true_model_parameters.png`
- `outputs/plots/learned_model_parameters.png`
- `outputs/plots/training_history.png`
- `outputs/plots/model_comparison.png`

### Models
- `outputs/models/final_model.pt` - Final epoch
- `outputs/checkpoints/best_model.pt` - Best validation loss

## Key Classes

### DataGenerator
```python
from models import DataGenerator

data_gen = DataGenerator(config)
df = data_gen.generate(save_dir='outputs/data')
true_model = data_gen.get_true_model()
```

### BoltzmannMachineDataset
```python
from models import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    dataset_path='outputs/data/dataset.csv',
    batch_size=128,
    train_ratio=0.7,
    val_ratio=0.15
)
```

### BoltzmannMachineTrainer
```python
from trainers import BoltzmannMachineTrainer

trainer = BoltzmannMachineTrainer(model, config, sampler)
trainer.train(train_loader, val_loader)
test_metrics = trainer.test(test_loader)
history = trainer.get_history()
```

## Advanced Features

### Early Stopping
```yaml
training:
  early_stopping:
    enabled: true
    patience: 20
    min_delta: 0.0001
```

### Custom Sampler Parameters
```yaml
data:
  sampler_params:
    beta_range: [1.0, 1.0]
    proposal_acceptance_criteria: "Gibbs"
```

### Checkpoint Management
```yaml
training:
  save_best_model: true
  checkpoint_dir: "outputs/checkpoints"
```

## Development

The codebase follows best practices:
- Type hints for clarity
- Docstrings for all classes/functions
- Modular design for extensibility
- Configuration-driven for flexibility
- Git-ignored outputs for clean repository

## Extending the Pipeline

### Add Custom Metrics
Edit `trainers/bm_trainer.py` to track additional metrics in `history`.

### Custom Visualization
Add functions to `utils/visualization.py`.

### New Architecture Types
Extend `utils/topology.py` with new graph generation functions.

## Troubleshooting

**Import errors**: Run from `bm_pipeline/` directory
```bash
cd bm_pipeline
python main.py ...
```

**D-Wave not found**: Install dependencies
```bash
pip install dwave-ocean-sdk dwave-pytorch-plugin
```

**Out of memory**: Reduce batch size or sample size in `configs/config.yaml`

## License

MIT License

## Citation

If you use this code, please cite:
```
Boltzmann Machine Training Pipeline
https://github.com/yourusername/bm_practice
```

## Contact

For questions or issues, please open a GitHub issue.
