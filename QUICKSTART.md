# Quick Start Guide

## Installation

```bash
cd bm_practice/bm_pipeline
pip install -r requirements.txt
```

## Run the Full Pipeline

Navigate to the `bm_pipeline` directory and run:

```bash
cd bm_pipeline
python main.py --mode full --config configs/config.yaml
```

This will:
1. Generate 5000 training samples from a true BM
2. Split data into train/val/test (70%/15%/15%)
3. Train a learned model to reverse-engineer the true parameters
4. Compare and visualize results

## Output

After running, all outputs are organized in `bm_pipeline/outputs/`:

```
bm_pipeline/outputs/{dataset_name+date_time}
├── data/
│   └── {dataset_name}.csv            # Generated samples
├── plots/
│   ├── true_model_parameters.png     # True model visualization
│   ├── learned_model_parameters.png  # Learned model visualization
│   ├── training_history.png          # Loss, gradients, temperature
│   └── model_comparison.png          # Side-by-side comparison
├── models/
│   └── final_model.pt                # Final epoch checkpoint
└── checkpoints/
    └── best_model.pt                 # Best validation checkpoint
```

## Individual Pipeline Steps

### 1. Generate Data Only

```bash
python main.py --mode generate --config configs/config.yaml
```

### 2. Train Model Only (requires existing dataset)

```bash
python main.py --mode train --config configs/config.yaml
```

## Customize Configuration

Edit `configs/config.yaml`:

```yaml
# Change model size
true_model:
  n_visible: 6      # Try: 4, 6, 8
  n_hidden: 2       # Try: 0 (fully visible), 2, 3

# Change architecture
true_model:
  architecture: "restricted"  # or "fully-connected"
  connectivity: 0.5           # for restricted

# Change training
training:
  batch_size: 256
  n_epochs: 200
  learning_rate: 0.05
  model_sample_size: 2000
```

## Example Configurations

### Small Fully-Connected Model (Default)
```yaml
n_visible: 4
n_hidden: 0
architecture: "fully-connected"
```

### Restricted Boltzmann Machine
```yaml
n_visible: 6
n_hidden: 3
architecture: "fully-connected"  # Bipartite
hidden_kind: "exact-disc"
```

### Large Sparse Network
```yaml
n_visible: 10
n_hidden: 0
architecture: "restricted"
connectivity: 0.4
```

## Troubleshooting

### Issue: Import errors
**Solution**: Make sure you're running from the `bm_pipeline` directory:
```bash
cd bm_pipeline
python main.py --mode full --config configs/config.yaml
```

### Issue: D-Wave not found
**Solution**:
```bash
pip install dwave-ocean-sdk dwave-pytorch-plugin
```

### Issue: Out of memory
**Solution**: Reduce in `configs/config.yaml`:
```yaml
training:
  batch_size: 32  # Reduce from 128
  model_sample_size: 500  # Reduce from 1000
```

## Project Structure

```
bm_pipeline/
├── configs/
│   └── config.yaml        # Configuration file
├── models/
│   ├── data_generator.py  # Sample from true BM
│   └── dataset.py         # PyTorch DataLoader
├── trainers/
│   └── bm_trainer.py      # Training logic
├── utils/
│   ├── topology.py        # Graph creation
│   ├── parameters.py      # Parameter generation
│   ├── visualization.py   # Plotting
│   └── config_loader.py   # Config management
├── outputs/               # All generated files (git-ignored)
│   ├── data/
│   ├── plots/
│   ├── models/
│   └── checkpoints/
├── main.py               # Main entry point
├── run.py                # Alternative entry point
├── .gitignore            # Ignore outputs
├── README.md             # Full documentation
└── requirements.txt      # Dependencies
```

## Key Features

- **Clean Organization**: All outputs in one `outputs/` directory
- **Modular Design**: Separated data generation, training, and evaluation
- **Reproducibility**: Fixed random seeds throughout
- **Flexibility**: Supports fully-connected and restricted topologies
- **Hidden Units**: Optional hidden units with exact marginalization
- **Early Stopping**: Automatic stopping when validation loss plateaus
- **Checkpointing**: Saves best and final models
- **Visualization**: Comprehensive plots for analysis

## Next Steps

1. Try different architectures in `configs/config.yaml`
2. Experiment with hyperparameters
3. Compare results across different seeds
4. Extend with custom metrics or visualizations
