# Run Directory System

## Overview

The pipeline now uses timestamped run directories to organize all outputs from each training run. This makes it easy to:
- Track different experiments
- Compare results across runs
- Reproduce experiments with saved configs
- Keep outputs organized

## Directory Structure

Each run creates a timestamped directory:

```
outputs/
  {dataset_name}_{timestamp}/
    ├── config.yaml           # Copy of configuration used for this run
    ├── data/                 # Generated training data
    │   └── {dataset_name}.csv
    ├── models/               # Final trained models
    │   └── final_model.pt
    ├── checkpoints/          # Best model checkpoints during training
    │   └── best_model.pt
    ├── plots/                # All visualizations
    │   ├── true_model_parameters.png
    │   ├── learned_model_parameters.png
    │   ├── training_history.png
    │   └── model_comparison.png
    └── logs/                 # Training logs (future use)
```

## Run Naming Convention

Format: `{dataset_name}_{YYYYMMDD}_{HHMMSS}`

Examples:
- `bm_toy_dataset_20251204_103752`
- `bm_experiment1_20251204_141530`

The dataset name comes from the `data.dataset_name` field in `config.yaml`.

## Usage

### Running the Pipeline

Simply run the pipeline as usual:

```bash
# Full pipeline (generate + train + compare)
python main.py --mode full --config configs/config.yaml

# Generate data only
python main.py --mode generate --config configs/config.yaml

# Train only (requires existing data)
python main.py --mode train --config configs/config.yaml --dataset path/to/data.csv
```

Each run automatically:
1. Creates a timestamped directory
2. Copies the config to the run directory
3. Saves all outputs to that directory
4. Prints the run location at completion

### Viewing Past Runs

Use the `list_runs.py` utility:

```bash
# List all runs
python list_runs.py

# Show latest run details
python list_runs.py --latest

# Show specific run details
python list_runs.py --run bm_toy_dataset_20251204_103752

# Filter by dataset name
python list_runs.py --dataset bm_toy_dataset
```

### Reproducing a Run

To reproduce a past experiment:

1. Navigate to the run directory:
   ```bash
   cd outputs/bm_toy_dataset_20251204_103752/
   ```

2. Use the saved config:
   ```bash
   python ../../main.py --mode full --config config.yaml
   ```

The saved config contains the exact hyperparameters, architecture, and settings used for that run.

## Configuration

The run directory is automatically created based on `config.yaml`:

```yaml
data:
  dataset_name: "bm_toy_dataset"  # Used in run directory name
  # ... other settings
```

No manual path configuration needed! The system handles everything automatically.

## Benefits

### 1. **Organization**
- No more scattered files
- Easy to find specific experiments
- Clean separation between runs

### 2. **Reproducibility**
- Config saved with each run
- Know exactly what settings were used
- Can reproduce any experiment

### 3. **Comparison**
- Easy to compare multiple runs
- All outputs in one place per run
- Timestamped for chronological tracking

### 4. **Collaboration**
- Share entire run directories
- Recipients get config + results
- No ambiguity about settings

## Advanced Usage

### Programmatic Access

You can use the run manager utilities in your own scripts:

```python
from utils import create_run_directory, list_runs, get_latest_run, print_run_summary

# Create a new run directory
config = load_config('configs/config.yaml')
run_paths = create_run_directory(config)
# Returns: {'run_dir': '...', 'data_dir': '...', 'model_dir': '...', ...}

# List all runs
runs = list_runs('outputs', dataset_name='bm_toy_dataset')

# Get latest run
latest_run = get_latest_run('outputs')

# Print run summary
print_run_summary(latest_run)
```

### Custom Base Directory

By default, runs are saved to `outputs/`. To change this, modify `run_manager.py` or pass a custom base directory:

```python
run_paths = create_run_directory(config, base_output_dir='my_experiments')
```

## Migration from Old System

If you have old output files (before the run directory system), they will remain in:
- `outputs/data/`
- `outputs/models/`
- `outputs/plots/`

These won't interfere with the new timestamped run system. New runs will create their own timestamped directories.

## Tips

### 1. Descriptive Dataset Names
Use descriptive dataset names in config for easier identification:
```yaml
data:
  dataset_name: "bm_10nodes_fc_exp1"  # Better than "bm_dataset_v1"
```

### 2. Run Notes
Add a README.md to a run directory for notes:
```bash
echo "Testing learning rate 0.001" > outputs/bm_toy_dataset_20251204_103752/README.md
```

### 3. Cleaning Old Runs
Safely delete old runs by removing their directories:
```bash
rm -rf outputs/bm_toy_dataset_20251203_*
```

### 4. Archiving Important Runs
Archive important runs:
```bash
tar -czf important_run.tar.gz outputs/bm_toy_dataset_20251204_103752/
```

## Summary

The run directory system provides:
✅ Automatic organization of all outputs
✅ Timestamped runs for easy tracking
✅ Config archiving for reproducibility
✅ Clean separation between experiments
✅ Utilities for viewing past runs
✅ No manual path management needed

Just run your experiments and everything is automatically organized!
