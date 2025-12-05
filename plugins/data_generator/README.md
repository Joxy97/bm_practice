# Data Generator Plugin

Generates synthetic data by sampling from a true Boltzmann Machine with known parameters.

## Purpose

This plugin is useful for:
- **Testing** - Generate ground truth data for testing training algorithms
- **Benchmarking** - Compare samplers with known distributions
- **Validation** - Verify implementations against known parameters
- **Debugging** - Quick data generation for development

## Usage

### Standalone CLI

```bash
python -m plugins.data_generator.run_generator \
  --config plugins/data_generator/data_generator_config.yaml \
  --output-dir my_data/
```

### As Python Module

```python
from plugins.data_generator import SyntheticDataGenerator
from plugins.sampler_factory import SamplerFactory
import yaml

# Load config
with open('plugins/data_generator/data_generator_config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize
factory = SamplerFactory()
generator = SyntheticDataGenerator(config, factory.get_sampler_dict())

# Generate data
df = generator.generate('output_directory/')

# Access true model
true_model = generator.get_true_model()
```

## Configuration

Edit `data_generator_config.yaml`:

```yaml
true_model:
  n_visible: 10
  n_hidden: 0
  model_type: "fvbm"
  connectivity: "dense"

data:
  dataset_name: "synthetic_bm_data"
  n_samples: 5000
  sampler_type: "gibbs"
```

## Output Format

Generated CSV files have the format:

```csv
v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,sample_id
-1,1,-1,1,1,-1,-1,1,1,-1,0
1,1,-1,-1,1,1,-1,-1,1,1,1
...
```

- Columns `v0` to `vN`: Visible unit values in SPIN format ({-1, +1})
- Column `sample_id`: Sample identifier

## Integration with Projects

Use generated data in your projects:

```bash
# Generate data
python -m plugins.data_generator.run_generator \
  --config plugins/data_generator/data_generator_config.yaml \
  --output-dir projects/my_project/data/

# Train on generated data
python -m bm_core.bm --mode train \
  --config projects/my_project/project_config.py \
  --dataset projects/my_project/data/synthetic_bm_data.csv
```

## Examples

### Small Dense FVBM
```yaml
true_model:
  n_visible: 10
  n_hidden: 0
  model_type: "fvbm"
data:
  n_samples: 5000
  sampler_type: "gibbs"
```

### Restricted Boltzmann Machine
```yaml
true_model:
  n_visible: 100
  n_hidden: 20
  model_type: "rbm"
data:
  n_samples: 10000
  sampler_type: "gibbs_gpu"
```

### Sparse BM
```yaml
true_model:
  n_visible: 1000
  n_hidden: 0
  model_type: "fvbm"
  connectivity: "sparse"
  connectivity_density: 0.05
data:
  n_samples: 20000
  sampler_type: "gibbs_gpu"
```

## Tips

1. **Large models**: Use GPU samplers (`gibbs_gpu`, `metropolis_gpu`)
2. **Quality**: Increase `num_sweeps` and `burn_in` for better samples
3. **Speed**: Reduce `burn_in` for faster generation (less accurate)
4. **Reproducibility**: Set `seed` for consistent results
