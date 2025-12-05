# Architecture Guide

## Overview

The BM Practice project is organized into three main components:

1. **Core BM Pipeline** (`bm_core/`) - Main training and inference pipeline
2. **Plugins** (`plugins/`) - Self-contained extensions (sampler factory, benchmarking, data generation)
3. **Projects** (`projects/`) - User-specific use-case implementations

This architecture provides clean separation of concerns, modularity, and extensibility.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User's Project                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  project_config.py (BMConfig)                        │  │
│  │  custom_dataset.py (MyDataset extends BMDataset)     │  │
│  │  data/*.csv (User's data files)                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Core BM Pipeline                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  bm.py (CLI: build/train/test)                      │  │
│  │  BoltzmannMachine (abstraction over D-Wave GRBM)    │  │
│  │  BoltzmannMachineTrainer (CD/PCD training)          │  │
│  │  BMDataset (base class for data loading)            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Sampler Factory Plugin                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SamplerFactory → {name: sampler_instance}          │  │
│  │  25+ Samplers: Gibbs, Metropolis, GPU variants,     │  │
│  │                Exact, Optimization, Baseline         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    D-Wave GRBM Backend                       │
│  (dwave.plugins.torch.models.GraphRestrictedBoltzmannMachine) │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Core BM Pipeline (`bm_core/`)

The core pipeline provides the fundamental BM training and inference functionality.

#### Key Modules

**`bm_core/bm.py`** - Main entry point
- CLI with three modes: `build`, `train`, `test`
- Loads Python-based configuration
- Orchestrates the training pipeline
- Integrates sampler factory

**`bm_core/models/bm_model.py`** - BoltzmannMachine abstraction
- Wraps D-Wave's GRBM implementation
- Provides clean API: `sample()`, `get_parameters()`, `set_parameters()`
- Accepts sampler_dict from factory
- Enables future backend swapping (PyTorch native, TensorFlow, JAX)

**`bm_core/models/dataset.py`** - Data loading
- `BMDataset` base class for custom data loading
- Users extend and override `load_data()` method
- Standard PyTorch Dataset pattern
- `create_dataloaders()` utility for train/val/test splits

**`bm_core/trainers/bm_trainer.py`** - Training algorithms
- Supports Contrastive Divergence (CD) and Persistent CD (PCD)
- Gradient clipping, regularization, learning rate scheduling
- Early stopping, checkpointing
- Validation and testing

**`bm_core/config/bm_config_template.py`** - Configuration
- Python dataclasses for type-safe configuration
- `BMConfig`, `GRBMConfig`, `TrainingConfig`, etc.
- Validation and helper functions
- IDE autocomplete support

**`bm_core/utils/`** - Utilities
- `topology.py` - Graph topology creation (FVBM, RBM, SBM)
- `parameters.py` - Parameter initialization
- `device.py` - GPU/CPU device management
- `visualization.py` - Plotting utilities
- `run_manager.py` - Run directory management

#### Design Patterns

**Abstraction Layer:**
```python
# BoltzmannMachine wraps D-Wave GRBM
class BoltzmannMachine:
    def __init__(self, nodes, edges, hidden_nodes, linear, quadratic, sampler_dict):
        self._backend = GRBM(...)  # D-Wave implementation
        self.sampler_dict = sampler_dict

    def sample(self, sampler_name: str, **kwargs):
        sampler = self.sampler_dict[sampler_name]
        return self._backend.sample(sampler, ...)
```

**Benefits:**
- Decouples pipeline from D-Wave specifics
- Easy to swap backends in future
- Cleaner API for users
- Easier testing and mocking

### 2. Sampler Factory Plugin (`plugins/sampler_factory/`)

The sampler factory provides centralized sampler management.

#### Architecture

**`sampler_factory.py`** - Factory class
```python
class SamplerFactory:
    def __init__(self, config_path: Optional[str] = None):
        self._registry = {}
        self._register_all_samplers()

    def get_sampler_dict(self) -> Dict[str, Sampler]:
        """Returns {name: sampler_instance}"""
        return self._registry

    def get_sampler(self, name: str) -> Sampler:
        """Get specific sampler by name"""
        return self._registry[name]
```

**Sampler Categories:**

1. **Classical MCMC** - `gibbs`, `metropolis`, `parallel_tempering`, `simulated_annealing`
2. **GPU Accelerated** - `gibbs_gpu`, `metropolis_gpu`, `parallel_tempering_gpu`, etc.
3. **Exact** - `exact`, `gumbel_max` (N ≤ 20)
4. **Optimization** - `steepest_descent`, `tabu`, `greedy`
5. **Baseline** - `random`

**Integration Points:**

```python
# Core pipeline uses sampler_dict
factory = SamplerFactory()
sampler_dict = factory.get_sampler_dict()
model = BoltzmannMachine(..., sampler_dict=sampler_dict)

# Trainer uses samplers by name
trainer = BoltzmannMachineTrainer(model, config, device, sampler_name='gibbs')
```

#### Plugin Benefits

- **Self-contained:** Own directory, config, samplers
- **No coupling:** Core doesn't know about specific samplers
- **Extensible:** Easy to add new samplers
- **Reusable:** Can be used by benchmark, data generator, etc.

### 3. Project Template System (`projects/`)

The project system provides templates for user-specific use-cases.

#### Structure

```
projects/
├── project_manager.py     # CLI for creating projects
├── template/              # Base template
│   ├── project_config.py  # BMConfig instance
│   ├── custom_dataset.py  # Custom Dataset implementation
│   ├── data/             # User's CSV files
│   └── outputs/          # Training outputs
└── [user_projects]/       # User-created projects
```

#### Workflow

**1. User creates project:**
```bash
python project_manager.py create --name mnist_project
```

**2. User customizes config:**
```python
# projects/mnist_project/project_config.py
config = BMConfig(
    grbm=GRBMConfig(n_visible=784, n_hidden=100),
    training=TrainingConfig(batch_size=128, n_epochs=50)
)
```

**3. User implements dataset:**
```python
# projects/mnist_project/custom_dataset.py
class MNISTDataset(BMDataset):
    def load_data(self, csv_path: str):
        # Custom MNIST loading logic
        return data
```

**4. User trains:**
```bash
python -m bm_core.bm --mode train \
  --config projects/mnist_project/project_config.py \
  --dataset projects/mnist_project/data/train.csv
```

#### Benefits

- **Standardization:** All projects follow same structure
- **Isolation:** Each project is self-contained
- **Minimal boilerplate:** Template handles common setup
- **Clear responsibility:** User only implements data loading

## Data Flow

### Training Pipeline

```
1. User CSV Data
   └─> custom_dataset.py::load_data()
       └─> BMDataset::__getitem__()
           └─> DataLoader batches
               └─> Trainer

2. Trainer
   └─> BoltzmannMachine::sample(sampler_name)
       └─> sampler_dict[sampler_name]
           └─> D-Wave GRBM::sample()
               └─> Model samples

3. Training Step
   Data batch + Model samples
   └─> BoltzmannMachine::quasi_objective()
       └─> Loss computation
           └─> Backpropagation
               └─> Parameter update
```

### Configuration Flow

```
1. project_config.py (Python dataclass)
   └─> BMConfig instance
       └─> Validation
           └─> bm.py loads config
               └─> Creates BoltzmannMachine
                   └─> Creates Trainer
                       └─> Training loop
```

## Extension Points

### Adding New Samplers

**1. Implement sampler:**
```python
# plugins/sampler_factory/samplers/custom.py
from .base import BaseSampler, register_sampler

@register_sampler('my_sampler')
class MySampler(BaseSampler):
    def sample(self, energy_fn, n_variables, num_samples, **kwargs):
        # Your sampling logic
        return samples
```

**2. Register in factory:**
```python
# plugins/sampler_factory/sampler_factory.py
def _register_my_sampler(self):
    from plugins.sampler_factory.samplers.custom import MySampler
    self._registry['my_sampler'] = MySampler(...)
```

**3. Use in config:**
```python
config = BMConfig(
    training=TrainingConfig(sampler_name='my_sampler')
)
```

### Adding New Model Types

**Option 1: Extend D-Wave GRBM (current)**
```python
# Modify topology.py to add new connectivity patterns
def create_topology(..., model_type="my_custom_type"):
    if model_type == "my_custom_type":
        # Custom topology logic
        return nodes, edges, hidden_nodes
```

**Option 2: New Backend (future)**
```python
# bm_core/models/bm_model.py
class BoltzmannMachine:
    def _create_backend(self, backend_type='dwave'):
        if backend_type == 'dwave':
            return GRBM(...)
        elif backend_type == 'pytorch_native':
            return PyTorchBM(...)  # Custom implementation
        elif backend_type == 'jax':
            return JAXBM(...)  # JAX implementation
```

### Adding New Training Algorithms

**Extend trainer:**
```python
# bm_core/trainers/bm_trainer.py
class BoltzmannMachineTrainer:
    def train_epoch(self, train_loader):
        if self.training_mode == 'cd':
            return self._train_epoch_cd(train_loader)
        elif self.training_mode == 'pcd':
            return self._train_epoch_pcd(train_loader)
        elif self.training_mode == 'my_algorithm':
            return self._train_epoch_my_algorithm(train_loader)
```

## Configuration System

### Python vs YAML

**Old (YAML):**
```yaml
learned_model:
  n_visible: 10
  n_hidden: 0
training:
  batch_size: 5000
```

**New (Python dataclasses):**
```python
config = BMConfig(
    grbm=GRBMConfig(n_visible=10, n_hidden=0),
    training=TrainingConfig(batch_size=5000)
)
config.validate()  # Type-safe validation
```

**Benefits:**
- Type safety and IDE autocomplete
- Validation at load time
- Programmatic manipulation
- Better error messages
- No YAML syntax issues

### Configuration Hierarchy

1. **Template Config** - `bm_core/config/bm_config_template.py`
   - Defines dataclasses
   - Provides helper functions
   - Documents all options

2. **Project Config** - `projects/my_project/project_config.py`
   - Imports from template
   - Creates BMConfig instance
   - Customizes for use-case

3. **Plugin Configs** - `plugins/*/config.yaml`
   - YAML for plugin-specific settings
   - No sharing between plugins
   - Flexibility for plugin parameters

## Modularity and Coupling

### Loose Coupling

**Core ↔ Sampler Factory:**
- Core receives dict from factory
- Core doesn't know about specific samplers
- Factory can change implementations without affecting core

**Core ↔ Projects:**
- Core defines BMDataset interface
- Projects implement custom dataset
- Core doesn't know about project specifics

**Plugins ↔ Plugins:**
- Plugins are independent
- No direct communication
- Can be added/removed independently

### Dependency Graph

```
projects/my_project/
    ↓ (uses)
bm_core/
    ↓ (uses)
plugins/sampler_factory/
    ↓ (uses)
D-Wave GRBM (external)
```

**Key principle:** Dependencies flow downward, never upward or sideways.

## Testing Strategy

### Unit Tests
- Test individual components in isolation
- Mock dependencies
- Fast execution

### Integration Tests
- Test component interactions
- Use test fixtures
- Verify data flow

### End-to-End Tests
- Test complete workflow
- Create test project
- Run full training pipeline

## Performance Considerations

### GPU Acceleration
- Use `*_gpu` samplers for large models
- Set `use_cuda: true` in config
- Batch processing for efficiency

### Memory Management
- Use PCD for large models (persistent chains)
- Adjust batch_size based on available memory
- Use sparse connectivity for very large models

### Sampling Efficiency
- Gibbs sampling: Best for probability estimation
- GPU samplers: Best for N > 1000
- Exact samplers: Only for N ≤ 20

## Future Extensions

### Planned Features

1. **Full Graph Constructor (v2)**
   - GUI for designing BM topologies
   - Custom connectivity patterns
   - Visual graph editor

2. **Benchmark Plugin**
   - Standalone benchmarking tool
   - Compare sampler performance
   - Metrics: KL divergence, convergence, speed

3. **Data Generator Plugin**
   - Synthetic data generation
   - Ground truth for testing
   - Various distribution types

4. **Additional Backends**
   - Native PyTorch implementation
   - JAX implementation
   - TensorFlow implementation

5. **Advanced Training**
   - Variational methods
   - Score matching
   - Energy-based models

## Design Decisions

### Why Python Config over YAML?
- **Type safety:** Catch errors at config load time
- **IDE support:** Autocomplete, inline docs
- **Validation:** Dataclass validation
- **Flexibility:** Programmatic config generation

### Why BM Abstraction Layer?
- **Future-proofing:** Can swap backends
- **Clean API:** Hide D-Wave specifics
- **Testing:** Easy to mock
- **Flexibility:** Support multiple implementations

### Why Plugin Architecture?
- **Modularity:** Components can evolve independently
- **Reusability:** Plugins can be shared
- **Extensibility:** Easy to add new plugins
- **Isolation:** No coupling between plugins

### Why Project Templates?
- **Standardization:** Consistent structure
- **Simplicity:** Users focus on data logic
- **Scalability:** Easy to manage multiple projects
- **Collaboration:** Team-friendly structure

## Summary

The refactored architecture provides:

1. **Clean separation:** Core, plugins, projects
2. **Type safety:** Python dataclass configs
3. **Extensibility:** Easy to add samplers, models, projects
4. **Modularity:** Components are independent
5. **User-friendly:** Simple project creation and customization

This architecture supports both research experimentation and production deployment, with clear paths for extending functionality without breaking existing code.
