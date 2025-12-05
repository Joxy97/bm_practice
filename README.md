# BM Practice - Boltzmann Machine Training Pipeline

A modular, extensible Python package for training and deploying Boltzmann Machines using D-Wave's GRBM implementation with PyTorch.

## Features

- 🎯 **Clean Architecture** - Modular design with clear separation of concerns
- 🔧 **Type-Safe Configuration** - Python dataclasses with IDE autocomplete
- 🚀 **25+ Samplers** - Classical MCMC, GPU-accelerated, exact, and optimization-based
- 📦 **Plugin System** - Self-contained, reusable components
- 🎨 **Project Templates** - Quick project initialization with standardized structure
- ⚡ **GPU Support** - CUDA-accelerated samplers for large-scale models
- 🔬 **Multiple Algorithms** - CD, PCD, various training strategies
- 📊 **Built-in Visualization** - Training curves, parameter evolution

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/bm_practice.git
cd bm_practice
pip install -r requirements.txt
```

### Create Your First Project

```bash
# Create a new project
python -m projects.project_manager create --name my_first_bm

# Prepare your data (CSV with columns v0, v1, ..., vN)
# Place it in: projects/my_first_bm/data/train.csv

# Train
python -m bm_core.bm --mode train \
  --config projects/my_first_bm/project_config.py \
  --dataset projects/my_first_bm/data/train.csv
```

That's it! Your model will be trained and saved to `projects/my_first_bm/outputs/`.

## Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get started in 5 minutes
- **[User Guide](docs/user_guide.md)** - Comprehensive usage documentation  
- **[Architecture Guide](docs/architecture.md)** - System design and patterns
- **[API Reference](docs/api_reference.md)** - Complete API documentation

## Project Structure

```
bm_practice/
├── bm_core/              # Core training pipeline
├── plugins/              # Self-contained plugins
│   └── sampler_factory/ # 25+ sampler implementations
└── projects/             # Your use-cases
    └── [your_projects]/
```

See [Architecture Guide](docs/architecture.md) for complete structure.

## License

MIT License
