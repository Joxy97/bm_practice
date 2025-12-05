"""
BM Core - Main entry point for Boltzmann Machine pipeline.

This module provides the command-line interface for building, training, and
testing Boltzmann Machine models.

Usage:
    python -m bm_core.bm --mode train --config projects/my_project/project_config.py --dataset data/train.csv
    python -m bm_core.bm --mode test --config projects/my_project/project_config.py --checkpoint outputs/best_model.pt
"""

import sys
import os
import argparse
import importlib.util
import traceback
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bm_core.models import BoltzmannMachine, create_dataloaders
from bm_core.trainers import BoltzmannMachineTrainer
from bm_core.utils import create_topology, generate_random_parameters, get_device
from plugins.sampler_factory import SamplerFactory


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_python_config(config_path: str):
    """
    Load Python configuration file.

    Args:
        config_path: Path to project_config.py

    Returns:
        BMConfig instance
    """
    print(f"\nLoading configuration from: {config_path}")

    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")

    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # Get config object (should be named 'config')
    if hasattr(config_module, 'config'):
        config = config_module.config
    else:
        raise AttributeError(
            f"Config file {config_path} must define a 'config' variable"
        )

    # Validate config
    if hasattr(config, 'validate'):
        config.validate()
        print("✓ Configuration validated")

    return config


def build_model(config, sampler_dict: dict, device: torch.device) -> BoltzmannMachine:
    """
    Build BoltzmannMachine from configuration.

    Args:
        config: BMConfig instance
        sampler_dict: Dictionary of available samplers
        device: PyTorch device

    Returns:
        BoltzmannMachine instance
    """
    print(f"\n{'='*70}")
    print("BUILDING BOLTZMANN MACHINE")
    print(f"{'='*70}")

    # Extract GRBM config
    if hasattr(config, 'grbm'):
        grbm_config = config.grbm
    else:
        # Handle dict config
        grbm_config = config.get('learned_model', config.get('grbm', {}))

    # Determine connectivity
    if hasattr(grbm_config, 'sparsity'):
        connectivity = "sparse" if grbm_config.sparsity is not None else "dense"
        connectivity_density = grbm_config.sparsity if grbm_config.sparsity is not None else 0.5
    else:
        connectivity = grbm_config.get('connectivity', 'dense')
        connectivity_density = grbm_config.get('connectivity_density', 0.5)

    # Create topology
    n_visible = grbm_config.n_visible if hasattr(grbm_config, 'n_visible') else grbm_config['n_visible']
    n_hidden = grbm_config.n_hidden if hasattr(grbm_config, 'n_hidden') else grbm_config['n_hidden']
    model_type = grbm_config.model_type if hasattr(grbm_config, 'model_type') else grbm_config['model_type']

    nodes, edges, hidden_nodes = create_topology(
        n_visible=n_visible,
        n_hidden=n_hidden,
        model_type=model_type,
        connectivity=connectivity,
        connectivity_density=connectivity_density,
        seed=config.seed if hasattr(config, 'seed') else config.get('seed', 42)
    )

    print(f"\nTopology:")
    print(f"  Model Type: {model_type.upper()}")
    print(f"  Visible Units: {n_visible}")
    print(f"  Hidden Units: {n_hidden}")
    print(f"  Total Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Connectivity: {connectivity}")

    # Generate initial parameters
    init_linear_scale = grbm_config.init_linear_scale if hasattr(grbm_config, 'init_linear_scale') else grbm_config.get('init_linear_scale', 0.1)
    init_quadratic_scale = grbm_config.init_quadratic_scale if hasattr(grbm_config, 'init_quadratic_scale') else grbm_config.get('init_quadratic_scale', 0.1)

    linear, quadratic = generate_random_parameters(
        nodes,
        edges,
        seed=config.seed if hasattr(config, 'seed') else config.get('seed', 42),
        linear_scale=init_linear_scale,
        quadratic_scale=init_quadratic_scale
    )

    # Create BoltzmannMachine
    model = BoltzmannMachine(
        nodes=nodes,
        edges=edges,
        hidden_nodes=hidden_nodes if n_hidden > 0 else None,
        linear=linear,
        quadratic=quadratic,
        sampler_dict=sampler_dict
    )

    model = model.to(device)

    print(f"\nModel initialized:")
    print(model.summary())

    return model


def train_model(args, config):
    """
    Train model.

    Args:
        args: Command-line arguments
        config: BMConfig instance
    """
    print(f"\n{'='*70}")
    print("TRAINING MODE")
    print(f"{'='*70}")

    # Set seeds
    seed = config.seed if hasattr(config, 'seed') else config.get('seed', 42)
    set_seeds(seed)

    # Get device
    device_config = config.device if hasattr(config, 'device') else config.get('device', {})
    device = get_device(device_config)
    print(f"\nUsing device: {device}")

    # Initialize sampler factory
    print("\nInitializing sampler factory...")
    factory = SamplerFactory()
    sampler_dict = factory.get_sampler_dict()
    print(f"  Registered {len(sampler_dict)} samplers")

    # Build model
    model = build_model(config, sampler_dict, device)

    # Load dataset
    if not args.dataset:
        raise ValueError("--dataset is required for training")

    print(f"\nLoading dataset from: {args.dataset}")

    # Get data config
    if hasattr(config, 'data'):
        data_config = config.data
        train_ratio = data_config.train_ratio
        val_ratio = data_config.val_ratio
        test_ratio = data_config.test_ratio
    else:
        data_config = config.get('data', {})
        train_ratio = data_config.get('train_ratio', 0.7)
        val_ratio = data_config.get('val_ratio', 0.15)
        test_ratio = data_config.get('test_ratio', 0.15)

    # Get training config
    if hasattr(config, 'training'):
        training_config = config.training
        batch_size = training_config.batch_size
    else:
        training_config = config.get('training', {})
        batch_size = training_config.get('batch_size', 5000)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path=args.dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    # Get sampler name
    if hasattr(training_config, 'sampler_name'):
        sampler_name = training_config.sampler_name
    else:
        sampler_config = training_config.get('sampler', {})
        sampler_name = sampler_config.get('type', 'gibbs')

    print(f"\nUsing sampler: {sampler_name}")

    # Create trainer
    # Convert dataclass to dict if needed for trainer
    if hasattr(config, '__dataclass_fields__'):
        training_dict = asdict(training_config) if hasattr(training_config, '__dataclass_fields__') else training_config
        config_dict = {
            'training': training_dict,
            'seed': seed
        }
    else:
        config_dict = config

    trainer = BoltzmannMachineTrainer(
        model=model,
        config=config_dict,
        device=device,
        sampler_name=sampler_name
    )

    # Train
    n_epochs = training_config.n_epochs if hasattr(training_config, 'n_epochs') else training_config.get('n_epochs', 100)

    print(f"\nStarting training for {n_epochs} epochs...")
    print(f"{'='*70}\n")

    trainer.train(train_loader, val_loader, n_epochs=n_epochs)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")


def test_model(args, config):
    """
    Test model.

    Args:
        args: Command-line arguments
        config: BMConfig instance
    """
    print(f"\n{'='*70}")
    print("TEST MODE")
    print(f"{'='*70}")

    if not args.checkpoint:
        raise ValueError("--checkpoint is required for testing")

    if not args.dataset:
        raise ValueError("--dataset is required for testing")

    # Set seeds
    seed = config.seed if hasattr(config, 'seed') else config.get('seed', 42)
    set_seeds(seed)

    # Get device
    device_config = config.device if hasattr(config, 'device') else config.get('device', {})
    device = get_device(device_config)
    print(f"\nUsing device: {device}")

    # Initialize sampler factory
    factory = SamplerFactory()
    sampler_dict = factory.get_sampler_dict()

    # Load model from checkpoint
    print(f"\nLoading model from: {args.checkpoint}")
    model = BoltzmannMachine.load_checkpoint(
        args.checkpoint,
        sampler_dict=sampler_dict,
        device=device
    )

    print("✓ Model loaded")
    print(model.summary())

    # Load test dataset
    print(f"\nLoading test dataset from: {args.dataset}")

    # For testing, we can use the full dataset
    from bm_core.models.dataset import BMDataset
    test_dataset = BMDataset(args.dataset)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    print(f"  Test samples: {len(test_dataset)}")

    # Test (would need trainer's test method, for now just load)
    print("\n✓ Model ready for testing")
    print("  (Full test implementation requires trainer.test() method)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BM Core - Boltzmann Machine Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python -m bm_core.bm --mode train --config projects/my_project/project_config.py --dataset data/train.csv

  # Test a model
  python -m bm_core.bm --mode test --config projects/my_project/project_config.py --checkpoint outputs/best_model.pt --dataset data/test.csv

  # Build a model (initialize only)
  python -m bm_core.bm --mode build --config projects/my_project/project_config.py
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['build', 'train', 'test'],
        help='Operation mode: build (initialize model), train, or test'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to project_config.py (Python configuration file)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset CSV file'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (for test mode)'
    )

    args = parser.parse_args()

    try:
        # Load config
        config = load_python_config(args.config)

        # Execute mode
        if args.mode == 'build':
            # Just build and display model
            seed = config.seed if hasattr(config, 'seed') else config.get('seed', 42)
            set_seeds(seed)
            device_config = config.device if hasattr(config, 'device') else config.get('device', {})
            device = get_device(device_config)
            factory = SamplerFactory()
            sampler_dict = factory.get_sampler_dict()
            model = build_model(config, sampler_dict, device)
            print("\n✓ Model built successfully")

        elif args.mode == 'train':
            train_model(args, config)

        elif args.mode == 'test':
            test_model(args, config)

    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR")
        print(f"{'='*70}")
        print(f"\n{str(e)}\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
