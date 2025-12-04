"""
Main pipeline script for Boltzmann Machine training.

Usage:
    python main.py --mode [generate|train|full] --config configs/config.yaml
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine as GRBM

from models import DataGenerator, create_dataloaders
from trainers import BoltzmannMachineTrainer
from utils import (
    load_config,
    create_topology,
    generate_random_parameters,
    plot_model_parameters,
    plot_training_history,
    plot_model_comparison,
    visualize_topology_from_config,
    create_sampler,
    get_device,
    print_device_info,
    set_device_seeds,
    create_run_directory,
    update_config_paths
)


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_data(config: dict):
    """
    Generate training data from true model.

    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*70)
    print("STEP 1: DATA GENERATION")
    print("="*70)

    # Set seeds
    set_seeds(config['seed'])

    # Create data generator
    data_gen = DataGenerator(config)

    # Generate and save data
    data_dir = config.get('paths', {}).get('data_dir', 'data')
    df = data_gen.generate(save_dir=data_dir)

    # Visualize true model
    true_model = data_gen.get_true_model()
    plot_dir = config.get('logging', {}).get('plot_dir', 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    plot_model_parameters(
        true_model,
        title="TRUE Boltzmann Machine",
        save_path=os.path.join(plot_dir, "true_model_parameters.png")
    )

    # Visualize true model graph topology
    visualize_topology_from_config(
        config,
        model_key='true_model',
        save_path=os.path.join(plot_dir, "true_model_graph.png"),
        show=False
    )

    print(f"\n[OK] Data generation complete!")
    print(f"  Dataset: {os.path.join(data_dir, config['data']['dataset_name'])}.csv")
    print(f"  Samples: {len(df)}")
    print(f"  Graph visualization: {os.path.join(plot_dir, 'true_model_graph.png')}")


def train_model(config: dict, dataset_path: str = None, true_model=None):
    """
    Train a Boltzmann Machine.

    Args:
        config: Configuration dictionary
        dataset_path: Path to dataset CSV (optional, will be inferred from config)
        true_model: Optional true model for parameter comparison in testing
    """
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)

    # Initialize device
    device = get_device(config)
    print_device_info(device)

    # Set seeds
    set_device_seeds(config['seed'], device)

    # Determine dataset path
    if dataset_path is None:
        data_dir = config.get('paths', {}).get('data_dir', 'data')
        dataset_name = config['data']['dataset_name']
        dataset_path = os.path.join(data_dir, f"{dataset_name}.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path=dataset_path,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        batch_size=config['training']['batch_size'],
        shuffle=True,
        seed=config['seed']
    )

    # Create learned model topology
    learned_config = config['learned_model']
    n_visible = learned_config['n_visible']
    n_hidden = learned_config['n_hidden']
    model_type = learned_config['model_type']
    connectivity = learned_config['connectivity']
    connectivity_density = learned_config.get('connectivity_density', 0.5)

    nodes, edges, hidden_nodes = create_topology(
        n_visible=n_visible,
        n_hidden=n_hidden,
        model_type=model_type,
        connectivity=connectivity,
        connectivity_density=connectivity_density,
        seed=config['seed']
    )

    print(f"\nLearned model topology:")
    print(f"  Model Type: {model_type.upper()}")
    print(f"  Connectivity: {connectivity}")
    if connectivity == "sparse":
        print(f"  Connectivity Density: {connectivity_density:.1%}")
    print(f"  Nodes: {len(nodes)} ({n_visible} visible, {n_hidden} hidden)")
    print(f"  Edges: {len(edges)}")

    # Initialize learned model with random parameters
    linear, quadratic = generate_random_parameters(
        nodes,
        edges,
        seed=config['seed'] + 1000,  # Different seed
        linear_scale=learned_config['init_linear_scale'],
        quadratic_scale=learned_config['init_quadratic_scale']
    )

    learned_model = GRBM(
        nodes=nodes,
        edges=edges,
        hidden_nodes=hidden_nodes if learned_config['n_hidden'] > 0 else None,
        linear=linear,
        quadratic=quadratic
    )

    print(f"\nLearned model initialized:")
    print(f"  Linear bias range: [{learned_model.linear.min():.3f}, {learned_model.linear.max():.3f}]")
    print(f"  Quadratic weight range: [{learned_model.quadratic.min():.3f}, {learned_model.quadratic.max():.3f}]")

    # Create sampler from config
    sampler_config = config['training'].get('sampler', {})
    sampler_type = sampler_config.get('type', 'simulated_annealing')
    sampler_params = sampler_config.get('params', {})

    print(f"\nTraining sampler configuration:")
    print(f"  Type: {sampler_type}")
    sampler = create_sampler(sampler_type, sampler_params)
    print(f"  Created: {sampler.__class__.__name__}")

    # Create trainer
    trainer = BoltzmannMachineTrainer(learned_model, config, device, sampler)

    # Train
    trainer.train(train_loader, val_loader, verbose=True)

    # Test (with true model if provided for parameter comparison)
    test_metrics = trainer.test(test_loader, true_model=true_model)

    # Print test results
    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
    print(f"Test Loss:     {test_metrics['test_loss']:.4f} +/- {test_metrics['test_loss_std']:.4f}")
    if 'parameter_mae' in test_metrics:
        print(f"\nParameter Comparison (vs True Model):")
        print(f"  Overall MAE:    {test_metrics['parameter_mae']:.4f}")
        print(f"  Linear MAE:     {test_metrics['linear_mae']:.4f}")
        print(f"  Quadratic MAE:  {test_metrics['quadratic_mae']:.4f}")
    print(f"{'='*70}\n")

    # Save test results to file
    log_dir = config.get('paths', {}).get('log_dir', 'outputs/logs')
    test_results_path = os.path.join(log_dir, 'test_results.json')
    trainer.save_test_results(test_metrics, test_results_path)
    print(f"[OK] Test results saved to: {test_results_path}")

    # Save final model
    model_dir = config.get('paths', {}).get('model_dir', 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    final_model_path = os.path.join(model_dir, 'final_model.pt')
    trainer.save_checkpoint(final_model_path)
    print(f"[OK] Final model saved to: {final_model_path}")

    # Visualizations
    plot_dir = config.get('logging', {}).get('plot_dir', 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Plot learned model
    plot_model_parameters(
        learned_model,
        title="LEARNED Boltzmann Machine",
        save_path=os.path.join(plot_dir, "learned_model_parameters.png")
    )

    # Visualize learned model graph topology
    visualize_topology_from_config(
        config,
        model_key='learned_model',
        save_path=os.path.join(plot_dir, "learned_model_graph.png"),
        show=False
    )

    # Plot training history
    plot_training_history(
        trainer.get_history(),
        save_path=os.path.join(plot_dir, "training_history.png")
    )

    print(f"\n[OK] Model training complete!")
    print(f"  Graph visualization: {os.path.join(plot_dir, 'learned_model_graph.png')}")

    return learned_model, trainer


def test_model(config: dict, checkpoint_path: str, dataset_path: str = None):
    """
    Test a trained model on a dataset.

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
        dataset_path: Path to test dataset (optional)
    """
    print("\n" + "="*70)
    print("MODEL TESTING")
    print("="*70)

    # Initialize device
    device = get_device(config)
    set_device_seeds(config['seed'], device)

    # Determine dataset path
    if dataset_path is None:
        data_dir = config.get('paths', {}).get('data_dir', 'outputs/data')
        dataset_name = config['data']['dataset_name']
        dataset_path = os.path.join(data_dir, f"{dataset_name}.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"\nDataset: {dataset_path}")

    # Create dataloader (use all data as test set for this mode)
    from models import BoltzmannMachineDataset
    from torch.utils.data import DataLoader

    test_dataset = BoltzmannMachineDataset(dataset_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    print(f"Test samples: {len(test_dataset)}")

    # Load model topology from config
    learned_config = config['learned_model']
    nodes, edges, hidden_nodes = create_topology(
        n_visible=learned_config['n_visible'],
        n_hidden=learned_config['n_hidden'],
        model_type=learned_config['model_type'],
        connectivity=learned_config['connectivity'],
        connectivity_density=learned_config.get('connectivity_density', 0.5),
        seed=config['seed']
    )

    # Initialize model
    linear, quadratic = generate_random_parameters(nodes, edges, seed=config['seed'])
    model = GRBM(
        nodes=nodes,
        edges=edges,
        hidden_nodes=hidden_nodes if learned_config['n_hidden'] > 0 else None,
        linear=linear,
        quadratic=quadratic
    )

    # Create sampler from config
    sampler_config = config['training'].get('sampler', {})
    sampler_type = sampler_config.get('type', 'simulated_annealing')
    sampler_params = sampler_config.get('params', {})
    sampler = create_sampler(sampler_type, sampler_params)

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    trainer = BoltzmannMachineTrainer(model, config, device, sampler)
    trainer.load_checkpoint(checkpoint_path)
    print("[OK] Checkpoint loaded")

    # Test
    test_metrics = trainer.test(test_loader)

    # Print results
    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
    print(f"Test Loss:     {test_metrics['test_loss']:.4f} +/- {test_metrics['test_loss_std']:.4f}")
    print(f"{'='*70}\n")

    # Save results
    log_dir = config.get('paths', {}).get('log_dir', 'outputs/logs')
    test_results_path = os.path.join(log_dir, 'test_results.json')
    trainer.save_test_results(test_metrics, test_results_path)
    print(f"[OK] Test results saved to: {test_results_path}")


def compare_models(config: dict, learned_model: GRBM):
    """
    Compare learned model with true model.

    Args:
        config: Configuration dictionary
        learned_model: Trained GRBM model
    """
    print("\n" + "="*70)
    print("STEP 3: MODEL COMPARISON")
    print("="*70)

    # Recreate true model
    set_seeds(config['seed'])
    data_gen = DataGenerator(config)
    true_model = data_gen.get_true_model()

    # Plot comparison
    plot_dir = config.get('logging', {}).get('plot_dir', 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    plot_model_comparison(
        true_model,
        learned_model,
        save_path=os.path.join(plot_dir, "model_comparison.png")
    )

    print(f"\n[OK] Model comparison complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Boltzmann Machine Training Pipeline")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['generate', 'train', 'test', 'full'],
        default='full',
        help='Pipeline mode: generate data, train model, test model, or run full pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to dataset (for train/test modes)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (for test mode)'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    print("\n" + "="*70)
    print("BOLTZMANN MACHINE TRAINING PIPELINE")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Seed: {config['seed']}")
    print("="*70)

    try:
        # Create run directory with timestamp
        run_paths = create_run_directory(config)

        # Update config with run-specific paths
        config = update_config_paths(config, run_paths)

        if args.mode == 'generate':
            # Only generate data
            generate_data(config)

        elif args.mode == 'train':
            # Only train model (requires existing dataset)
            learned_model, trainer = train_model(config, args.dataset)

        elif args.mode == 'test':
            # Test mode: load checkpoint and test on dataset
            test_model(config, args.checkpoint, args.dataset)

        elif args.mode == 'full':
            # Full pipeline: generate + train + compare
            # Generate data and get true model for comparison
            generate_data(config)

            # Get true model for parameter comparison in testing
            set_seeds(config['seed'])
            data_gen = DataGenerator(config)
            true_model = data_gen.get_true_model()

            # Train with true model for enhanced testing
            learned_model, trainer = train_model(config, true_model=true_model)
            compare_models(config, learned_model)

        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        print(f"\nRun directory: {run_paths['run_dir']}")
        print(f"\nGenerated files:")
        print(f"  Config:      {os.path.join(run_paths['run_dir'], 'config.yaml')}")
        print(f"  Plots:       {run_paths['plot_dir']}/")
        if args.mode in ['train', 'full']:
            print(f"  Models:      {run_paths['model_dir']}/")
            print(f"  Checkpoints: {run_paths['checkpoint_dir']}/")
        if args.mode in ['generate', 'full']:
            print(f"  Data:        {run_paths['data_dir']}/")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
