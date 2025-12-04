"""
Run management utilities for organizing outputs.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import yaml


def create_run_directory(config: Dict[str, Any], base_output_dir: str = "outputs") -> Dict[str, str]:
    """
    Create a timestamped run directory structure.

    Directory structure:
        outputs/
            {dataset_name}_{timestamp}/
                data/
                models/
                checkpoints/
                plots/
                config.yaml

    Args:
        config: Configuration dictionary
        base_output_dir: Base output directory (default: "outputs")

    Returns:
        Dictionary with paths to all subdirectories
    """
    # Get dataset name from config
    dataset_name = config.get('data', {}).get('dataset_name', 'bm_run')

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create run directory name
    run_name = f"{dataset_name}_{timestamp}"
    run_dir = os.path.join(base_output_dir, run_name)

    # Create directory structure
    paths = {
        'run_dir': run_dir,
        'data_dir': os.path.join(run_dir, 'data'),
        'model_dir': os.path.join(run_dir, 'models'),
        'checkpoint_dir': os.path.join(run_dir, 'checkpoints'),
        'plot_dir': os.path.join(run_dir, 'plots'),
        'log_dir': os.path.join(run_dir, 'logs')
    }

    # Create all directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    # Save config to run directory
    config_copy_path = os.path.join(run_dir, 'config.yaml')
    save_config(config, config_copy_path)

    # Print run information
    print(f"\n{'='*70}")
    print("RUN DIRECTORY CREATED")
    print(f"{'='*70}")
    print(f"Run name:    {run_name}")
    print(f"Location:    {run_dir}")
    print(f"Config saved: {config_copy_path}")
    print(f"{'='*70}\n")

    return paths


def save_config(config: Dict[str, Any], filepath: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        filepath: Path to save config file
    """
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def update_config_paths(config: Dict[str, Any], run_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Update config with run-specific paths.

    Args:
        config: Configuration dictionary
        run_paths: Dictionary of run paths from create_run_directory()

    Returns:
        Updated configuration dictionary
    """
    # Update paths in config
    config['paths'] = {
        'run_dir': run_paths['run_dir'],
        'data_dir': run_paths['data_dir'],
        'model_dir': run_paths['model_dir'],
        'log_dir': run_paths['log_dir']
    }

    # Update training checkpoint directory
    config['training']['checkpoint_dir'] = run_paths['checkpoint_dir']

    # Update logging plot directory
    if 'logging' not in config:
        config['logging'] = {}
    config['logging']['plot_dir'] = run_paths['plot_dir']

    return config


def list_runs(base_output_dir: str = "outputs", dataset_name: str = None) -> list:
    """
    List all runs in the output directory.

    Args:
        base_output_dir: Base output directory
        dataset_name: Filter by dataset name (optional)

    Returns:
        List of run directory names
    """
    if not os.path.exists(base_output_dir):
        return []

    runs = []
    for item in os.listdir(base_output_dir):
        item_path = os.path.join(base_output_dir, item)
        if os.path.isdir(item_path):
            # Check if it matches the pattern
            if dataset_name is None or item.startswith(dataset_name):
                runs.append(item)

    # Sort by timestamp (newest first)
    runs.sort(reverse=True)
    return runs


def get_latest_run(base_output_dir: str = "outputs", dataset_name: str = None) -> str:
    """
    Get the most recent run directory.

    Args:
        base_output_dir: Base output directory
        dataset_name: Filter by dataset name (optional)

    Returns:
        Path to latest run directory or None
    """
    runs = list_runs(base_output_dir, dataset_name)
    if runs:
        return os.path.join(base_output_dir, runs[0])
    return None


def print_run_summary(run_dir: str):
    """
    Print summary of a run directory.

    Args:
        run_dir: Path to run directory
    """
    if not os.path.exists(run_dir):
        print(f"Run directory not found: {run_dir}")
        return

    print(f"\n{'='*70}")
    print(f"RUN SUMMARY: {os.path.basename(run_dir)}")
    print(f"{'='*70}")

    # Check for config
    config_path = os.path.join(run_dir, 'config.yaml')
    if os.path.exists(config_path):
        print(f"Config:      [OK] {config_path}")
    else:
        print(f"Config:      [MISSING]")

    # Check subdirectories
    subdirs = ['data', 'models', 'checkpoints', 'plots', 'logs']
    for subdir in subdirs:
        subdir_path = os.path.join(run_dir, subdir)
        if os.path.exists(subdir_path):
            num_files = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
            print(f"{subdir.capitalize():12s} [{num_files:3d} files]")
        else:
            print(f"{subdir.capitalize():12s} [MISSING]")

    print(f"{'='*70}\n")
