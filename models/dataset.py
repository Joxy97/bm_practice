"""
PyTorch Dataset and DataLoader for Boltzmann Machine training.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
from pathlib import Path


class BoltzmannMachineDataset(Dataset):
    """
    PyTorch Dataset for Boltzmann Machine samples.
    """

    def __init__(self, data: pd.DataFrame, visible_columns: Optional[list] = None):
        """
        Initialize the dataset.

        Args:
            data: DataFrame containing samples
            visible_columns: List of column names for visible units (default: all 'v*' columns)
        """
        if visible_columns is None:
            visible_columns = [col for col in data.columns if col.startswith('v')]

        self.data = data[visible_columns].values.astype(np.float32)
        self.visible_columns = visible_columns

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sample.

        Args:
            idx: Index of the sample

        Returns:
            Tensor of shape (n_visible,)
        """
        sample = torch.from_numpy(self.data[idx])
        return sample


def create_dataloaders(
    dataset_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders from a CSV file.

    Args:
        dataset_path: Path to the CSV file
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        batch_size: Batch size for dataloaders
        shuffle: Whether to shuffle training data
        seed: Random seed for reproducibility
        num_workers: Number of workers for dataloaders

    Returns:
        train_loader, val_loader, test_loader
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Train, val, and test ratios must sum to 1.0"

    # Load data
    df = pd.read_csv(dataset_path)

    # Get visible columns
    visible_columns = [col for col in df.columns if col.startswith('v')]
    n_samples = len(df)

    print(f"\nLoading dataset from: {dataset_path}")
    print(f"  Total samples: {n_samples}")
    print(f"  Visible units: {len(visible_columns)}")

    # Set random seed for reproducibility
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)

    # Calculate split sizes
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    print(f"\nDataset split:")
    print(f"  Train: {len(train_indices)} ({train_ratio:.1%})")
    print(f"  Val:   {len(val_indices)} ({val_ratio:.1%})")
    print(f"  Test:  {len(test_indices)} ({test_ratio:.1%})")

    # Create datasets
    train_data = df.iloc[train_indices].reset_index(drop=True)
    val_data = df.iloc[val_indices].reset_index(drop=True)
    test_data = df.iloc[test_indices].reset_index(drop=True)

    train_dataset = BoltzmannMachineDataset(train_data, visible_columns)
    val_dataset = BoltzmannMachineDataset(val_data, visible_columns)
    test_dataset = BoltzmannMachineDataset(test_data, visible_columns)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print(f"  Batch size:    {batch_size}")

    return train_loader, val_loader, test_loader


def load_full_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load the full dataset without splitting.

    Args:
        dataset_path: Path to the CSV file

    Returns:
        DataFrame containing all samples
    """
    return pd.read_csv(dataset_path)
