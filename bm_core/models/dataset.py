"""
PyTorch Dataset and DataLoader for Boltzmann Machine training.

This module provides a base Dataset class that users extend for their specific
use-cases. Users implement the load_data() method to handle their custom CSV format.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


class BMDataset(Dataset):
    """
    Base PyTorch Dataset for Boltzmann Machine samples.

    Users should extend this class and override the load_data() method
    to implement custom data loading logic for their use-case.

    Example:
        class MyDataset(BMDataset):
            def load_data(self, csv_path: str):
                df = pd.read_csv(csv_path)
                # Custom preprocessing
                data = df[[col for col in df.columns if col.startswith('v')]].values
                return data.astype(np.float32)
    """

    def __init__(self, csv_path: str, **kwargs):
        """
        Initialize the dataset.

        Args:
            csv_path: Path to CSV file
            **kwargs: Additional arguments for custom load_data implementations
        """
        self.csv_path = csv_path
        self.kwargs = kwargs
        self.data = self.load_data(csv_path)

        if self.data is None:
            raise ValueError("load_data() returned None. Must return numpy array.")

        if not isinstance(self.data, np.ndarray):
            raise TypeError(
                f"load_data() must return numpy array, got {type(self.data)}"
            )

    def load_data(self, csv_path: str) -> np.ndarray:
        """
        Load data from CSV file.

        Users should override this method for custom loading logic.

        Default implementation:
        - Loads CSV with pandas
        - Extracts columns starting with 'v' (visible units)
        - Returns as float32 numpy array

        Args:
            csv_path: Path to CSV file

        Returns:
            Numpy array of shape (n_samples, n_visible)
        """
        df = pd.read_csv(csv_path)

        # Extract visible columns (columns starting with 'v')
        visible_cols = [col for col in df.columns if col.startswith('v')]

        if not visible_cols:
            raise ValueError(
                f"No visible columns found in {csv_path}. "
                f"Expected columns starting with 'v'."
            )

        data = df[visible_cols].values.astype(np.float32)
        return data

    def __len__(self) -> int:
        """Return number of samples."""
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

    def get_n_visible(self) -> int:
        """Return number of visible units."""
        return self.data.shape[1]

    def get_data_numpy(self) -> np.ndarray:
        """Return data as numpy array."""
        return self.data


def create_dataloaders(
    dataset_path: str,
    dataset_class: type = BMDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders from a CSV file.

    Args:
        dataset_path: Path to the CSV file
        dataset_class: Dataset class to use (must inherit from BMDataset)
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        batch_size: Batch size for dataloaders
        shuffle: Whether to shuffle training data
        seed: Random seed for reproducibility
        num_workers: Number of workers for dataloaders
        **dataset_kwargs: Additional arguments passed to dataset_class

    Returns:
        train_loader, val_loader, test_loader
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
        raise ValueError(
            f"Train, val, and test ratios must sum to 1.0, got {total_ratio}"
        )

    # Load full dataset to get size
    df = pd.read_csv(dataset_path)
    n_samples = len(df)

    print(f"\nLoading dataset from: {dataset_path}")
    print(f"  Total samples: {n_samples}")

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

    # Create temporary split CSV files (or use indices)
    # For simplicity, we'll create split datasets directly
    train_data = df.iloc[train_indices].reset_index(drop=True)
    val_data = df.iloc[val_indices].reset_index(drop=True)
    test_data = df.iloc[test_indices].reset_index(drop=True)

    # Save temporary files
    import tempfile
    import os
    temp_dir = tempfile.mkdtemp()
    train_path = os.path.join(temp_dir, 'train.csv')
    val_path = os.path.join(temp_dir, 'val.csv')
    test_path = os.path.join(temp_dir, 'test.csv')

    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)

    # Create datasets using custom dataset class
    train_dataset = dataset_class(train_path, **dataset_kwargs)
    val_dataset = dataset_class(val_path, **dataset_kwargs)
    test_dataset = dataset_class(test_path, **dataset_kwargs)

    print(f"  Visible units: {train_dataset.get_n_visible()}")

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

    # Clean up temp files
    import shutil
    shutil.rmtree(temp_dir)

    return train_loader, val_loader, test_loader


def load_full_dataset(
    dataset_path: str,
    dataset_class: type = BMDataset,
    **dataset_kwargs
) -> BMDataset:
    """
    Load the full dataset without splitting.

    Args:
        dataset_path: Path to the CSV file
        dataset_class: Dataset class to use
        **dataset_kwargs: Additional arguments passed to dataset_class

    Returns:
        Dataset instance containing all samples
    """
    return dataset_class(dataset_path, **dataset_kwargs)


# Backward compatibility: Legacy class name
class BoltzmannMachineDataset(BMDataset):
    """
    Legacy name for BMDataset.

    Kept for backward compatibility. Use BMDataset instead.
    """

    def __init__(self, data: pd.DataFrame, visible_columns: Optional[list] = None):
        """
        Initialize with DataFrame (legacy interface).

        Args:
            data: DataFrame containing samples
            visible_columns: List of column names for visible units
        """
        if visible_columns is None:
            visible_columns = [col for col in data.columns if col.startswith('v')]

        self.data = data[visible_columns].values.astype(np.float32)
        self.visible_columns = visible_columns
