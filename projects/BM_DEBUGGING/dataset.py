"""
Custom Dataset Implementation

This module defines a custom PyTorch Dataset class for your project.
Extend BMDataset and override load_data() to implement custom preprocessing
and data loading logic for your CSV files.

The BM training pipeline uses this dataset with PyTorch DataLoader for
batching and efficient data loading.

Dependencies:
    - bm_core.models.dataset.BMDataset: Base dataset class with PyTorch integration
    - This provides __len__, __getitem__, and DataLoader compatibility
    - Training pipeline uses create_dataloaders() which handles train/val/test splits
"""

import pandas as pd
import numpy as np
from bm_core.models.dataset import BMDataset


class CustomDataset(BMDataset):
    """
    Custom dataset for this project.

    Extends BMDataset to provide custom data loading and preprocessing.
    The base class (BMDataset) is a PyTorch Dataset that:
    - Implements __len__ and __getitem__ for DataLoader compatibility
    - Returns torch.Tensor samples for batch processing
    - Handles train/val/test splitting via create_dataloaders()

    Usage in training:
        from bm_core.models.dataset import create_dataloaders
        from projects.your_project.dataset import CustomDataset

        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_path="data/your_data.csv",
            dataset_class=CustomDataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=128
        )
    """

    def load_data(self, csv_path: str) -> np.ndarray:
        """
        Load and preprocess data from CSV file.

        This method is called by BMDataset.__init__ to load your data.
        Override this to implement custom preprocessing logic.

        Args:
            csv_path: Path to CSV file

        Returns:
            Numpy array of shape (n_samples, n_visible) with dtype float32
            - Each row is one sample
            - Each column is one visible unit (binary/continuous value)

        Example CSV formats:
            1. Visible units as columns (default):
               v0,v1,v2,v3,...
               1,0,1,0,...
               0,1,0,1,...

            2. Custom column names:
               feature_1,feature_2,...,label
               0.5,0.8,...,1
               0.3,0.6,...,0

            3. Multiple data sources (combine multiple CSVs):
               You can load and concatenate data from multiple files here
        """
        # Load CSV
        df = pd.read_csv(csv_path)

        # =================================================================
        # CUSTOMIZE THIS SECTION FOR YOUR DATA FORMAT
        # =================================================================

        # Option 1: Default - Extract columns starting with 'v' (visible units)
        # This matches the default BM data format: v0, v1, v2, ...
        visible_cols = [col for col in df.columns if col.startswith('v')]

        if not visible_cols:
            # Option 2: Use all numeric columns (if no 'v' columns found)
            visible_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            # Option 3: Specify exact column names for your data
            # visible_cols = ['feature_1', 'feature_2', 'feature_3', ...]

            # Option 4: Exclude certain columns (e.g., labels, metadata)
            # visible_cols = [col for col in df.columns if col not in ['label', 'id', 'timestamp']]

            if not visible_cols:
                raise ValueError(
                    f"No valid data columns found in {csv_path}. "
                    f"Available columns: {df.columns.tolist()}"
                )

        # Extract data
        data = df[visible_cols].values

        # =================================================================
        # PREPROCESSING (CUSTOMIZE AS NEEDED)
        # =================================================================

        # Example 1: Normalize to [0, 1] range
        # data_min = data.min(axis=0)
        # data_max = data.max(axis=0)
        # data_range = data_max - data_min
        # data_range[data_range == 0] = 1.0  # Avoid division by zero
        # data = (data - data_min) / data_range

        # Example 2: Standardize (zero mean, unit variance)
        # data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

        # Example 3: Binarize (threshold at 0.5)
        # data = (data > 0.5).astype(np.float32)

        # Example 4: Clip to valid range
        # data = np.clip(data, -1, 1)

        # Example 5: Handle missing values
        # data = np.nan_to_num(data, nan=0.0)

        # =================================================================
        # RETURN AS FLOAT32 (REQUIRED)
        # =================================================================

        # Convert to float32 for PyTorch compatibility
        data = data.astype(np.float32)

        print(f"Loaded dataset from {csv_path}")
        print(f"  Shape: {data.shape}")
        print(f"  Columns used: {visible_cols}")
        print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")

        return data


# =============================================================================
# Helper Functions (Optional)
# =============================================================================

def load_multiple_csvs(csv_paths: list[str]) -> np.ndarray:
    """
    Load and concatenate data from multiple CSV files.

    Useful if your dataset is split across multiple files.

    Args:
        csv_paths: List of paths to CSV files

    Returns:
        Combined numpy array with shape (total_samples, n_visible)

    Example:
        data = load_multiple_csvs([
            'data/batch1.csv',
            'data/batch2.csv',
            'data/batch3.csv'
        ])
    """
    datasets = []
    for path in csv_paths:
        df = pd.read_csv(path)
        # Apply consistent column selection
        visible_cols = [col for col in df.columns if col.startswith('v')]
        datasets.append(df[visible_cols].values)

    combined = np.vstack(datasets).astype(np.float32)
    return combined


def apply_feature_engineering(data: np.ndarray) -> np.ndarray:
    """
    Apply feature engineering transformations.

    Args:
        data: Raw data array

    Returns:
        Transformed data array

    Example transformations:
        - Polynomial features
        - Interaction terms
        - Log transforms
        - PCA dimensionality reduction
    """
    # Example: Add squared features
    # squared_features = data ** 2
    # data = np.hstack([data, squared_features])

    # Example: Add interaction terms (product of pairs)
    # interactions = data[:, :-1] * data[:, 1:]
    # data = np.hstack([data, interactions])

    return data
