"""
Custom Dataset Implementation Template

Implement your custom data loading logic here.

The BMDataset base class expects you to override the load_data() method
to handle your specific CSV format and preprocessing needs.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bm_core.models import BMDataset


class MyCustomDataset(BMDataset):
    """
    Custom dataset for [YOUR PROJECT NAME].

    TODO: Implement your data loading and preprocessing logic.
    """

    def load_data(self, csv_path: str) -> np.ndarray:
        """
        Load and preprocess data from CSV.

        Args:
            csv_path: Path to CSV file

        Returns:
            Numpy array of shape (n_samples, n_visible) with float32 dtype
        """
        # TODO: Implement your custom loading logic

        # Example: Default implementation (extract columns starting with 'v')
        df = pd.read_csv(csv_path)

        # Extract visible columns
        visible_cols = [col for col in df.columns if col.startswith('v')]

        if not visible_cols:
            raise ValueError(
                f"No visible columns found in {csv_path}. "
                f"Expected columns starting with 'v'."
            )

        # TODO: Add your custom preprocessing here
        # Examples:
        # - Normalization: data = (data - data.mean()) / data.std()
        # - Filtering: data = data[data['some_col'] > threshold]
        # - Feature engineering: data['new_feature'] = data['col1'] * data['col2']

        data = df[visible_cols].values.astype(np.float32)

        return data


# Example usage:
if __name__ == '__main__':
    # Test your dataset implementation
    dataset = MyCustomDataset('data/test.csv')
    print(f"Loaded {len(dataset)} samples")
    print(f"Visible units: {dataset.get_n_visible()}")
    print(f"First sample: {dataset[0]}")
