"""
Data Generator Plugin - Synthetic data generation for BM training.

This plugin generates synthetic data by sampling from a true BM model,
useful for testing, validation, and benchmarking.
"""

from .data_generator import SyntheticDataGenerator

__all__ = ['SyntheticDataGenerator']
