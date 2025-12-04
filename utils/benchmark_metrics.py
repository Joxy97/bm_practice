"""
Benchmark Metrics Module

Provides modular distance/divergence metrics for comparing probability distributions
and sample sets from Boltzmann Machine samplers.
"""

import numpy as np
import torch
from typing import Dict, Tuple, List, Callable
from collections import Counter


class BenchmarkMetrics:
    """
    Computes various distance and divergence metrics between true and empirical distributions.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize metrics computer.

        Args:
            config: Configuration dictionary with metrics settings
        """
        self.config = config or {}
        self.metrics_config = self.config.get('benchmark', {}).get('metrics', {})

        # Get enabled metrics (default: all enabled)
        self.enabled_metrics = self.metrics_config.get('enabled', [
            'kl_divergence',
            'reverse_kl_divergence',
            'total_variation',
            'mmd',
            'moment_matching'
        ])

        # MMD kernel configuration
        self.mmd_kernel = self.metrics_config.get('mmd_kernel', 'rbf_hamming')
        self.mmd_kernel_bandwidth = self.metrics_config.get('mmd_kernel_bandwidth', 1.0)

    def compute_all_metrics(
        self,
        p_true: Dict[Tuple, float],
        p_empirical: Dict[Tuple, float],
        samples: np.ndarray,
        true_samples: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute all enabled metrics.

        Args:
            p_true: True distribution p(x) as dict mapping states to probabilities
            p_empirical: Empirical distribution p-hat(x) from samples
            samples: Empirical samples, shape (n_samples, n_variables)
            true_samples: Optional samples from true distribution for MMD

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        if 'kl_divergence' in self.enabled_metrics:
            metrics['kl_divergence'] = self.kl_divergence(p_true, p_empirical)

        if 'reverse_kl_divergence' in self.enabled_metrics:
            metrics['reverse_kl_divergence'] = self.reverse_kl_divergence(p_true, p_empirical)

        if 'total_variation' in self.enabled_metrics:
            metrics['total_variation'] = self.total_variation_distance(p_true, p_empirical)

        if 'mmd' in self.enabled_metrics:
            # Generate true samples if not provided
            if true_samples is None:
                true_samples = self._sample_from_distribution(p_true, len(samples))
            metrics['mmd'] = self.maximum_mean_discrepancy(samples, true_samples)

        if 'moment_matching' in self.enabled_metrics:
            # Generate true samples if not provided
            if true_samples is None:
                true_samples = self._sample_from_distribution(p_true, len(samples))
            metrics['moment_matching'] = self.moment_matching_error(samples, true_samples)

        return metrics

    def kl_divergence(
        self,
        p_true: Dict[Tuple, float],
        p_empirical: Dict[Tuple, float]
    ) -> float:
        """
        Compute forward KL divergence: D_KL(p_true || p_empirical)

        KL(p || q) = sum_x p(x) log(p(x) / q(x))

        Args:
            p_true: True distribution p(x)
            p_empirical: Empirical distribution q(x)

        Returns:
            KL divergence value (lower is better, 0 is perfect)
        """
        kl_div = 0.0

        for state, p_x in p_true.items():
            if p_x > 1e-15:  # Avoid log(0)
                p_hat_x = p_empirical.get(state, 1e-15)  # Smoothing for unseen states
                kl_div += p_x * np.log(p_x / p_hat_x)

        return kl_div

    def reverse_kl_divergence(
        self,
        p_true: Dict[Tuple, float],
        p_empirical: Dict[Tuple, float]
    ) -> float:
        """
        Compute reverse KL divergence: D_KL(p_empirical || p_true)

        Reverse KL(q || p) = sum_x q(x) log(q(x) / p(x))

        This penalizes the empirical distribution for placing mass where
        the true distribution has low mass (mode-seeking behavior).

        Args:
            p_true: True distribution p(x)
            p_empirical: Empirical distribution q(x)

        Returns:
            Reverse KL divergence value (lower is better, 0 is perfect)
        """
        reverse_kl = 0.0

        for state, q_x in p_empirical.items():
            if q_x > 1e-15:  # Avoid log(0)
                p_x = p_true.get(state, 1e-15)  # Smoothing
                reverse_kl += q_x * np.log(q_x / p_x)

        return reverse_kl

    def total_variation_distance(
        self,
        p_true: Dict[Tuple, float],
        p_empirical: Dict[Tuple, float]
    ) -> float:
        """
        Compute Total Variation Distance: TV(p, q) = 0.5 * sum_x |p(x) - q(x)|

        Args:
            p_true: True distribution p(x)
            p_empirical: Empirical distribution q(x)

        Returns:
            Total variation distance (range [0, 1], lower is better)
        """
        # Get all states from both distributions
        all_states = set(p_true.keys()) | set(p_empirical.keys())

        tv_distance = 0.0
        for state in all_states:
            p_x = p_true.get(state, 0.0)
            q_x = p_empirical.get(state, 0.0)
            tv_distance += abs(p_x - q_x)

        return 0.5 * tv_distance

    def maximum_mean_discrepancy(
        self,
        samples_p: np.ndarray,
        samples_q: np.ndarray
    ) -> float:
        """
        Compute Maximum Mean Discrepancy (MMD) between two sample sets.

        MMD^2(P, Q) = E[k(x, x')] - 2*E[k(x, y)] + E[k(y, y')]
        where x, x' ~ P and y, y' ~ Q

        Args:
            samples_p: Samples from first distribution, shape (n_p, n_variables)
            samples_q: Samples from second distribution, shape (n_q, n_variables)

        Returns:
            MMD value (lower is better, 0 is perfect)
        """
        # Select kernel
        if self.mmd_kernel == 'rbf_hamming':
            kernel = self._rbf_hamming_kernel
        elif self.mmd_kernel == 'linear_hamming':
            kernel = self._linear_hamming_kernel
        else:
            raise ValueError(f"Unknown MMD kernel: {self.mmd_kernel}")

        # Compute kernel matrices
        k_pp = kernel(samples_p, samples_p)
        k_qq = kernel(samples_q, samples_q)
        k_pq = kernel(samples_p, samples_q)

        # MMD^2 = E[k(p,p)] - 2*E[k(p,q)] + E[k(q,q)]
        # Note: We exclude diagonal for unbiased estimate
        n_p = len(samples_p)
        n_q = len(samples_q)

        # Unbiased estimates
        term1 = (np.sum(k_pp) - np.trace(k_pp)) / (n_p * (n_p - 1))
        term2 = np.mean(k_pq)
        term3 = (np.sum(k_qq) - np.trace(k_qq)) / (n_q * (n_q - 1))

        mmd_squared = term1 - 2 * term2 + term3

        # MMD can be slightly negative due to sampling variance, so clip
        mmd = np.sqrt(np.maximum(mmd_squared, 0.0))

        return mmd

    def moment_matching_error(
        self,
        samples_empirical: np.ndarray,
        samples_true: np.ndarray
    ) -> float:
        """
        Compute moment matching error between empirical and true samples.

        Δ_mom = (1/N) * Σ_i |μ_i - μ̂_i| + 1/(N(N-1)) * Σ_{i<j} |C_ij - Ĉ_ij|

        Where:
        - μ_i is the true first moment (mean) of variable i
        - Ĉ_ij is the empirical second moment (covariance) between variables i, j

        Args:
            samples_empirical: Empirical samples, shape (n_samples, n_variables)
            samples_true: True distribution samples, shape (n_samples, n_variables)

        Returns:
            Aggregate moment matching error (lower is better, 0 is perfect)
        """
        n_vars = samples_empirical.shape[1]

        # Compute first moments (means)
        mu_true = np.mean(samples_true, axis=0)
        mu_empirical = np.mean(samples_empirical, axis=0)

        # First moment error
        first_moment_error = np.mean(np.abs(mu_true - mu_empirical))

        # Compute second moments (covariances)
        cov_true = np.cov(samples_true.T)
        cov_empirical = np.cov(samples_empirical.T)

        # Second moment error (only upper triangle, excluding diagonal)
        second_moment_error = 0.0
        count = 0
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                second_moment_error += np.abs(cov_true[i, j] - cov_empirical[i, j])
                count += 1

        if count > 0:
            second_moment_error /= count
        else:
            second_moment_error = 0.0

        # Aggregate: equal weighting
        aggregate_error = first_moment_error + second_moment_error

        return aggregate_error

    def _rbf_hamming_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        RBF kernel based on Hamming distance for binary vectors.

        k(x, y) = exp(-gamma * d_hamming(x, y))

        where d_hamming(x, y) = number of positions where x != y

        Args:
            X: First sample set, shape (n_x, n_features)
            Y: Second sample set, shape (n_y, n_features)

        Returns:
            Kernel matrix, shape (n_x, n_y)
        """
        # Compute Hamming distances
        # For binary vectors: d_hamming = sum(x != y) = sum(|x - y|) / 2 (if x, y in {-1, 1})
        # But we can use: d_hamming = 0.5 * ||x - y||_1

        # Expand dimensions for broadcasting
        X_expanded = X[:, np.newaxis, :]  # (n_x, 1, n_features)
        Y_expanded = Y[np.newaxis, :, :]  # (1, n_y, n_features)

        # Hamming distance for {-1, 1} vectors
        hamming_dist = np.sum(X_expanded != Y_expanded, axis=2)

        # RBF kernel
        gamma = 1.0 / (2 * self.mmd_kernel_bandwidth ** 2)
        kernel_matrix = np.exp(-gamma * hamming_dist)

        return kernel_matrix

    def _linear_hamming_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Linear kernel based on Hamming similarity for binary vectors.

        k(x, y) = (n_features - d_hamming(x, y)) / n_features

        This gives 1 for identical vectors and 0 for completely different ones.

        Args:
            X: First sample set, shape (n_x, n_features)
            Y: Second sample set, shape (n_y, n_features)

        Returns:
            Kernel matrix, shape (n_x, n_y)
        """
        n_features = X.shape[1]

        # Expand dimensions for broadcasting
        X_expanded = X[:, np.newaxis, :]  # (n_x, 1, n_features)
        Y_expanded = Y[np.newaxis, :, :]  # (1, n_y, n_features)

        # Hamming distance
        hamming_dist = np.sum(X_expanded != Y_expanded, axis=2)

        # Linear similarity kernel
        kernel_matrix = (n_features - hamming_dist) / n_features

        return kernel_matrix

    def _sample_from_distribution(
        self,
        distribution: Dict[Tuple, float],
        n_samples: int
    ) -> np.ndarray:
        """
        Generate samples from a discrete probability distribution.

        Args:
            distribution: Dictionary mapping states (tuples) to probabilities
            n_samples: Number of samples to generate

        Returns:
            Samples array, shape (n_samples, n_variables)
        """
        states = list(distribution.keys())
        probabilities = list(distribution.values())

        # Normalize probabilities (in case of numerical errors)
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()

        # Sample state indices
        n_states = len(states)
        state_indices = np.random.choice(n_states, size=n_samples, p=probabilities)

        # Convert to array
        samples = np.array([states[idx] for idx in state_indices])

        return samples


def get_metric_display_name(metric_name: str) -> str:
    """
    Get human-readable display name for a metric.

    Args:
        metric_name: Internal metric name

    Returns:
        Display name for plots and tables
    """
    display_names = {
        'kl_divergence': 'KL Divergence',
        'reverse_kl_divergence': 'Reverse KL Divergence',
        'total_variation': 'Total Variation Distance',
        'mmd': 'Maximum Mean Discrepancy',
        'moment_matching': 'Moment Matching Error'
    }
    return display_names.get(metric_name, metric_name)


def get_metric_direction(metric_name: str) -> str:
    """
    Get optimization direction for a metric.

    Args:
        metric_name: Internal metric name

    Returns:
        'lower' if lower is better, 'higher' if higher is better
    """
    # All implemented metrics: lower is better
    return 'lower'
