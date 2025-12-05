"""
Benchmark Metrics - Compute various metrics for sampler comparison.

Includes:
- Sampling time and throughput
- KL divergence from true distribution
- Total variation distance
- Autocorrelation and effective sample size
- Convergence metrics
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from scipy.special import logsumexp
from scipy.stats import entropy
import time


class BenchmarkMetrics:
    """
    Compute benchmarking metrics for MCMC samplers.
    """

    @staticmethod
    def compute_sampling_time(
        model,
        sampler_name: str,
        n_samples: int,
        sampler_params: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Measure sampling time and throughput.

        Args:
            model: BoltzmannMachine instance
            sampler_name: Name of sampler to test
            n_samples: Number of samples to generate
            sampler_params: Parameters for sampler

        Returns:
            (sampling_time, samples_per_second)
        """
        start_time = time.time()

        samples = model.sample(
            sampler_name=sampler_name,
            prefactor=1.0,
            sample_params={
                'num_reads': n_samples,
                **sampler_params
            },
            as_tensor=True
        )

        elapsed_time = time.time() - start_time
        samples_per_second = n_samples / elapsed_time if elapsed_time > 0 else 0

        return elapsed_time, samples_per_second

    @staticmethod
    def compute_exact_distribution(model) -> np.ndarray:
        """
        Compute exact probability distribution for small BMs.

        Args:
            model: BoltzmannMachine instance

        Returns:
            Probability distribution over all states
        """
        n_visible = len(model.visible_idx)

        # Generate all possible states
        n_states = 2 ** n_visible
        all_states = np.zeros((n_states, n_visible), dtype=np.float32)

        for i in range(n_states):
            binary = format(i, f'0{n_visible}b')
            all_states[i] = [1.0 if b == '1' else -1.0 for b in binary]

        # Convert to tensor
        states_tensor = torch.tensor(all_states, dtype=torch.float32)

        # Compute energies
        linear, quadratic = model.get_parameters()
        energies = torch.zeros(n_states)

        for i, state in enumerate(states_tensor):
            # Linear term
            energy = -torch.sum(linear[model.visible_idx] * state)

            # Quadratic term
            for edge_idx, (u, v) in enumerate(model.edges):
                if u in model.visible_idx and v in model.visible_idx:
                    u_vis_idx = model.visible_idx.index(u)
                    v_vis_idx = model.visible_idx.index(v)
                    energy -= quadratic[edge_idx] * state[u_vis_idx] * state[v_vis_idx]

            energies[i] = energy

        # Compute probabilities using Boltzmann distribution
        energies_np = energies.cpu().numpy()
        log_probs = -energies_np
        log_Z = logsumexp(log_probs)
        probs = np.exp(log_probs - log_Z)

        return probs

    @staticmethod
    def compute_empirical_distribution(
        samples: np.ndarray,
        n_visible: int
    ) -> np.ndarray:
        """
        Compute empirical distribution from samples.

        Args:
            samples: Array of samples (n_samples, n_visible)
            n_visible: Number of visible units

        Returns:
            Empirical probability distribution
        """
        n_states = 2 ** n_visible
        counts = np.zeros(n_states)

        # Convert samples to state indices
        for sample in samples:
            # Convert spin {-1, +1} to binary {0, 1}
            binary = [(s + 1) / 2 for s in sample]
            # Convert to state index
            state_idx = sum(b * (2 ** i) for i, b in enumerate(reversed(binary)))
            counts[int(state_idx)] += 1

        # Normalize to get probabilities
        probs = counts / len(samples)

        return probs

    @staticmethod
    def compute_kl_divergence(
        p: np.ndarray,
        q: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Compute KL divergence KL(p || q).

        Args:
            p: True distribution
            q: Approximate distribution
            epsilon: Small value to avoid log(0)

        Returns:
            KL divergence
        """
        # Add epsilon to avoid log(0)
        p_safe = np.clip(p, epsilon, 1.0)
        q_safe = np.clip(q, epsilon, 1.0)

        return entropy(p_safe, q_safe)

    @staticmethod
    def compute_total_variation_distance(
        p: np.ndarray,
        q: np.ndarray
    ) -> float:
        """
        Compute total variation distance between distributions.

        Args:
            p: Distribution 1
            q: Distribution 2

        Returns:
            Total variation distance
        """
        return 0.5 * np.sum(np.abs(p - q))

    @staticmethod
    def compute_autocorrelation(
        samples: np.ndarray,
        max_lag: int = 100
    ) -> np.ndarray:
        """
        Compute autocorrelation of samples.

        Args:
            samples: Array of samples (n_samples, n_visible)
            max_lag: Maximum lag to compute

        Returns:
            Autocorrelation values for each lag
        """
        n_samples, n_visible = samples.shape

        # Compute autocorrelation for first variable
        x = samples[:, 0]
        x_mean = np.mean(x)
        x_centered = x - x_mean

        autocorr = np.zeros(max_lag + 1)
        c0 = np.sum(x_centered ** 2) / n_samples

        for lag in range(max_lag + 1):
            if lag < n_samples:
                c_lag = np.sum(x_centered[:n_samples - lag] * x_centered[lag:]) / n_samples
                autocorr[lag] = c_lag / c0 if c0 > 0 else 0

        return autocorr

    @staticmethod
    def compute_effective_sample_size(
        autocorr: np.ndarray,
        n_samples: int
    ) -> float:
        """
        Compute effective sample size from autocorrelation.

        Args:
            autocorr: Autocorrelation values
            n_samples: Total number of samples

        Returns:
            Effective sample size
        """
        # Integrated autocorrelation time
        tau_int = 0.5 + np.sum(autocorr[1:])

        # Effective sample size
        n_eff = n_samples / (2 * tau_int) if tau_int > 0 else n_samples

        return max(1.0, n_eff)

    @staticmethod
    def compute_all_metrics(
        model,
        sampler_name: str,
        n_samples: int,
        sampler_params: Dict[str, Any],
        compute_exact: bool = True
    ) -> Dict[str, Any]:
        """
        Compute all benchmark metrics for a sampler.

        Args:
            model: BoltzmannMachine instance
            sampler_name: Name of sampler to test
            n_samples: Number of samples to generate
            sampler_params: Parameters for sampler
            compute_exact: Whether to compute exact distribution metrics

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Sampling time
        sampling_time, samples_per_second = BenchmarkMetrics.compute_sampling_time(
            model, sampler_name, n_samples, sampler_params
        )
        metrics['sampling_time'] = sampling_time
        metrics['samples_per_second'] = samples_per_second

        # Generate samples for analysis
        samples = model.sample(
            sampler_name=sampler_name,
            prefactor=1.0,
            sample_params={
                'num_reads': n_samples,
                **sampler_params
            },
            as_tensor=True
        )

        # Extract visible units
        visible_samples = samples[:n_samples, model.visible_idx].cpu().numpy()

        # Autocorrelation and ESS
        autocorr = BenchmarkMetrics.compute_autocorrelation(visible_samples, max_lag=100)
        ess = BenchmarkMetrics.compute_effective_sample_size(autocorr, n_samples)
        metrics['autocorrelation'] = autocorr
        metrics['effective_sample_size'] = ess

        # Exact distribution metrics (only for small problems)
        if compute_exact:
            try:
                n_visible = len(model.visible_idx)
                exact_dist = BenchmarkMetrics.compute_exact_distribution(model)
                empirical_dist = BenchmarkMetrics.compute_empirical_distribution(
                    visible_samples, n_visible
                )

                kl_div = BenchmarkMetrics.compute_kl_divergence(exact_dist, empirical_dist)
                tv_dist = BenchmarkMetrics.compute_total_variation_distance(
                    exact_dist, empirical_dist
                )

                metrics['kl_divergence'] = kl_div
                metrics['total_variation_distance'] = tv_dist
                metrics['exact_distribution'] = exact_dist
                metrics['empirical_distribution'] = empirical_dist

            except Exception as e:
                print(f"  Warning: Could not compute exact metrics: {e}")
                metrics['kl_divergence'] = None
                metrics['total_variation_distance'] = None

        return metrics
