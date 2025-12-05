"""
Sampler Benchmarking Module

This module benchmarks different sampling methods by computing the KL divergence
between the empirical distribution (p-hat) from samples and the true Boltzmann
distribution (p).
"""

import numpy as np
import pandas as pd
import torch
import time
from typing import Dict, List, Tuple
from collections import Counter
import itertools
from tqdm import tqdm

from utils.topology import create_topology
from utils.parameters import generate_random_parameters
from utils.sampler_factory import create_sampler
from utils.benchmark_metrics import BenchmarkMetrics
from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine as GRBM


class SamplerBenchmark:
    """
    Benchmarks samplers by computing KL divergence between empirical and true distributions.
    """

    def __init__(self, config: Dict):
        """
        Initialize the benchmark with configuration.

        Args:
            config: Configuration dictionary containing benchmark settings
        """
        self.config = config
        self.benchmark_config = config.get('benchmark', {})

        # Device setup
        device_config = config.get('device', {})
        use_cuda = device_config.get('use_cuda', 'auto')

        if use_cuda == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif use_cuda == 'cuda' or use_cuda is True:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Initialize metrics computer
        self.metrics_computer = BenchmarkMetrics(config)

        self.results = []

    def _create_true_model(self, n_variables: int) -> Tuple[GRBM, Dict]:
        """
        Create a fully visible dense Boltzmann Machine.

        Args:
            n_variables: Number of visible variables

        Returns:
            Tuple of (GRBM model, config dict with model info)
        """
        # Create dense FVBM topology
        nodes, edges, hidden_nodes = create_topology(
            n_visible=n_variables,
            n_hidden=0,
            model_type="fvbm",
            connectivity="dense",
            seed=self.config.get('seed', 42)
        )

        # Generate random parameters
        # Use benchmark config for model parameters
        linear_scale = self.benchmark_config.get('linear_bias_scale', 1.0)
        quadratic_scale = self.benchmark_config.get('quadratic_weight_scale', 1.5)

        linear, quadratic = generate_random_parameters(
            nodes=nodes,
            edges=edges,
            seed=self.config.get('seed', 42),
            linear_scale=linear_scale,
            quadratic_scale=quadratic_scale
        )

        # Create GRBM
        model = GRBM(
            nodes=nodes,
            edges=edges,
            hidden_nodes=hidden_nodes,
            linear=linear,
            quadratic=quadratic
        )

        model = model.to(self.device)

        model_info = {
            'n_variables': n_variables,
            'nodes': nodes,
            'edges': edges,
            'linear': linear,
            'quadratic': quadratic
        }

        return model, model_info

    def _compute_exact_distribution(self, model: GRBM, n_variables: int) -> Dict[Tuple, float]:
        """
        Compute the exact Boltzmann distribution p(x) in closed form.

        For a Boltzmann Machine, the probability is:
        p(x) = exp(-E(x)) / Z
        where E(x) = -sum(h_i * x_i) - sum(J_ij * x_i * x_j)
        and Z is the partition function (sum over all configurations)

        Args:
            model: The GRBM model
            n_variables: Number of variables

        Returns:
            Dictionary mapping state tuples to probabilities
        """
        # Generate all possible binary states {-1, +1}^n
        all_states = list(itertools.product([-1, 1], repeat=n_variables))

        # Convert to tensor for batch energy computation
        states_tensor = torch.tensor(all_states, dtype=torch.float32, device=self.device)

        # Compute energies for all states
        # Energy: E(x) = -linear^T x - x^T quadratic x
        linear_params = model.linear.to(self.device)
        quadratic_params = model.quadratic.to(self.device)

        # Linear term: (batch_size,) = (batch_size, n_var) @ (n_var,)
        linear_energy = torch.matmul(states_tensor, linear_params)

        # Quadratic term: We need to reconstruct the quadratic matrix
        # quadratic_params is a flat vector of edge weights
        # We need to map it back to the interaction matrix
        n = n_variables
        quadratic_matrix = torch.zeros((n, n), device=self.device)

        # Get edges from model
        edge_list = model.edges
        for idx, (i, j) in enumerate(edge_list):
            if i < n and j < n:  # Only visible-visible edges for FVBM
                quadratic_matrix[i, j] = quadratic_params[idx]
                quadratic_matrix[j, i] = quadratic_params[idx]  # Symmetric

        # Quadratic energy: (batch_size,) = sum over (batch_size, n) * ((n, n) @ (batch_size, n)^T)^T
        # For each sample: x^T Q x
        quadratic_energy = torch.sum(
            states_tensor * torch.matmul(states_tensor, quadratic_matrix),
            dim=1
        )

        # Total energy (note: BM uses negative sign convention)
        energies = -(linear_energy + quadratic_energy)

        # Compute probabilities using Boltzmann distribution
        # p(x) = exp(-E(x)) / Z
        with torch.no_grad():  # Don't track gradients for exact distribution
            log_unnormalized = -energies
            log_Z = torch.logsumexp(log_unnormalized, dim=0)
            log_probs = log_unnormalized - log_Z
            probs = torch.exp(log_probs)

        # Convert to dictionary
        prob_dict = {}
        for state, prob in zip(all_states, probs.cpu().numpy()):
            prob_dict[tuple(state)] = float(prob)

        return prob_dict

    def _compute_empirical_distribution(self, samples: np.ndarray) -> Dict[Tuple, float]:
        """
        Compute empirical distribution p-hat(x) from samples.

        p-hat(x) = (1/M) * sum_{m=1}^M 1(x^(m) = x)

        Args:
            samples: Array of shape (n_samples, n_variables)

        Returns:
            Dictionary mapping state tuples to empirical probabilities
        """
        n_samples = len(samples)

        # Count occurrences of each state
        state_counts = Counter()
        for sample in samples:
            state = tuple(sample)
            state_counts[state] += 1

        # Convert counts to probabilities
        prob_dict = {}
        for state, count in state_counts.items():
            prob_dict[state] = count / n_samples

        return prob_dict

    def _compute_metrics(
        self,
        p_true: Dict[Tuple, float],
        p_empirical: Dict[Tuple, float],
        samples: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all enabled benchmark metrics.

        Args:
            p_true: True distribution p(x)
            p_empirical: Empirical distribution p-hat(x)
            samples: Empirical samples array

        Returns:
            Dictionary of metric names to values
        """
        return self.metrics_computer.compute_all_metrics(
            p_true=p_true,
            p_empirical=p_empirical,
            samples=samples,
            true_samples=None  # Will be generated internally if needed
        )

    def _sample_from_model(
        self,
        model: GRBM,
        sampler_type: str,
        sampler_params: Dict,
        n_samples: int
    ) -> np.ndarray:
        """
        Sample from the model using a specific sampler.

        Args:
            model: The GRBM model
            sampler_type: Type of sampler to use
            sampler_params: Parameters for the sampler
            n_samples: Number of samples to generate

        Returns:
            Array of samples, shape (n_samples, n_variables)
        """
        # Create sampler
        sampler = create_sampler(sampler_type, sampler_params)

        # Sample from model
        prefactor = self.config['data_generation'].get('prefactor', 1.0)
        num_reads = sampler_params.get('num_reads', n_samples)

        samples = model.sample(
            sampler=sampler,
            prefactor=prefactor,
            sample_params=sampler_params,
            as_tensor=True
        )

        # Extract visible units only and limit to n_samples
        visible_idx = model.visible_idx
        visible_samples = samples[:, visible_idx].cpu().numpy()

        # Take first n_samples if we got more
        if len(visible_samples) > n_samples:
            visible_samples = visible_samples[:n_samples]

        return visible_samples

    def benchmark_sampler(
        self,
        n_variables: int,
        sampler_type: str,
        n_samples: int
    ) -> Dict:
        """
        Benchmark a single sampler on a specific problem size.

        Args:
            n_variables: Number of variables in the BM
            sampler_type: Type of sampler to benchmark
            n_samples: Number of samples to generate

        Returns:
            Dictionary with benchmark results including timing information
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking: {sampler_type} | n_variables={n_variables} | n_samples={n_samples}")
        print(f"{'='*60}")

        # Start total timing
        total_start = time.time()

        # Create true model
        model, model_info = self._create_true_model(n_variables)

        # Compute exact distribution
        print("Computing exact Boltzmann distribution...")
        exact_start = time.time()
        p_true = self._compute_exact_distribution(model, n_variables)
        exact_time = time.time() - exact_start
        print(f"  Time: {exact_time:.3f}s")

        # Get sampler parameters from config
        default_params = {
            'beta_range': [1.0, 1.0],
            'proposal_acceptance_criteria': 'Gibbs',
            'num_reads': n_samples
        }

        # Get params from benchmark config
        if 'sampler_params' in self.benchmark_config:
            sampler_params = self.benchmark_config['sampler_params'].copy()
            sampler_params['num_reads'] = n_samples
        else:
            sampler_params = default_params

        # Generate samples with timing
        print(f"Generating {n_samples} samples using {sampler_type}...")
        sampling_start = time.time()
        samples = self._sample_from_model(model, sampler_type, sampler_params, n_samples)
        sampling_time = time.time() - sampling_start
        print(f"  Time: {sampling_time:.3f}s")
        print(f"  Samples/sec: {n_samples/sampling_time:.1f}")

        # Compute empirical distribution
        print("Computing empirical distribution...")
        empirical_start = time.time()
        p_empirical = self._compute_empirical_distribution(samples)
        empirical_time = time.time() - empirical_start

        # Compute all metrics with timing
        print("Computing benchmark metrics...")
        metrics_start = time.time()
        metrics = self._compute_metrics(p_true, p_empirical, samples)
        metrics_time = time.time() - metrics_start
        print(f"  Time: {metrics_time:.3f}s")

        # Total time
        total_time = time.time() - total_start

        # Print metrics
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.6f}")

        print(f"\nTotal benchmark time: {total_time:.3f}s")

        # Build result dictionary
        result = {
            'n_variables': n_variables,
            'sampler_type': sampler_type,
            'n_samples': n_samples,
            'n_unique_states_true': len(p_true),
            'n_unique_states_empirical': len(p_empirical),
            'sampling_time_sec': sampling_time,
            'samples_per_sec': n_samples / sampling_time,
            'exact_distribution_time_sec': exact_time,
            'metrics_computation_time_sec': metrics_time,
            'total_time_sec': total_time,
            'model_info': model_info
        }

        # Add all metrics to result
        result.update(metrics)

        return result

    def run_benchmark_suite(self) -> pd.DataFrame:
        """
        Run the full benchmark suite across all configurations.

        Returns:
            DataFrame with all benchmark results
        """
        # Get benchmark configuration
        samplers = self.benchmark_config.get('samplers', [])
        n_variables_range = self.benchmark_config.get('n_variables_range', [2, 3, 4, 5])
        n_samples = self.benchmark_config.get('n_samples', 10000)

        print(f"\n{'#'*60}")
        print(f"# Starting Benchmark Suite")
        print(f"# Samplers: {samplers}")
        print(f"# Variables: {n_variables_range}")
        print(f"# Samples per run: {n_samples}")
        print(f"# Total runs: {len(samplers) * len(n_variables_range)}")
        print(f"{'#'*60}\n")

        results = []

        # Iterate through all combinations
        for n_vars in n_variables_range:
            for sampler_type in samplers:
                try:
                    result = self.benchmark_sampler(
                        n_variables=n_vars,
                        sampler_type=sampler_type,
                        n_samples=n_samples
                    )
                    results.append(result)

                except Exception as e:
                    print(f"ERROR: Failed to benchmark {sampler_type} with {n_vars} variables")
                    print(f"Exception: {str(e)}")
                    # Continue with next configuration
                    continue

        # Convert to DataFrame
        # Extract all metric columns dynamically
        if len(results) == 0:
            df_results = pd.DataFrame()
        else:
            # Get all metric names from first result
            metric_names = [k for k in results[0].keys()
                          if k not in ['n_variables', 'sampler_type', 'n_samples',
                                       'n_unique_states_true', 'n_unique_states_empirical', 'model_info']]

            df_results = pd.DataFrame([
                {
                    'n_variables': r['n_variables'],
                    'sampler': r['sampler_type'],
                    'n_samples': r['n_samples'],
                    'n_unique_states_true': r['n_unique_states_true'],
                    'n_unique_states_empirical': r['n_unique_states_empirical'],
                    **{metric: r.get(metric, np.nan) for metric in metric_names}
                }
                for r in results
            ])

        self.results = results

        return df_results

    def save_results(self, df_results: pd.DataFrame, save_path: str):
        """
        Save benchmark results to CSV.

        Args:
            df_results: DataFrame with results
            save_path: Path to save CSV file
        """
        df_results.to_csv(save_path, index=False)
        print(f"\nResults saved to: {save_path}")
