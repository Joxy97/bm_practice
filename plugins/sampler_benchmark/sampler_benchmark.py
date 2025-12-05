"""
Sampler Benchmark - Benchmark and compare BM samplers.

Benchmarks different samplers on various problem sizes and configurations,
computing metrics like sampling time, KL divergence, autocorrelation, etc.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bm_core.models import BoltzmannMachine
from bm_core.utils import create_topology, generate_random_parameters
from plugins.sampler_benchmark.utils import BenchmarkMetrics, BenchmarkVisualizer


class SamplerBenchmark:
    """
    Benchmark different samplers on BM problems.

    Tests samplers across various problem sizes and computes comprehensive
    metrics including performance, accuracy, and convergence properties.
    """

    def __init__(self, config: Dict[str, Any], sampler_dict: Dict[str, Any]):
        """
        Initialize benchmark.

        Args:
            config: Benchmark configuration from YAML
            sampler_dict: Dictionary of available samplers from factory
        """
        self.config = config
        self.sampler_dict = sampler_dict
        self.seed = config.get('seed', 42)

        # Set random seeds
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Initialize results storage
        self.results = {}

        print(f"\nSampler Benchmark initialized")
        print(f"  Samplers to test: {len(config['benchmark']['samplers'])}")
        print(f"  Problem sizes: {config['benchmark']['n_variables_range']}")

    def _create_test_model(self, n_visible: int) -> BoltzmannMachine:
        """
        Create a test BM model for benchmarking.

        Args:
            n_visible: Number of visible units

        Returns:
            BoltzmannMachine instance
        """
        benchmark_config = self.config['benchmark']
        model_params = benchmark_config.get('model_params', {})

        # Create topology
        nodes, edges, hidden_nodes = create_topology(
            n_visible=n_visible,
            n_hidden=model_params.get('n_hidden', 0),
            model_type=benchmark_config['model_type'],
            connectivity=model_params.get('connectivity', 'dense'),
            connectivity_density=model_params.get('connectivity_density', 0.5),
            seed=self.seed
        )

        # Generate random parameters
        linear, quadratic = generate_random_parameters(
            nodes,
            edges,
            seed=self.seed,
            linear_scale=model_params.get('linear_bias_scale', 1.0),
            quadratic_scale=model_params.get('quadratic_weight_scale', 1.5)
        )

        # Create BoltzmannMachine
        model = BoltzmannMachine(
            nodes=nodes,
            edges=edges,
            hidden_nodes=hidden_nodes if model_params.get('n_hidden', 0) > 0 else None,
            linear=linear,
            quadratic=quadratic,
            sampler_dict=self.sampler_dict
        )

        return model

    def benchmark_sampler(
        self,
        sampler_name: str,
        n_variables: int
    ) -> Dict[str, Any]:
        """
        Benchmark a single sampler on a problem of given size.

        Args:
            sampler_name: Name of sampler to test
            n_variables: Number of visible variables

        Returns:
            Dictionary of benchmark metrics
        """
        print(f"\n  Testing {sampler_name} on n={n_variables}...")

        # Create test model
        model = self._create_test_model(n_variables)

        # Get sampler parameters
        sampler_params = self.config.get('sampler_params', {}).get(sampler_name, {})
        n_samples = self.config['benchmark']['n_samples']

        # Compute whether to use exact metrics (only for small problems)
        compute_exact = n_variables <= 10

        # Compute all metrics
        try:
            metrics = BenchmarkMetrics.compute_all_metrics(
                model=model,
                sampler_name=sampler_name,
                n_samples=n_samples,
                sampler_params=sampler_params,
                compute_exact=compute_exact
            )

            print(f"    Time: {metrics['sampling_time']:.4f}s")
            print(f"    Throughput: {metrics['samples_per_second']:.2f} samples/s")
            print(f"    ESS: {metrics['effective_sample_size']:.2f}")

            if metrics.get('kl_divergence') is not None:
                print(f"    KL divergence: {metrics['kl_divergence']:.6f}")

            return metrics

        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_benchmark(self) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Run full benchmark suite.

        Returns:
            Nested dict {sampler_name: {n_variables: metrics}}
        """
        benchmark_config = self.config['benchmark']

        print(f"\n{'='*70}")
        print("SAMPLER BENCHMARK SUITE")
        print(f"{'='*70}")

        samplers = benchmark_config['samplers']
        n_variables_range = benchmark_config['n_variables_range']

        results = {}

        for sampler_name in samplers:
            print(f"\nBenchmarking: {sampler_name}")

            # Check if sampler is available
            if sampler_name not in self.sampler_dict:
                print(f"  Warning: Sampler '{sampler_name}' not found in factory. Skipping.")
                continue

            results[sampler_name] = {}

            for n_vars in n_variables_range:
                metrics = self.benchmark_sampler(sampler_name, n_vars)

                if metrics is not None:
                    results[sampler_name][n_vars] = metrics

        self.results = results
        return results

    def save_results(self, output_dir: str):
        """
        Save benchmark results to file.

        Args:
            output_dir: Directory to save results
        """
        import json

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, 'benchmark_results.json')

        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for sampler_name, sampler_results in self.results.items():
            results_serializable[sampler_name] = {}
            for n_vars, metrics in sampler_results.items():
                serializable_metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, np.ndarray):
                        serializable_metrics[key] = value.tolist()
                    elif isinstance(value, (np.int64, np.int32)):
                        serializable_metrics[key] = int(value)
                    elif isinstance(value, (np.float64, np.float32)):
                        serializable_metrics[key] = float(value)
                    else:
                        serializable_metrics[key] = value

                results_serializable[sampler_name][str(n_vars)] = serializable_metrics

        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"\n[OK] Results saved to: {filepath}")

    def create_visualizations(self, plot_dir: str, plot_format: str = 'png'):
        """
        Create benchmark visualizations.

        Args:
            plot_dir: Directory to save plots
            plot_format: Format for plots ('png' or 'pdf')
        """
        visualizer = BenchmarkVisualizer(plot_dir, plot_format)
        visualizer.create_all_plots(self.results)

    def print_summary(self):
        """Print summary of benchmark results."""
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")

        for sampler_name, sampler_results in self.results.items():
            print(f"\n{sampler_name}:")

            for n_vars in sorted(sampler_results.keys()):
                metrics = sampler_results[n_vars]
                print(f"  n={n_vars:2d}: "
                      f"time={metrics['sampling_time']:6.3f}s, "
                      f"throughput={metrics['samples_per_second']:8.2f} samples/s, "
                      f"ESS={metrics['effective_sample_size']:7.2f}", end='')

                if metrics.get('kl_divergence') is not None:
                    print(f", KL={metrics['kl_divergence']:.6f}")
                else:
                    print()

        print(f"\n{'='*70}")
