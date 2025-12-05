"""
Benchmark Visualization - Create plots for benchmark results.

Includes:
- Performance comparison plots
- Convergence plots
- Distribution comparison plots
- Autocorrelation plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


class BenchmarkVisualizer:
    """
    Create visualizations for benchmark results.
    """

    def __init__(self, output_dir: str, plot_format: str = 'png'):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
            plot_format: Format for plots ('png' or 'pdf')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_format = plot_format

        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10

    def plot_sampling_time_comparison(
        self,
        results: Dict[str, Dict[int, Dict[str, Any]]],
        filename: str = 'sampling_time_comparison'
    ):
        """
        Plot sampling time comparison across samplers and problem sizes.

        Args:
            results: Nested dict {sampler_name: {n_variables: metrics}}
            filename: Output filename (without extension)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for sampler_name, sampler_results in results.items():
            n_vars = sorted(sampler_results.keys())
            times = [sampler_results[n]['sampling_time'] for n in n_vars]

            ax.plot(n_vars, times, marker='o', label=sampler_name, linewidth=2)

        ax.set_xlabel('Number of Variables', fontsize=12)
        ax.set_ylabel('Sampling Time (seconds)', fontsize=12)
        ax.set_title('Sampling Time Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.{self.plot_format}', dpi=300)
        plt.close()

    def plot_throughput_comparison(
        self,
        results: Dict[str, Dict[int, Dict[str, Any]]],
        filename: str = 'throughput_comparison'
    ):
        """
        Plot sampling throughput (samples/sec) comparison.

        Args:
            results: Nested dict {sampler_name: {n_variables: metrics}}
            filename: Output filename (without extension)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for sampler_name, sampler_results in results.items():
            n_vars = sorted(sampler_results.keys())
            throughput = [sampler_results[n]['samples_per_second'] for n in n_vars]

            ax.plot(n_vars, throughput, marker='o', label=sampler_name, linewidth=2)

        ax.set_xlabel('Number of Variables', fontsize=12)
        ax.set_ylabel('Samples per Second', fontsize=12)
        ax.set_title('Sampling Throughput Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.{self.plot_format}', dpi=300)
        plt.close()

    def plot_kl_divergence_comparison(
        self,
        results: Dict[str, Dict[int, Dict[str, Any]]],
        filename: str = 'kl_divergence_comparison'
    ):
        """
        Plot KL divergence comparison (accuracy metric).

        Args:
            results: Nested dict {sampler_name: {n_variables: metrics}}
            filename: Output filename (without extension)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for sampler_name, sampler_results in results.items():
            n_vars = []
            kl_divs = []

            for n, metrics in sorted(sampler_results.items()):
                if metrics.get('kl_divergence') is not None:
                    n_vars.append(n)
                    kl_divs.append(metrics['kl_divergence'])

            if kl_divs:
                ax.plot(n_vars, kl_divs, marker='o', label=sampler_name, linewidth=2)

        ax.set_xlabel('Number of Variables', fontsize=12)
        ax.set_ylabel('KL Divergence', fontsize=12)
        ax.set_title('KL Divergence from True Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.{self.plot_format}', dpi=300)
        plt.close()

    def plot_effective_sample_size(
        self,
        results: Dict[str, Dict[int, Dict[str, Any]]],
        filename: str = 'effective_sample_size'
    ):
        """
        Plot effective sample size comparison.

        Args:
            results: Nested dict {sampler_name: {n_variables: metrics}}
            filename: Output filename (without extension)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for sampler_name, sampler_results in results.items():
            n_vars = sorted(sampler_results.keys())
            ess = [sampler_results[n]['effective_sample_size'] for n in n_vars]

            ax.plot(n_vars, ess, marker='o', label=sampler_name, linewidth=2)

        ax.set_xlabel('Number of Variables', fontsize=12)
        ax.set_ylabel('Effective Sample Size', fontsize=12)
        ax.set_title('Effective Sample Size Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.{self.plot_format}', dpi=300)
        plt.close()

    def plot_autocorrelation(
        self,
        results: Dict[str, Dict[int, Dict[str, Any]]],
        n_variables: int,
        max_lag: int = 100,
        filename: Optional[str] = None
    ):
        """
        Plot autocorrelation for different samplers at fixed problem size.

        Args:
            results: Nested dict {sampler_name: {n_variables: metrics}}
            n_variables: Problem size to plot
            max_lag: Maximum lag to plot
            filename: Output filename (without extension)
        """
        if filename is None:
            filename = f'autocorrelation_n{n_variables}'

        fig, ax = plt.subplots(figsize=(12, 6))

        for sampler_name, sampler_results in results.items():
            if n_variables in sampler_results:
                autocorr = sampler_results[n_variables]['autocorrelation']
                lags = np.arange(min(len(autocorr), max_lag + 1))
                ax.plot(lags, autocorr[:len(lags)], marker='.', label=sampler_name, linewidth=2)

        ax.set_xlabel('Lag', fontsize=12)
        ax.set_ylabel('Autocorrelation', fontsize=12)
        ax.set_title(f'Autocorrelation Comparison (n={n_variables})', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.{self.plot_format}', dpi=300)
        plt.close()

    def plot_distribution_comparison(
        self,
        results: Dict[str, Dict[int, Dict[str, Any]]],
        n_variables: int,
        filename: Optional[str] = None
    ):
        """
        Plot exact vs empirical distributions for different samplers.

        Args:
            results: Nested dict {sampler_name: {n_variables: metrics}}
            n_variables: Problem size to plot
            filename: Output filename (without extension)
        """
        if filename is None:
            filename = f'distribution_comparison_n{n_variables}'

        # Only plot if n_variables is small enough
        if n_variables > 10:
            print(f"Skipping distribution plot for n={n_variables} (too many states)")
            return

        fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)))
        if len(results) == 1:
            axes = [axes]

        for idx, (sampler_name, sampler_results) in enumerate(results.items()):
            if n_variables in sampler_results:
                metrics = sampler_results[n_variables]

                if 'exact_distribution' in metrics and 'empirical_distribution' in metrics:
                    exact_dist = metrics['exact_distribution']
                    empirical_dist = metrics['empirical_distribution']

                    x = np.arange(len(exact_dist))
                    width = 0.35

                    axes[idx].bar(x - width/2, exact_dist, width, label='Exact', alpha=0.7)
                    axes[idx].bar(x + width/2, empirical_dist, width, label='Empirical', alpha=0.7)

                    axes[idx].set_xlabel('State Index', fontsize=10)
                    axes[idx].set_ylabel('Probability', fontsize=10)
                    axes[idx].set_title(f'{sampler_name} (n={n_variables})', fontsize=12, fontweight='bold')
                    axes[idx].legend(loc='best', fontsize=9)
                    axes[idx].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.{self.plot_format}', dpi=300)
        plt.close()

    def create_summary_table(
        self,
        results: Dict[str, Dict[int, Dict[str, Any]]],
        filename: str = 'benchmark_summary'
    ):
        """
        Create a summary table of benchmark results.

        Args:
            results: Nested dict {sampler_name: {n_variables: metrics}}
            filename: Output filename (without extension)
        """
        rows = []

        for sampler_name, sampler_results in results.items():
            for n_vars, metrics in sorted(sampler_results.items()):
                row = {
                    'Sampler': sampler_name,
                    'N Variables': n_vars,
                    'Sampling Time (s)': f"{metrics['sampling_time']:.4f}",
                    'Samples/sec': f"{metrics['samples_per_second']:.2f}",
                    'ESS': f"{metrics['effective_sample_size']:.2f}",
                }

                if metrics.get('kl_divergence') is not None:
                    row['KL Divergence'] = f"{metrics['kl_divergence']:.6f}"
                    row['TV Distance'] = f"{metrics['total_variation_distance']:.6f}"

                rows.append(row)

        df = pd.DataFrame(rows)

        # Save to CSV
        df.to_csv(self.output_dir / f'{filename}.csv', index=False)

        # Create text table
        with open(self.output_dir / f'{filename}.txt', 'w') as f:
            f.write(df.to_string(index=False))

        return df

    def create_all_plots(
        self,
        results: Dict[str, Dict[int, Dict[str, Any]]]
    ):
        """
        Create all standard benchmark plots.

        Args:
            results: Nested dict {sampler_name: {n_variables: metrics}}
        """
        print("\nCreating benchmark visualizations...")

        # Performance plots
        self.plot_sampling_time_comparison(results)
        print("  [OK] Sampling time comparison")

        self.plot_throughput_comparison(results)
        print("  [OK] Throughput comparison")

        # Accuracy plots
        self.plot_kl_divergence_comparison(results)
        print("  [OK] KL divergence comparison")

        self.plot_effective_sample_size(results)
        print("  [OK] Effective sample size comparison")

        # Autocorrelation plots for small problems
        for n_vars in sorted(set(n for sampler_results in results.values() for n in sampler_results.keys())):
            if n_vars <= 10:
                self.plot_autocorrelation(results, n_vars)
                print(f"  [OK] Autocorrelation (n={n_vars})")

        # Distribution plots for very small problems
        for n_vars in sorted(set(n for sampler_results in results.values() for n in sampler_results.keys())):
            if n_vars <= 6:
                self.plot_distribution_comparison(results, n_vars)
                print(f"  [OK] Distribution comparison (n={n_vars})")

        # Summary table
        df = self.create_summary_table(results)
        print("  [OK] Summary table")

        print(f"\nAll plots saved to: {self.output_dir}")
