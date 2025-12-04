"""
Benchmark Visualization Module

Creates comprehensive visualizations for sampler benchmark results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union


def plot_benchmark_results(
    df_results: pd.DataFrame,
    save_dir: Union[str, Path],
    show: bool = False
):
    """
    Create comprehensive visualizations of benchmark results.

    Args:
        df_results: DataFrame with columns ['n_variables', 'sampler', 'kl_divergence', ...]
        save_dir: Directory to save plots
        show: Whether to display plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11

    # 1. Line plot: KL divergence vs number of variables
    _plot_kl_vs_variables(df_results, save_dir / 'kl_vs_variables.png', show)

    # 2. Heatmap: KL divergence across samplers and problem sizes
    _plot_kl_heatmap(df_results, save_dir / 'kl_heatmap.png', show)

    # 3. Bar plot: Average KL divergence per sampler
    _plot_average_kl_per_sampler(df_results, save_dir / 'average_kl_per_sampler.png', show)

    # 4. Box plot: KL divergence distribution per sampler
    _plot_kl_distribution(df_results, save_dir / 'kl_distribution.png', show)

    # 5. Log-scale comparison
    _plot_kl_vs_variables_logscale(df_results, save_dir / 'kl_vs_variables_logscale.png', show)

    # 6. Relative performance comparison
    _plot_relative_performance(df_results, save_dir / 'relative_performance.png', show)

    print(f"\nVisualization complete! Saved plots:")
    print(f"  - {save_dir / 'kl_vs_variables.png'}")
    print(f"  - {save_dir / 'kl_heatmap.png'}")
    print(f"  - {save_dir / 'average_kl_per_sampler.png'}")
    print(f"  - {save_dir / 'kl_distribution.png'}")
    print(f"  - {save_dir / 'kl_vs_variables_logscale.png'}")
    print(f"  - {save_dir / 'relative_performance.png'}")


def _plot_kl_vs_variables(df: pd.DataFrame, save_path: Path, show: bool):
    """Plot KL divergence vs number of variables for each sampler."""
    plt.figure(figsize=(12, 7))

    # Get unique samplers and assign colors
    samplers = df['sampler'].unique()
    colors = sns.color_palette('tab10', n_colors=len(samplers))

    for sampler, color in zip(samplers, colors):
        data = df[df['sampler'] == sampler].sort_values('n_variables')
        plt.plot(
            data['n_variables'],
            data['kl_divergence'],
            marker='o',
            linewidth=2,
            markersize=8,
            label=sampler,
            color=color
        )

    plt.xlabel('Number of Variables', fontsize=13, fontweight='bold')
    plt.ylabel('KL Divergence', fontsize=13, fontweight='bold')
    plt.title('Sampler Performance: KL Divergence vs Problem Size', fontsize=15, fontweight='bold')
    plt.legend(loc='best', frameon=True, shadow=True, fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def _plot_kl_heatmap(df: pd.DataFrame, save_path: Path, show: bool):
    """Create a heatmap of KL divergence."""
    # Pivot data for heatmap
    pivot_data = df.pivot(index='sampler', columns='n_variables', values='kl_divergence')

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.4f',
        cmap='YlOrRd',
        cbar_kws={'label': 'KL Divergence'},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.xlabel('Number of Variables', fontsize=13, fontweight='bold')
    plt.ylabel('Sampler', fontsize=13, fontweight='bold')
    plt.title('KL Divergence Heatmap: Sampler vs Problem Size', fontsize=15, fontweight='bold')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def _plot_average_kl_per_sampler(df: pd.DataFrame, save_path: Path, show: bool):
    """Bar plot of average KL divergence per sampler."""
    # Calculate average KL divergence per sampler
    avg_kl = df.groupby('sampler')['kl_divergence'].mean().sort_values()

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette('viridis', n_colors=len(avg_kl))

    bars = plt.barh(range(len(avg_kl)), avg_kl.values, color=colors)
    plt.yticks(range(len(avg_kl)), avg_kl.index)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, avg_kl.values)):
        plt.text(val, i, f' {val:.4f}', va='center', fontsize=10, fontweight='bold')

    plt.xlabel('Average KL Divergence', fontsize=13, fontweight='bold')
    plt.ylabel('Sampler', fontsize=13, fontweight='bold')
    plt.title('Average Sampler Performance (Lower is Better)', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def _plot_kl_distribution(df: pd.DataFrame, save_path: Path, show: bool):
    """Box plot showing KL divergence distribution per sampler."""
    plt.figure(figsize=(12, 7))

    # Sort samplers by median KL divergence
    sampler_order = df.groupby('sampler')['kl_divergence'].median().sort_values().index

    sns.boxplot(
        data=df,
        x='sampler',
        y='kl_divergence',
        order=sampler_order,
        palette='Set2'
    )

    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Sampler', fontsize=13, fontweight='bold')
    plt.ylabel('KL Divergence', fontsize=13, fontweight='bold')
    plt.title('KL Divergence Distribution Across Problem Sizes', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def _plot_kl_vs_variables_logscale(df: pd.DataFrame, save_path: Path, show: bool):
    """Plot KL divergence vs variables on log scale."""
    plt.figure(figsize=(12, 7))

    samplers = df['sampler'].unique()
    colors = sns.color_palette('tab10', n_colors=len(samplers))

    for sampler, color in zip(samplers, colors):
        data = df[df['sampler'] == sampler].sort_values('n_variables')
        plt.semilogy(
            data['n_variables'],
            data['kl_divergence'],
            marker='o',
            linewidth=2,
            markersize=8,
            label=sampler,
            color=color
        )

    plt.xlabel('Number of Variables', fontsize=13, fontweight='bold')
    plt.ylabel('KL Divergence (log scale)', fontsize=13, fontweight='bold')
    plt.title('Sampler Performance: KL Divergence vs Problem Size (Log Scale)', fontsize=15, fontweight='bold')
    plt.legend(loc='best', frameon=True, shadow=True, fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def _plot_relative_performance(df: pd.DataFrame, save_path: Path, show: bool):
    """Plot relative performance compared to best sampler."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # For each problem size, compute relative performance
    n_vars_list = sorted(df['n_variables'].unique())
    samplers = df['sampler'].unique()

    # Plot 1: Relative to best sampler (ratio)
    ax = axes[0]
    for sampler in samplers:
        relative_perf = []
        for n_vars in n_vars_list:
            subset = df[df['n_variables'] == n_vars]
            best_kl = subset['kl_divergence'].min()
            sampler_kl = subset[subset['sampler'] == sampler]['kl_divergence'].values
            if len(sampler_kl) > 0:
                relative_perf.append(sampler_kl[0] / best_kl)
            else:
                relative_perf.append(np.nan)

        ax.plot(n_vars_list, relative_perf, marker='o', linewidth=2, markersize=8, label=sampler)

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Best Performance')
    ax.set_xlabel('Number of Variables', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative KL Divergence (ratio to best)', fontsize=12, fontweight='bold')
    ax.set_title('Relative Performance: KL Ratio to Best Sampler', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Ranking over problem sizes
    ax = axes[1]
    rank_data = []
    for n_vars in n_vars_list:
        subset = df[df['n_variables'] == n_vars].sort_values('kl_divergence')
        for rank, row in enumerate(subset.itertuples(), 1):
            rank_data.append({
                'n_variables': n_vars,
                'sampler': row.sampler,
                'rank': rank
            })

    rank_df = pd.DataFrame(rank_data)

    for sampler in samplers:
        data = rank_df[rank_df['sampler'] == sampler].sort_values('n_variables')
        ax.plot(data['n_variables'], data['rank'], marker='o', linewidth=2, markersize=8, label=sampler)

    ax.set_xlabel('Number of Variables', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rank (1 = Best)', fontsize=12, fontweight='bold')
    ax.set_title('Sampler Ranking Across Problem Sizes', fontsize=14, fontweight='bold')
    ax.set_yticks(range(1, len(samplers) + 1))
    ax.invert_yaxis()  # Best rank (1) at top
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def create_summary_table(df: pd.DataFrame, save_path: Path):
    """
    Create a summary table of benchmark results.

    Args:
        df: DataFrame with benchmark results
        save_path: Path to save summary table (CSV)
    """
    summary_data = []

    for sampler in df['sampler'].unique():
        sampler_data = df[df['sampler'] == sampler]

        summary_data.append({
            'Sampler': sampler,
            'Mean KL Divergence': sampler_data['kl_divergence'].mean(),
            'Std KL Divergence': sampler_data['kl_divergence'].std(),
            'Min KL Divergence': sampler_data['kl_divergence'].min(),
            'Max KL Divergence': sampler_data['kl_divergence'].max(),
            'Median KL Divergence': sampler_data['kl_divergence'].median()
        })

    summary_df = pd.DataFrame(summary_data).sort_values('Mean KL Divergence')
    summary_df.to_csv(save_path, index=False)

    print(f"\nSummary table saved to: {save_path}")
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

    return summary_df
