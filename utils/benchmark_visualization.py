"""
Benchmark Visualization Module

Creates comprehensive visualizations for sampler benchmark results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, List
from utils.benchmark_metrics import get_metric_display_name


def plot_benchmark_results(
    df_results: pd.DataFrame,
    save_dir: Union[str, Path],
    show: bool = False
):
    """
    Create comprehensive visualizations of benchmark results.

    Args:
        df_results: DataFrame with columns ['n_variables', 'sampler', <metrics>, ...]
        save_dir: Directory to save plots
        show: Whether to display plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11

    # Detect available metrics
    metric_columns = [col for col in df_results.columns
                     if col not in ['n_variables', 'sampler', 'n_samples',
                                   'n_unique_states_true', 'n_unique_states_empirical']]

    print(f"\nDetected metrics: {metric_columns}")

    # Create plots for each metric
    plot_files = []
    for metric in metric_columns:
        # 1. Average metric per sampler
        plot_file = save_dir / f'average_{metric}_per_sampler.png'
        _plot_average_metric_per_sampler(df_results, metric, plot_file, show)
        plot_files.append(plot_file)

    # 2. Combined metrics comparison plot
    if len(metric_columns) > 0:
        combined_file = save_dir / 'combined_metrics_comparison.png'
        _plot_combined_metrics(df_results, metric_columns, combined_file, show)
        plot_files.append(combined_file)

    print(f"\nVisualization complete! Saved plots:")
    for plot_file in plot_files:
        print(f"  - {plot_file}")


def _plot_average_metric_per_sampler(
    df: pd.DataFrame,
    metric: str,
    save_path: Path,
    show: bool
):
    """Bar plot of average metric value per sampler."""
    # Calculate average metric per sampler
    avg_metric = df.groupby('sampler')[metric].mean().sort_values()

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette('viridis', n_colors=len(avg_metric))

    bars = plt.barh(range(len(avg_metric)), avg_metric.values, color=colors)
    plt.yticks(range(len(avg_metric)), avg_metric.index)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, avg_metric.values)):
        plt.text(val, i, f' {val:.4f}', va='center', fontsize=10, fontweight='bold')

    metric_display = get_metric_display_name(metric)
    plt.xlabel(f'Average {metric_display}', fontsize=13, fontweight='bold')
    plt.ylabel('Sampler', fontsize=13, fontweight='bold')
    plt.title(f'Average Sampler Performance: {metric_display} (Lower is Better)',
              fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def _plot_combined_metrics(
    df: pd.DataFrame,
    metrics: List[str],
    save_path: Path,
    show: bool
):
    """
    Create a combined visualization showing all metrics side-by-side.

    Args:
        df: DataFrame with benchmark results
        metrics: List of metric column names
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    # Get unique samplers and colors
    samplers = df['sampler'].unique()
    colors = sns.color_palette('tab10', n_colors=len(samplers))
    color_map = dict(zip(samplers, colors))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Calculate average metric per sampler
        avg_metric = df.groupby('sampler')[metric].mean().sort_values()

        # Create bar plot
        bars = ax.barh(
            range(len(avg_metric)),
            avg_metric.values,
            color=[color_map[sampler] for sampler in avg_metric.index]
        )

        ax.set_yticks(range(len(avg_metric)))
        ax.set_yticklabels(avg_metric.index)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, avg_metric.values)):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=9, fontweight='bold')

        metric_display = get_metric_display_name(metric)
        ax.set_xlabel(metric_display, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_display}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle('Sampler Performance Comparison Across All Metrics (Lower is Better)',
                 fontsize=14, fontweight='bold', y=1.02)
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
    Create a summary table of benchmark results with all metrics.

    Args:
        df: DataFrame with benchmark results
        save_path: Path to save summary table (CSV)
    """
    # Detect metric columns
    metric_columns = [col for col in df.columns
                     if col not in ['n_variables', 'sampler', 'n_samples',
                                   'n_unique_states_true', 'n_unique_states_empirical']]

    summary_data = []

    for sampler in df['sampler'].unique():
        sampler_data = df[df['sampler'] == sampler]

        row = {'Sampler': sampler}

        # Add statistics for each metric
        for metric in metric_columns:
            metric_display = get_metric_display_name(metric)
            row[f'{metric_display} (Mean)'] = sampler_data[metric].mean()
            row[f'{metric_display} (Std)'] = sampler_data[metric].std()

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    # Sort by first metric (usually KL divergence)
    if len(metric_columns) > 0:
        first_metric_col = f'{get_metric_display_name(metric_columns[0])} (Mean)'
        if first_metric_col in summary_df.columns:
            summary_df = summary_df.sort_values(first_metric_col)

    summary_df.to_csv(save_path, index=False)

    print(f"\nSummary table saved to: {save_path}")
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100)

    return summary_df
