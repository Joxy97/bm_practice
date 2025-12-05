"""
Standalone Sampler Benchmark - CLI for benchmarking BM samplers.

Usage:
    python -m plugins.sampler_benchmark.run_benchmark --config plugins/sampler_benchmark/benchmark_config.yaml --output-dir outputs/benchmark
"""

import sys
import argparse
import yaml
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plugins.sampler_benchmark import SamplerBenchmark
from plugins.sampler_factory import SamplerFactory


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sampler Benchmark for Boltzmann Machines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with default config
  python -m plugins.sampler_benchmark.run_benchmark \\
    --config plugins/sampler_benchmark/benchmark_config.yaml

  # Specify output directory
  python -m plugins.sampler_benchmark.run_benchmark \\
    --config plugins/sampler_benchmark/benchmark_config.yaml \\
    --output-dir outputs/benchmark
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to benchmark_config.yaml'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )

    args = parser.parse_args()

    try:
        # Load config
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        # Override output dir if specified
        if args.output_dir:
            if 'paths' not in config:
                config['paths'] = {}
            config['paths']['output_dir'] = args.output_dir
            config['paths']['plot_dir'] = args.output_dir

        # Initialize sampler factory
        print("\nInitializing sampler factory...")
        factory = SamplerFactory()
        sampler_dict = factory.get_sampler_dict()
        print(f"  Registered {len(sampler_dict)} samplers")

        # Create benchmark
        benchmark = SamplerBenchmark(config, sampler_dict)

        # Run benchmark suite
        results = benchmark.run_benchmark()

        # Print summary
        benchmark.print_summary()

        # Save results
        if config['output'].get('save_results', True):
            output_dir = config.get('paths', {}).get('output_dir', 'plugins/sampler_benchmark/results')
            benchmark.save_results(output_dir)

        # Create visualizations
        if config['output'].get('save_plots', True):
            plot_dir = config.get('paths', {}).get('plot_dir', 'plugins/sampler_benchmark/plots')
            plot_format = config['output'].get('plot_format', 'png')
            benchmark.create_visualizations(plot_dir, plot_format)

        print(f"\n[OK] Benchmark complete!")
        print(f"  Results: {output_dir}")
        print(f"  Plots: {plot_dir}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
