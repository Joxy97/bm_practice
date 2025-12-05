"""
Standalone Data Generator - CLI for generating synthetic BM data.

Usage:
    python -m plugins.data_generator.run_generator --config plugins/data_generator/data_generator_config.yaml --output-dir outputs/data
"""

import sys
import argparse
import yaml
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plugins.data_generator import SyntheticDataGenerator
from plugins.sampler_factory import SamplerFactory


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Synthetic Data Generator for Boltzmann Machines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data with default config
  python -m plugins.data_generator.run_generator \\
    --config plugins/data_generator/data_generator_config.yaml

  # Specify output directory
  python -m plugins.data_generator.run_generator \\
    --config plugins/data_generator/data_generator_config.yaml \\
    --output-dir my_data/
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to data_generator_config.yaml'
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

        # Initialize sampler factory
        print("\nInitializing sampler factory...")
        factory = SamplerFactory()
        sampler_dict = factory.get_sampler_dict()
        print(f"  Registered {len(sampler_dict)} samplers")

        # Create generator
        generator = SyntheticDataGenerator(config, sampler_dict)

        # Generate data
        output_dir = config.get('paths', {}).get('output_dir', 'plugins/data_generator/datasets')
        df = generator.generate(output_dir)

        print(f"\n✓ Data generation complete!")
        print(f"  Samples: {len(df)}")
        print(f"  Output: {output_dir}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
