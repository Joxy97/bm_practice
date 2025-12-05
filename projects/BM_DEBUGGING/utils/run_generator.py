"""
Standalone Data Generator CLI for BM_DEBUGGING.

Usage:
  python projects/BM_DEBUGGING/utils/run_generator.py --config projects/BM_DEBUGGING/utils/data_generator_config.yaml
"""

import sys
import argparse
import yaml
from pathlib import Path

# Ensure repo root is importable
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from projects.BM_DEBUGGING.utils.data_generator import SyntheticDataGenerator
from plugins.sampler_factory import SamplerFactory


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Synthetic data generator for BM_DEBUGGING",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data with default config
  python projects/BM_DEBUGGING/utils/run_generator.py \\
    --config projects/BM_DEBUGGING/utils/data_generator_config.yaml

  # Override output directory
  python projects/BM_DEBUGGING/utils/run_generator.py \\
    --config projects/BM_DEBUGGING/utils/data_generator_config.yaml \\
    --output-dir projects/BM_DEBUGGER/data
"""
    )

    parser.add_argument(
        '--config',
        type=str,
        default='projects/BM_DEBUGGING/utils/data_generator_config.yaml',
        help='Path to data_generator_config.yaml'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Optional override for dataset base directory'
    )

    args = parser.parse_args()

    try:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        if args.output_dir:
            config.setdefault('paths', {})
            config['paths']['output_dir'] = args.output_dir

        print("\nInitializing sampler factory...")
        factory = SamplerFactory()
        sampler_dict = factory.get_sampler_dict()
        print(f"  Registered {len(sampler_dict)} samplers")

        generator = SyntheticDataGenerator(config, sampler_dict)

        base_output_dir = config.get('paths', {}).get(
            'output_dir', 'projects/BM_DEBUGGER/data'
        )
        df, dataset_dir = generator.generate(base_output_dir)

        print(f"\n[OK] Data generation complete!")
        print(f"  Samples: {len(df)}")
        print(f"  Dataset directory: {dataset_dir}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
