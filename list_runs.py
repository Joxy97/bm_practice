"""
Utility script to list and view past training runs.

Usage:
    python list_runs.py              # List all runs
    python list_runs.py --latest     # Show latest run details
"""

import argparse
import os
from utils import list_runs, get_latest_run, print_run_summary


def main():
    parser = argparse.ArgumentParser(description="View training runs")
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Filter by dataset name'
    )
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Show details of latest run'
    )
    parser.add_argument(
        '--run',
        type=str,
        default=None,
        help='Show details of specific run directory'
    )

    args = parser.parse_args()

    base_output_dir = 'outputs'

    if args.run:
        # Show specific run
        run_path = os.path.join(base_output_dir, args.run)
        print_run_summary(run_path)

    elif args.latest:
        # Show latest run
        latest = get_latest_run(base_output_dir, args.dataset)
        if latest:
            print_run_summary(latest)
        else:
            print("No runs found.")

    else:
        # List all runs
        runs = list_runs(base_output_dir, args.dataset)

        if not runs:
            print("No runs found.")
            return

        print(f"\n{'='*70}")
        print(f"TRAINING RUNS ({len(runs)} total)")
        print(f"{'='*70}")

        for i, run in enumerate(runs, 1):
            run_path = os.path.join(base_output_dir, run)

            # Check for config
            config_exists = os.path.exists(os.path.join(run_path, 'config.yaml'))

            # Count files in each directory
            subdirs = ['data', 'models', 'checkpoints', 'plots']
            file_counts = {}
            for subdir in subdirs:
                subdir_path = os.path.join(run_path, subdir)
                if os.path.exists(subdir_path):
                    file_counts[subdir] = len([f for f in os.listdir(subdir_path)
                                               if os.path.isfile(os.path.join(subdir_path, f))])
                else:
                    file_counts[subdir] = 0

            print(f"\n{i}. {run}")
            print(f"   Config: {'[OK]' if config_exists else '[MISSING]'}")
            print(f"   Files: data={file_counts['data']}, models={file_counts['models']}, "
                  f"checkpoints={file_counts['checkpoints']}, plots={file_counts['plots']}")

        print(f"\n{'='*70}")
        print(f"\nTo view details: python list_runs.py --latest")
        print(f"                 python list_runs.py --run {runs[0]}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
