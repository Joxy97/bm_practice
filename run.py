"""
Entry point script to run from parent directory.

Usage from bm_practice/:
    python bm_pipeline/run.py --mode [generate|train|full] --config bm_pipeline/configs/config.yaml

Or:
    python -m bm_pipeline.run --mode [generate|train|full] --config bm_pipeline/configs/config.yaml
"""

import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run main
from main import main

if __name__ == "__main__":
    main()
