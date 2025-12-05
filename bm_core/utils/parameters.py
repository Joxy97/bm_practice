"""
Utilities for generating and managing Boltzmann Machine parameters.
"""

import numpy as np
from typing import Dict, Tuple, List


def generate_random_parameters(
    nodes: List,
    edges: List[Tuple],
    seed: int = 42,
    linear_scale: float = 1.0,
    quadratic_scale: float = 1.0
) -> Tuple[Dict, Dict]:
    """
    Generate random linear and quadratic parameters.

    Args:
        nodes: List of node identifiers
        edges: List of edge tuples
        seed: Random seed for reproducibility
        linear_scale: Scale for linear biases
        quadratic_scale: Scale for quadratic biases

    Returns:
        linear: Dictionary {node: bias}
        quadratic: Dictionary {(node1, node2): weight}
    """
    rng = np.random.RandomState(seed)

    # Generate linear biases: uniform in [-linear_scale, linear_scale]
    linear = {node: linear_scale * (2 * rng.random() - 1) for node in nodes}

    # Generate quadratic biases: uniform in [-quadratic_scale, quadratic_scale]
    quadratic = {edge: quadratic_scale * (2 * rng.random() - 1) for edge in edges}

    return linear, quadratic
