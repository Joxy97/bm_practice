"""
GPU utilities for accelerated sampling algorithms.

Provides PyTorch-based utilities for running samplers on GPU.
"""

import torch
import numpy as np
from typing import Optional, Tuple


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Get the appropriate device for computation.

    Args:
        use_cuda: Whether to try to use CUDA

    Returns:
        torch.device: Device to use for computation
    """
    if use_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def energy_fn_to_torch(h: np.ndarray, J: np.ndarray, offset: float = 0.0,
                       device: Optional[torch.device] = None) -> callable:
    """
    Convert numpy-based energy function parameters to PyTorch energy function.

    Args:
        h: Linear biases (n_variables,)
        J: Quadratic couplings (n_variables, n_variables)
        offset: Energy offset
        device: Device to use for computation

    Returns:
        Energy function that operates on torch tensors
    """
    if device is None:
        device = get_device()

    h_torch = torch.from_numpy(h).float().to(device)
    J_torch = torch.from_numpy(J).float().to(device)

    def torch_energy_fn(states: torch.Tensor) -> torch.Tensor:
        """
        Compute energy for batch of states.

        Args:
            states: Tensor of shape (batch_size, n_variables) with values in {-1, +1}

        Returns:
            energies: Tensor of shape (batch_size,)
        """
        # E(s) = -sum_i h_i * s_i - sum_{i,j} J_ij * s_i * s_j + offset
        linear_energy = torch.matmul(states, h_torch)
        quadratic_energy = torch.sum(states * torch.matmul(states, J_torch), dim=1)
        return -(linear_energy + quadratic_energy) + offset

    return torch_energy_fn


def numpy_energy_to_torch_params(energy_fn: callable, n_variables: int,
                                  device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Extract h, J, offset from numpy energy function by probing.

    This is a helper function to convert arbitrary energy functions to torch parameters.
    Note: This assumes the energy function is of the form E(s) = -h^T s - s^T J s + offset

    Args:
        energy_fn: Energy function that accepts numpy arrays
        n_variables: Number of variables
        device: Device to use for computation

    Returns:
        h_torch: Linear biases as torch tensor
        J_torch: Quadratic couplings as torch tensor
        offset: Energy offset
    """
    if device is None:
        device = get_device()

    # Probe energy function to extract parameters
    # This is a simple extraction that assumes quadratic form

    # Get offset (energy of zero configuration)
    zero_state = np.zeros(n_variables)
    offset = energy_fn(zero_state.reshape(1, -1))[0]

    # Extract linear terms
    h = np.zeros(n_variables)
    for i in range(n_variables):
        state = np.zeros(n_variables)
        state[i] = 1.0
        E_i = energy_fn(state.reshape(1, -1))[0]
        h[i] = -(E_i - offset)

    # Extract quadratic terms
    J = np.zeros((n_variables, n_variables))
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            state = np.zeros(n_variables)
            state[i] = 1.0
            state[j] = 1.0
            E_ij = energy_fn(state.reshape(1, -1))[0]
            E_i = energy_fn(np.array([1.0 if k == i else 0.0 for k in range(n_variables)]).reshape(1, -1))[0]
            E_j = energy_fn(np.array([1.0 if k == j else 0.0 for k in range(n_variables)]).reshape(1, -1))[0]

            J_ij = -(E_ij - E_i - E_j + offset)
            J[i, j] = J_ij
            J[j, i] = J_ij

    h_torch = torch.from_numpy(h).float().to(device)
    J_torch = torch.from_numpy(J).float().to(device)

    return h_torch, J_torch, offset


def compute_energy_batch(states: torch.Tensor, h: torch.Tensor, J: torch.Tensor,
                         offset: float = 0.0) -> torch.Tensor:
    """
    Compute energies for a batch of states on GPU.

    Args:
        states: Tensor of shape (batch_size, n_variables) with values in {-1, +1}
        h: Linear biases (n_variables,)
        J: Quadratic couplings (n_variables, n_variables)
        offset: Energy offset

    Returns:
        energies: Tensor of shape (batch_size,)
    """
    linear_energy = torch.matmul(states, h)
    quadratic_energy = torch.sum(states * torch.matmul(states, J), dim=1)
    return -(linear_energy + quadratic_energy) + offset


def batch_to_numpy(states: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.

    Args:
        states: PyTorch tensor

    Returns:
        numpy array
    """
    return states.cpu().numpy()
