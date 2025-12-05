"""
Device management utilities for GPU/CPU.
"""

import torch
from typing import Optional, Union


def get_device(config: Optional[dict] = None) -> torch.device:
    """
    Get the appropriate device based on configuration.

    Args:
        config: Configuration dictionary with 'device' settings

    Returns:
        torch.device object
    """
    if config is None:
        # Default to auto-detection
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device_config = config.get('device', {})
    use_cuda = device_config.get('use_cuda', 'auto')

    if use_cuda == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif use_cuda == 'cpu':
        device = torch.device('cpu')
    elif use_cuda == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
    elif isinstance(use_cuda, str) and use_cuda.startswith('cuda:'):
        # Specific GPU device like "cuda:0"
        if torch.cuda.is_available():
            device = torch.device(use_cuda)
        else:
            print(f"WARNING: {use_cuda} requested but CUDA not available. Falling back to CPU.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    return device


def print_device_info(device: torch.device):
    """
    Print information about the device being used.

    Args:
        device: torch.device object
    """
    print(f"\n{'='*70}")
    print("DEVICE INFORMATION")
    print(f"{'='*70}")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(device)}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
    else:
        print("Running on CPU")
        if torch.cuda.is_available():
            print(f"Note: CUDA is available but not being used")
            print(f"Available GPUs: {torch.cuda.device_count()}")

    print(f"{'='*70}\n")


def move_to_device(obj, device: torch.device):
    """
    Move an object (tensor, model, or dict/list of tensors) to device.

    Args:
        obj: Object to move
        device: Target device

    Returns:
        Object on the target device
    """
    if obj is None:
        return None
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    else:
        return obj


def set_device_seeds(seed: int, device: torch.device):
    """
    Set random seeds for both CPU and GPU.

    Args:
        seed: Random seed value
        device: Device being used
    """
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
