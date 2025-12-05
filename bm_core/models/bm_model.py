"""
Boltzmann Machine abstraction layer.

This module provides a clean abstraction over D-Wave's GRBM implementation,
decoupling the core pipeline from specific BM implementations and enabling
future backend swapping.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine as GRBM


class BoltzmannMachine:
    """
    Boltzmann Machine abstraction layer.

    Wraps D-Wave's GraphRestrictedBoltzmannMachine to provide a clean,
    implementation-independent API for the core BM pipeline.

    This abstraction enables:
    - Decoupling from D-Wave GRBM implementation details
    - Future backend swapping (PyTorch native, TensorFlow, JAX, etc.)
    - Cleaner API for users
    - Easier testing and mocking

    Attributes:
        config: GRBM configuration object
        sampler_dict: Dictionary of available samplers {name: sampler_instance}
        _backend: Underlying GRBM implementation (D-Wave)
        nodes: List of all node identifiers
        edges: List of edge tuples (u, v)
        hidden_nodes: List of hidden node identifiers (None for FVBM)
        visible_idx: Indices of visible units
    """

    def __init__(
        self,
        nodes: List[int],
        edges: List[Tuple[int, int]],
        hidden_nodes: Optional[List[int]] = None,
        linear: Optional[Dict[int, float]] = None,
        quadratic: Optional[Dict[Tuple[int, int], float]] = None,
        sampler_dict: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Boltzmann Machine.

        Args:
            nodes: List of all node identifiers
            edges: List of edge tuples (u, v)
            hidden_nodes: List of hidden node identifiers (None for FVBM)
            linear: Dictionary of linear biases {node: bias} (None for random init)
            quadratic: Dictionary of quadratic weights {(u,v): weight} (None for random init)
            sampler_dict: Dictionary of available samplers {name: sampler_instance}
        """
        self.nodes = nodes
        self.edges = edges
        self.hidden_nodes = hidden_nodes if hidden_nodes else None
        self.sampler_dict = sampler_dict if sampler_dict else {}

        # Create underlying GRBM backend
        self._backend = self._create_grbm_backend(linear, quadratic)

        # Cache visible indices for quick access
        self.visible_idx = self._backend.visible_idx

        # Store dimensions
        self.n_visible = len(self.visible_idx)
        self.n_hidden = len(hidden_nodes) if hidden_nodes else 0
        self.n_total = len(nodes)

    def _create_grbm_backend(
        self,
        linear: Optional[Dict[int, float]],
        quadratic: Optional[Dict[Tuple[int, int], float]]
    ) -> GRBM:
        """
        Create underlying D-Wave GRBM backend.

        Args:
            linear: Linear biases (None for default initialization)
            quadratic: Quadratic weights (None for default initialization)

        Returns:
            GRBM instance
        """
        grbm = GRBM(
            nodes=self.nodes,
            edges=self.edges,
            hidden_nodes=self.hidden_nodes,
            linear=linear,
            quadratic=quadratic
        )

        return grbm

    def sample(
        self,
        sampler_name: str,
        prefactor: float = 1.0,
        sample_params: Optional[Dict[str, Any]] = None,
        as_tensor: bool = True
    ) -> torch.Tensor:
        """
        Sample from the Boltzmann Machine using specified sampler.

        Args:
            sampler_name: Name of sampler from sampler_dict
            prefactor: Temperature scaling factor (inverse temperature / beta)
            sample_params: Sampler-specific parameters (num_reads, etc.)
            as_tensor: Return PyTorch tensor (True) or numpy array (False)

        Returns:
            Samples tensor of shape (num_samples, n_visible + n_hidden)

        Raises:
            ValueError: If sampler_name not in sampler_dict
        """
        if sampler_name not in self.sampler_dict:
            available = list(self.sampler_dict.keys())
            raise ValueError(
                f"Sampler '{sampler_name}' not found in sampler_dict. "
                f"Available samplers: {available}"
            )

        sampler = self.sampler_dict[sampler_name]
        sample_params = sample_params if sample_params else {}

        samples = self._backend.sample(
            sampler,
            prefactor=prefactor,
            sample_params=sample_params,
            as_tensor=as_tensor
        )

        return samples

    def quasi_objective(
        self,
        data_samples: torch.Tensor,
        model_samples: torch.Tensor,
        kind: Optional[str] = None,
        prefactor: float = 1.0,
        sampler_name: Optional[str] = None
    ) -> torch.Tensor:
        """
        Compute quasi-objective (loss) for training.

        The quasi-objective estimates the gradient of the log-likelihood
        using samples from the data distribution and model distribution.

        Args:
            data_samples: Samples from data distribution (visible units)
            model_samples: Samples from model distribution
            kind: Hidden unit treatment: "exact-disc", "sampling", or None
            prefactor: Temperature scaling factor
            sampler_name: Sampler to use if kind="sampling"

        Returns:
            Loss tensor (scalar)

        Raises:
            ValueError: If kind="sampling" but sampler_name not provided
        """
        # Get sampler if needed
        sampler = None
        if kind == "sampling":
            if sampler_name is None:
                raise ValueError("sampler_name required when kind='sampling'")
            if sampler_name not in self.sampler_dict:
                raise ValueError(f"Sampler '{sampler_name}' not in sampler_dict")
            sampler = self.sampler_dict[sampler_name]

        loss = self._backend.quasi_objective(
            data_samples,
            model_samples,
            kind=kind,
            prefactor=prefactor,
            sampler=sampler
        )

        return loss

    def get_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get model parameters.

        Returns:
            Tuple of (linear, quadratic) tensors
        """
        return self._backend.linear, self._backend.quadratic

    def set_parameters(
        self,
        linear: torch.Tensor,
        quadratic: torch.Tensor
    ):
        """
        Set model parameters.

        Args:
            linear: Linear bias tensor
            quadratic: Quadratic weight tensor
        """
        self._backend._linear.data = linear
        self._backend._quadratic.data = quadratic

    def get_parameter_dict(self) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float]]:
        """
        Get parameters as dictionaries (node -> bias, edge -> weight).

        Returns:
            Tuple of (linear_dict, quadratic_dict)
        """
        linear_tensor, quadratic_tensor = self.get_parameters()

        # Convert to dictionaries
        linear_dict = {
            node: linear_tensor[i].item()
            for i, node in enumerate(self.nodes)
        }

        quadratic_dict = {
            edge: quadratic_tensor[i].item()
            for i, edge in enumerate(self.edges)
        }

        return linear_dict, quadratic_dict

    def to(self, device: torch.device) -> 'BoltzmannMachine':
        """
        Move model to specified device.

        Args:
            device: PyTorch device (CPU or CUDA)

        Returns:
            Self (for chaining)
        """
        self._backend = self._backend.to(device)
        return self

    def parameters(self):
        """
        Get model parameters (for optimizer).

        Returns:
            Iterator over model parameters
        """
        return self._backend.parameters()

    def train(self):
        """Set model to training mode."""
        self._backend.train()

    def eval(self):
        """Set model to evaluation mode."""
        self._backend.eval()

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int = 0,
        optimizer_state: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dict (optional)
            metadata: Additional metadata to save (optional)
        """
        checkpoint = {
            'model_state_dict': self._backend.state_dict(),
            'nodes': self.nodes,
            'edges': self.edges,
            'hidden_nodes': self.hidden_nodes,
            'epoch': epoch,
            'n_visible': self.n_visible,
            'n_hidden': self.n_hidden,
        }

        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        if metadata is not None:
            checkpoint['metadata'] = metadata

        torch.save(checkpoint, filepath)

    @classmethod
    def load_checkpoint(
        cls,
        filepath: str,
        sampler_dict: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ) -> 'BoltzmannMachine':
        """
        Load model from checkpoint.

        Args:
            filepath: Path to checkpoint file
            sampler_dict: Dictionary of available samplers
            device: Device to load model to (None for CPU)

        Returns:
            BoltzmannMachine instance loaded from checkpoint
        """
        if device is None:
            device = torch.device('cpu')

        checkpoint = torch.load(filepath, map_location=device)

        # Create model with same topology
        model = cls(
            nodes=checkpoint['nodes'],
            edges=checkpoint['edges'],
            hidden_nodes=checkpoint['hidden_nodes'],
            sampler_dict=sampler_dict
        )

        # Load state dict
        model._backend.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        return model

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"BoltzmannMachine("
            f"n_visible={self.n_visible}, "
            f"n_hidden={self.n_hidden}, "
            f"n_edges={len(self.edges)}, "
            f"backend='D-Wave GRBM')"
        )

    def summary(self) -> str:
        """
        Get detailed model summary.

        Returns:
            Formatted summary string
        """
        linear, quadratic = self.get_parameters()

        model_type = "FVBM" if self.n_hidden == 0 else "RBM/SBM"

        summary = f"""
Boltzmann Machine Summary
{'='*50}
Architecture:
  Model Type: {model_type}
  Visible Units: {self.n_visible}
  Hidden Units: {self.n_hidden}
  Total Nodes: {self.n_total}
  Edges: {len(self.edges)}

Parameters:
  Linear Bias Range: [{linear.min():.4f}, {linear.max():.4f}]
  Quadratic Weight Range: [{quadratic.min():.4f}, {quadratic.max():.4f}]

Backend:
  Implementation: D-Wave GraphRestrictedBoltzmannMachine

Available Samplers: {list(self.sampler_dict.keys()) if self.sampler_dict else 'None'}
{'='*50}
"""
        return summary
