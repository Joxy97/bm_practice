"""
Base sampler interface and registry for Boltzmann Machine samplers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable
import numpy as np


class BaseSampler(ABC):
    """
    Abstract base class for all Boltzmann Machine samplers.

    All samplers must implement the sample() method which generates
    samples from a given energy function.
    """

    def __init__(self, **params):
        """
        Initialize sampler with parameters.

        Args:
            **params: Sampler-specific parameters
        """
        self.params = params

    @abstractmethod
    def sample(
        self,
        energy_fn: Callable[[np.ndarray], np.ndarray],
        n_variables: int,
        num_samples: int,
        **kwargs
    ) -> np.ndarray:
        """
        Generate samples from the Boltzmann distribution.

        Args:
            energy_fn: Function that computes energy E(x) for states x
                      Input: (batch_size, n_variables) array
                      Output: (batch_size,) array of energies
            n_variables: Number of binary variables
            num_samples: Number of samples to generate
            **kwargs: Additional sampler-specific arguments

        Returns:
            samples: Array of shape (num_samples, n_variables) with values in {-1, +1}
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this sampler.

        Returns:
            Dictionary with sampler name, type, and parameters
        """
        return {
            'name': self.__class__.__name__,
            'params': self.params
        }


# Global sampler registry
_SAMPLER_REGISTRY: Dict[str, type] = {}


def register_sampler(name: str):
    """
    Decorator to register a sampler class.

    Usage:
        @register_sampler('my_sampler')
        class MySampler(BaseSampler):
            ...

    Args:
        name: Name to register the sampler under
    """
    def decorator(cls):
        if not issubclass(cls, BaseSampler):
            raise TypeError(f"{cls.__name__} must inherit from BaseSampler")
        _SAMPLER_REGISTRY[name] = cls
        return cls
    return decorator


def get_sampler(name: str, **params) -> BaseSampler:
    """
    Get a sampler instance by name.

    Args:
        name: Name of the sampler
        **params: Parameters to pass to the sampler constructor

    Returns:
        Sampler instance

    Raises:
        ValueError: If sampler name is not registered
    """
    if name not in _SAMPLER_REGISTRY:
        available = list(_SAMPLER_REGISTRY.keys())
        raise ValueError(
            f"Unknown sampler '{name}'. Available samplers: {available}"
        )

    sampler_class = _SAMPLER_REGISTRY[name]
    return sampler_class(**params)


def list_samplers() -> Dict[str, type]:
    """
    List all registered samplers.

    Returns:
        Dictionary mapping sampler names to their classes
    """
    return dict(_SAMPLER_REGISTRY)
