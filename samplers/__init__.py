"""
Samplers module for Boltzmann Machine sampling.

Provides a unified interface for various sampling algorithms including
classical MCMC, exact methods, and quantum annealing.
"""

from .base import BaseSampler, register_sampler, get_sampler, list_samplers
from .classical import (
    GibbsSampler,
    MetropolisSampler,
    ParallelTemperingSampler,
    SimulatedAnnealingSampler,
    RandomSampler,
    ExactSampler,
    GumbelMaxSampler
)

# Import advanced samplers (registers them but not exported)
from . import advanced

__all__ = [
    'BaseSampler',
    'register_sampler',
    'get_sampler',
    'list_samplers',
    'GibbsSampler',
    'MetropolisSampler',
    'ParallelTemperingSampler',
    'SimulatedAnnealingSampler',
    'RandomSampler',
    'ExactSampler',
    'GumbelMaxSampler'
]
