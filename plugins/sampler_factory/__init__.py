"""
Sampler Factory Plugin - Central registry for BM samplers and solvers.

This plugin provides a factory that creates and manages all available samplers,
returning a dictionary that can be used by the core BM pipeline, benchmark plugin,
and data generator plugin.
"""

from .sampler_factory import SamplerFactory

__all__ = ['SamplerFactory']
