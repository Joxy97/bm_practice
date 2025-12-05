"""
Sampler Factory - Central registry for creating and managing BM samplers.

This factory creates all available samplers and returns them as a dictionary,
enabling clean integration with the core BM pipeline, benchmark plugin, and
data generator plugin.
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from dimod import Sampler


class SamplerFactory:
    """
    Sampler/Solver factory for Boltzmann Machines.

    Creates and manages all available samplers, providing them as a dictionary
    for easy access by other components.

    Example:
        factory = SamplerFactory()
        sampler_dict = factory.get_sampler_dict()

        # Use in BM model
        model = BoltzmannMachine(config, sampler_dict=sampler_dict)

        # Use specific sampler
        gibbs = factory.get_sampler('gibbs')
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the sampler factory.

        Args:
            config_path: Path to sampler_factory_config.yaml (optional)
        """
        self.config = self._load_config(config_path) if config_path else {}
        self._registry: Dict[str, Sampler] = {}
        self._register_all_samplers()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _register_all_samplers(self):
        """Register all available samplers."""
        # Get default parameters from config
        defaults = self.config.get('defaults', {})
        enabled = self.config.get('enabled_samplers', {})

        # Register classical MCMC samplers
        if 'classical' not in enabled or 'gibbs' in enabled.get('classical', []):
            self._register_gibbs(defaults.get('gibbs', {}))

        if 'classical' not in enabled or 'metropolis' in enabled.get('classical', []):
            self._register_metropolis(defaults.get('metropolis', {}))

        if 'classical' not in enabled or 'parallel_tempering' in enabled.get('classical', []):
            self._register_parallel_tempering(defaults.get('parallel_tempering', {}))

        if 'classical' not in enabled or 'simulated_annealing' in enabled.get('classical', []):
            self._register_simulated_annealing()

        # Register GPU samplers
        if 'gpu' not in enabled or 'gibbs_gpu' in enabled.get('gpu', []):
            self._register_gibbs_gpu(defaults.get('gibbs_gpu', {}))

        if 'gpu' not in enabled or 'metropolis_gpu' in enabled.get('gpu', []):
            self._register_metropolis_gpu(defaults.get('metropolis_gpu', {}))

        if 'gpu' not in enabled or 'parallel_tempering_gpu' in enabled.get('gpu', []):
            self._register_parallel_tempering_gpu(defaults.get('parallel_tempering_gpu', {}))

        if 'gpu' not in enabled or 'simulated_annealing_gpu' in enabled.get('gpu', []):
            self._register_simulated_annealing_gpu()

        if 'gpu' not in enabled or 'population_annealing_gpu' in enabled.get('gpu', []):
            self._register_population_annealing_gpu()

        # Register exact samplers
        if 'exact' not in enabled or 'exact' in enabled.get('exact', []):
            self._register_exact()

        if 'exact' not in enabled or 'gumbel_max' in enabled.get('exact', []):
            self._register_gumbel_max()

        # Register optimization samplers
        if 'optimization' not in enabled or enabled.get('optimization'):
            self._register_optimization_samplers()

        # Register baseline
        if 'baseline' not in enabled or 'random' in enabled.get('baseline', []):
            self._register_random()

    def _register_gibbs(self, defaults: Dict):
        """Register Gibbs sampler."""
        try:
            from plugins.sampler_factory.samplers.classical import GibbsSamplerSpin
            self._registry['gibbs'] = GibbsSamplerSpin(
                num_sweeps=defaults.get('num_sweeps', 1000),
                burn_in=defaults.get('burn_in', 100),
                thinning=defaults.get('thinning', 1),
                randomize_order=defaults.get('randomize_order', True)
            )
        except ImportError as e:
            print(f"Warning: Could not register Gibbs sampler: {e}")

    def _register_metropolis(self, defaults: Dict):
        """Register Metropolis sampler."""
        try:
            from plugins.sampler_factory.samplers.dimod_bridge import MetropolisSampler
            self._registry['metropolis'] = MetropolisSampler(
                temperature=defaults.get('temperature', 1.0),
                num_sweeps=defaults.get('num_sweeps', 1000),
                burn_in=defaults.get('burn_in', 100),
                thinning=defaults.get('thinning', 1)
            )
        except ImportError as e:
            print(f"Warning: Could not register Metropolis sampler: {e}")

    def _register_parallel_tempering(self, defaults: Dict):
        """Register Parallel Tempering sampler."""
        try:
            from plugins.sampler_factory.samplers.dimod_bridge import ParallelTemperingSampler
            self._registry['parallel_tempering'] = ParallelTemperingSampler(
                num_replicas=defaults.get('num_replicas', 8),
                T_min=defaults.get('T_min', 1.0),
                T_max=defaults.get('T_max', 4.0),
                swap_interval=defaults.get('swap_interval', 10),
                num_sweeps=defaults.get('num_sweeps', 1000),
                burn_in=defaults.get('burn_in', 100),
                thinning=defaults.get('thinning', 1)
            )
        except ImportError as e:
            print(f"Warning: Could not register Parallel Tempering sampler: {e}")

    def _register_simulated_annealing(self):
        """Register Simulated Annealing sampler."""
        try:
            from dwave.samplers import SimulatedAnnealingSampler
            self._registry['simulated_annealing'] = SimulatedAnnealingSampler()
        except ImportError as e:
            print(f"Warning: Could not register Simulated Annealing sampler: {e}")

    def _register_gibbs_gpu(self, defaults: Dict):
        """Register GPU Gibbs sampler."""
        try:
            from plugins.sampler_factory.samplers.gpu import GibbsGPUSampler
            from plugins.sampler_factory.samplers.dimod_bridge import DimodSamplerBridge
            gpu_config = self.config.get('gpu', {})
            base_sampler = GibbsGPUSampler(
                num_sweeps=defaults.get('num_sweeps', 1000),
                burn_in=defaults.get('burn_in', 100),
                thinning=defaults.get('thinning', 1),
                randomize_order=defaults.get('randomize_order', True),
                num_chains=defaults.get('num_chains', gpu_config.get('default_num_chains', 32)),
                use_cuda=gpu_config.get('use_cuda', True)
            )
            self._registry['gibbs_gpu'] = DimodSamplerBridge(base_sampler)
        except ImportError as e:
            print(f"Warning: Could not register Gibbs GPU sampler: {e}")

    def _register_metropolis_gpu(self, defaults: Dict):
        """Register GPU Metropolis sampler."""
        try:
            from plugins.sampler_factory.samplers.gpu import MetropolisGPUSampler
            from plugins.sampler_factory.samplers.dimod_bridge import DimodSamplerBridge
            gpu_config = self.config.get('gpu', {})
            base_sampler = MetropolisGPUSampler(
                temperature=defaults.get('temperature', 1.0),
                num_sweeps=defaults.get('num_sweeps', 1000),
                burn_in=defaults.get('burn_in', 100),
                thinning=defaults.get('thinning', 1),
                num_chains=defaults.get('num_chains', gpu_config.get('default_num_chains', 32)),
                use_cuda=gpu_config.get('use_cuda', True)
            )
            self._registry['metropolis_gpu'] = DimodSamplerBridge(base_sampler)
        except ImportError as e:
            print(f"Warning: Could not register Metropolis GPU sampler: {e}")

    def _register_parallel_tempering_gpu(self, defaults: Dict):
        """Register GPU Parallel Tempering sampler."""
        try:
            from plugins.sampler_factory.samplers.gpu import ParallelTemperingGPUSampler
            from plugins.sampler_factory.samplers.dimod_bridge import DimodSamplerBridge
            gpu_config = self.config.get('gpu', {})
            base_sampler = ParallelTemperingGPUSampler(
                num_replicas=defaults.get('num_replicas', 8),
                T_min=defaults.get('T_min', 1.0),
                T_max=defaults.get('T_max', 4.0),
                swap_interval=defaults.get('swap_interval', 10),
                num_sweeps=defaults.get('num_sweeps', 1000),
                burn_in=defaults.get('burn_in', 100),
                thinning=defaults.get('thinning', 1),
                use_cuda=gpu_config.get('use_cuda', True)
            )
            self._registry['parallel_tempering_gpu'] = DimodSamplerBridge(base_sampler)
        except ImportError as e:
            print(f"Warning: Could not register Parallel Tempering GPU sampler: {e}")

    def _register_simulated_annealing_gpu(self):
        """Register GPU Simulated Annealing sampler."""
        try:
            from plugins.sampler_factory.samplers.gpu import SimulatedAnnealingGPUSampler
            from plugins.sampler_factory.samplers.dimod_bridge import DimodSamplerBridge
            gpu_config = self.config.get('gpu', {})
            base_sampler = SimulatedAnnealingGPUSampler(
                beta_range=(1.0, 10.0),
                proposal_acceptance_criteria="Metropolis",
                num_sweeps=1000,
                num_chains=gpu_config.get('default_num_chains', 32),
                use_cuda=gpu_config.get('use_cuda', True)
            )
            self._registry['simulated_annealing_gpu'] = DimodSamplerBridge(base_sampler)
        except ImportError as e:
            print(f"Warning: Could not register Simulated Annealing GPU sampler: {e}")

    def _register_population_annealing_gpu(self):
        """Register GPU Population Annealing sampler."""
        try:
            from plugins.sampler_factory.samplers.gpu import PopulationAnnealingGPUSampler
            from plugins.sampler_factory.samplers.dimod_bridge import DimodSamplerBridge
            gpu_config = self.config.get('gpu', {})
            base_sampler = PopulationAnnealingGPUSampler(
                population_size=1000,
                num_sweeps=100,
                beta_min=0.1,
                beta_max=10.0,
                resample_threshold=0.5,
                use_cuda=gpu_config.get('use_cuda', True)
            )
            self._registry['population_annealing_gpu'] = DimodSamplerBridge(base_sampler)
        except ImportError as e:
            print(f"Warning: Could not register Population Annealing GPU sampler: {e}")

    def _register_exact(self):
        """Register Exact sampler."""
        try:
            from plugins.sampler_factory.samplers.dimod_bridge import ExactSamplerBridge
            self._registry['exact'] = ExactSamplerBridge(max_variables=20)
        except ImportError as e:
            print(f"Warning: Could not register Exact sampler: {e}")

    def _register_gumbel_max(self):
        """Register Gumbel-max sampler."""
        try:
            from plugins.sampler_factory.samplers.dimod_bridge import GumbelMaxSampler
            self._registry['gumbel_max'] = GumbelMaxSampler(max_variables=20)
        except ImportError as e:
            print(f"Warning: Could not register Gumbel-max sampler: {e}")

    def _register_optimization_samplers(self):
        """Register optimization samplers (local search)."""
        try:
            from dwave.samplers import SteepestDescentSampler, TabuSampler
            self._registry['steepest_descent'] = SteepestDescentSampler()
            self._registry['tabu'] = TabuSampler()
        except ImportError as e:
            print(f"Warning: Could not register optimization samplers: {e}")

        try:
            from dwave.samplers import GreedySampler
            self._registry['greedy'] = GreedySampler()
        except ImportError:
            pass  # Greedy not available in all D-Wave versions

    def _register_random(self):
        """Register Random sampler."""
        try:
            from dimod import RandomSampler
            self._registry['random'] = RandomSampler()
        except ImportError as e:
            print(f"Warning: Could not register Random sampler: {e}")

    def get_sampler_dict(self) -> Dict[str, Sampler]:
        """
        Get dictionary of all registered samplers.

        Returns:
            Dictionary mapping sampler names to sampler instances
        """
        return self._registry

    def get_sampler(self, name: str) -> Sampler:
        """
        Get specific sampler by name.

        Args:
            name: Sampler name

        Returns:
            Sampler instance

        Raises:
            ValueError: If sampler not found
        """
        if name not in self._registry:
            available = self.list_samplers()
            raise ValueError(
                f"Sampler '{name}' not found. Available samplers: {available}"
            )
        return self._registry[name]

    def list_samplers(self) -> List[str]:
        """
        List all available sampler names.

        Returns:
            List of sampler names
        """
        return sorted(list(self._registry.keys()))

    def __repr__(self) -> str:
        """String representation."""
        return f"SamplerFactory({len(self._registry)} samplers registered)"

    def summary(self) -> str:
        """Get detailed summary of available samplers."""
        samplers_by_category = {
            'Classical MCMC': [],
            'GPU Accelerated': [],
            'Exact': [],
            'Optimization': [],
            'Baseline': []
        }

        for name in sorted(self._registry.keys()):
            if 'gpu' in name:
                samplers_by_category['GPU Accelerated'].append(name)
            elif name in ['exact', 'gumbel_max']:
                samplers_by_category['Exact'].append(name)
            elif name in ['steepest_descent', 'tabu', 'greedy', 'simulated_annealing', 'simulated_annealing_gpu']:
                samplers_by_category['Optimization'].append(name)
            elif name == 'random':
                samplers_by_category['Baseline'].append(name)
            else:
                samplers_by_category['Classical MCMC'].append(name)

        summary = f"""
Sampler Factory Summary
{'='*60}
Total Samplers: {len(self._registry)}

Classical MCMC Samplers ({len(samplers_by_category['Classical MCMC'])}):
  {', '.join(samplers_by_category['Classical MCMC']) or 'None'}

GPU Accelerated Samplers ({len(samplers_by_category['GPU Accelerated'])}):
  {', '.join(samplers_by_category['GPU Accelerated']) or 'None'}

Exact Samplers ({len(samplers_by_category['Exact'])}):
  {', '.join(samplers_by_category['Exact']) or 'None'}

Optimization Samplers ({len(samplers_by_category['Optimization'])}):
  {', '.join(samplers_by_category['Optimization']) or 'None'}

Baseline Samplers ({len(samplers_by_category['Baseline'])}):
  {', '.join(samplers_by_category['Baseline']) or 'None'}
{'='*60}
"""
        return summary


# Convenience function for backward compatibility
def create_sampler(sampler_type: str, params: Optional[Dict[str, Any]] = None) -> Sampler:
    """
    Create a sampler (legacy function for backward compatibility).

    Args:
        sampler_type: Type of sampler
        params: Sampler parameters (currently ignored, uses defaults)

    Returns:
        Sampler instance

    Note:
        This function is kept for backward compatibility.
        Prefer using SamplerFactory.get_sampler() for new code.
    """
    factory = SamplerFactory()
    return factory.get_sampler(sampler_type)
