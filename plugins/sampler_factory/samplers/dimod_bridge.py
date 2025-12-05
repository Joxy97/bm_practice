"""
Bridge module to make custom samplers compatible with dimod.Sampler interface.

Wraps energy-function-based samplers to work with BinaryQuadraticModel objects.
"""

import numpy as np
from typing import Optional
from dimod import BinaryQuadraticModel, SampleSet, Sampler
import dimod

from .classical import (
    MetropolisSampler as MetropolisSamplerBase,
    ParallelTemperingSampler as ParallelTemperingSamplerBase,
    ExactSampler as ExactSamplerBase,
    GumbelMaxSampler as GumbelMaxSamplerBase
)


class DimodSamplerBridge(Sampler):
    """
    Base class for bridging custom samplers to dimod interface.

    Converts BinaryQuadraticModel to energy function and back.
    """

    def __init__(self, base_sampler, **params):
        """
        Initialize bridge.

        Args:
            base_sampler: Instance of BaseSampler to wrap
            **params: Additional parameters
        """
        self.base_sampler = base_sampler
        self._params = params

    @property
    def properties(self):
        return {
            'description': f'{self.base_sampler.__class__.__name__} (dimod bridge)'
        }

    @property
    def parameters(self):
        return self.base_sampler.params

    def _bqm_to_energy_fn(self, bqm: BinaryQuadraticModel):
        """
        Convert BinaryQuadraticModel to energy function.

        Args:
            bqm: Binary quadratic model (SPIN vartype)

        Returns:
            energy_fn: Function that computes E(x) for states x in {-1,+1}^N
            variables: List of variable names in order
        """
        # Convert to SPIN if not already
        if bqm.vartype != dimod.SPIN:
            bqm = bqm.change_vartype(dimod.SPIN)

        variables = list(bqm.variables)
        n_vars = len(variables)
        var_to_idx = {v: i for i, v in enumerate(variables)}

        # Extract linear and quadratic terms
        h = np.array([bqm.linear.get(v, 0.0) for v in variables])

        J = np.zeros((n_vars, n_vars))
        for (u, v), bias in bqm.quadratic.items():
            i, j = var_to_idx[u], var_to_idx[v]
            J[i, j] = bias
            J[j, i] = bias

        offset = bqm.offset

        def energy_fn(states: np.ndarray) -> np.ndarray:
            """
            Compute energy for batch of states.

            Args:
                states: Array of shape (batch_size, n_variables) with values in {-1, +1}

            Returns:
                energies: Array of shape (batch_size,)
            """
            # E(s) = -sum_i h_i * s_i - sum_{i,j} J_ij * s_i * s_j + offset
            # Linear term
            linear_energy = np.dot(states, h)

            # Quadratic term
            quadratic_energy = np.sum(
                states * np.dot(states, J),
                axis=1
            )

            return -(linear_energy + quadratic_energy) + offset

        return energy_fn, variables

    def sample(
        self,
        bqm: BinaryQuadraticModel,
        num_reads: int = 1,
        **kwargs
    ) -> SampleSet:
        """
        Sample from BinaryQuadraticModel.

        Args:
            bqm: Binary quadratic model
            num_reads: Number of samples
            **kwargs: Additional sampler-specific parameters

        Returns:
            SampleSet with samples
        """
        # Convert BQM to energy function
        energy_fn, variables = self._bqm_to_energy_fn(bqm)
        n_vars = len(variables)

        # Call base sampler
        samples = self.base_sampler.sample(
            energy_fn=energy_fn,
            n_variables=n_vars,
            num_samples=num_reads,
            **kwargs
        )

        # Convert back to SampleSet
        all_samples = []
        all_energies = []

        for sample_state in samples:
            sample_dict = {variables[i]: int(sample_state[i]) for i in range(n_vars)}
            energy = bqm.energy(sample_dict)

            all_samples.append(sample_dict)
            all_energies.append(energy)

        sampleset = SampleSet.from_samples(
            all_samples,
            vartype=dimod.SPIN,
            energy=all_energies
        )

        return sampleset


class MetropolisSampler(DimodSamplerBridge):
    """
    Metropolis-Hastings sampler with dimod interface.

    Standard Metropolis on discrete {-1,+1}^N with symmetric single-bit flip proposals.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        num_sweeps: int = 1000,
        burn_in: int = 100,
        thinning: int = 1,
        **params
    ):
        """
        Initialize Metropolis sampler.

        Args:
            temperature: Temperature T for acceptance probability
            num_sweeps: Number of MCMC sweeps (1 sweep = N proposals)
            burn_in: Number of burn-in sweeps
            thinning: Keep every nth sample
        """
        base = MetropolisSamplerBase(
            temperature=temperature,
            num_sweeps=num_sweeps,
            burn_in=burn_in,
            thinning=thinning,
            **params
        )
        super().__init__(base)

    @property
    def parameters(self):
        return {
            'temperature': ['Temperature for Metropolis acceptance'],
            'num_sweeps': ['Number of MCMC sweeps'],
            'burn_in': ['Number of burn-in sweeps'],
            'thinning': ['Thinning factor'],
            'num_reads': ['Number of independent chains']
        }


class ParallelTemperingSampler(DimodSamplerBridge):
    """
    Parallel Tempering (Replica Exchange) MCMC sampler with dimod interface.

    Runs multiple Metropolis chains at different temperatures and
    periodically proposes swaps between adjacent replicas.
    """

    def __init__(
        self,
        num_replicas: int = 8,
        T_min: float = 1.0,
        T_max: float = 4.0,
        swap_interval: int = 10,
        num_sweeps: int = 1000,
        burn_in: int = 100,
        thinning: int = 1,
        **params
    ):
        """
        Initialize Parallel Tempering sampler.

        Args:
            num_replicas: Number of temperature replicas
            T_min: Minimum temperature (physical/target)
            T_max: Maximum temperature
            swap_interval: Number of sweeps between swap attempts
            num_sweeps: Total number of sweeps
            burn_in: Number of burn-in sweeps
            thinning: Keep every nth sample
        """
        base = ParallelTemperingSamplerBase(
            num_replicas=num_replicas,
            T_min=T_min,
            T_max=T_max,
            swap_interval=swap_interval,
            num_sweeps=num_sweeps,
            burn_in=burn_in,
            thinning=thinning,
            **params
        )
        super().__init__(base)

    @property
    def parameters(self):
        return {
            'num_replicas': ['Number of temperature replicas'],
            'T_min': ['Minimum temperature'],
            'T_max': ['Maximum temperature'],
            'swap_interval': ['Sweeps between swap attempts'],
            'num_sweeps': ['Number of sweeps'],
            'burn_in': ['Burn-in period'],
            'thinning': ['Thinning factor'],
            'num_reads': ['Number of samples']
        }


class ExactSamplerBridge(DimodSamplerBridge):
    """
    Exact sampler via enumeration with dimod interface.

    Enumerates all 2^N states and samples from exact Boltzmann distribution.
    Only feasible for small N (N <= 20).
    """

    def __init__(self, max_variables: int = 20, **params):
        """
        Initialize exact sampler.

        Args:
            max_variables: Maximum number of variables
        """
        base = ExactSamplerBase(max_variables=max_variables, **params)
        super().__init__(base)

    @property
    def parameters(self):
        return {
            'max_variables': ['Maximum number of variables'],
            'num_reads': ['Number of exact samples']
        }


class GumbelMaxSampler(DimodSamplerBridge):
    """
    Gumbel-max trick sampler with dimod interface.

    Uses Gumbel-max reparameterization for exact independent samples.
    Only feasible for small N (N <= 20).
    """

    def __init__(self, max_variables: int = 20, **params):
        """
        Initialize Gumbel-max sampler.

        Args:
            max_variables: Maximum number of variables
        """
        base = GumbelMaxSamplerBase(max_variables=max_variables, **params)
        super().__init__(base)

    @property
    def parameters(self):
        return {
            'max_variables': ['Maximum number of variables'],
            'num_reads': ['Number of independent samples']
        }
