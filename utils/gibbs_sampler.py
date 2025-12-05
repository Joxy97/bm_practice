"""
Gibbs MCMC Sampler for Boltzmann Machines

A pure Gibbs sampling implementation compatible with dimod's Sampler interface.
Performs Markov Chain Monte Carlo sampling using the Gibbs sampling algorithm.
"""

import numpy as np
from typing import Union, Optional
from dimod import BinaryQuadraticModel, SampleSet, Sampler
import dimod


class GibbsSampler(Sampler):
    """
    Gibbs MCMC Sampler for binary quadratic models.

    Performs Gibbs sampling where each variable is sampled from its conditional
    distribution given all other variables:

    p(x_i | x_{-i}) = exp(-ΔE_i) / (exp(-ΔE_i) + 1)

    where ΔE_i is the energy change when flipping variable x_i.

    Parameters:
        num_sweeps (int): Number of full sweeps through all variables (default: 1000)
        burn_in (int): Number of initial sweeps to discard (default: 100)
        thinning (int): Keep every nth sample to reduce autocorrelation (default: 1)
        randomize_order (bool): Randomly permute variable order each sweep (default: True)
    """

    properties = {
        'description': 'Gibbs MCMC Sampler for binary quadratic models'
    }

    parameters = {
        'num_sweeps': ['Number of full sweeps through variables'],
        'burn_in': ['Number of initial sweeps to discard'],
        'thinning': ['Thinning factor for samples'],
        'randomize_order': ['Whether to randomize variable update order'],
        'num_reads': ['Number of independent chains to run']
    }

    def __init__(
        self,
        num_sweeps: int = 1000,
        burn_in: int = 100,
        thinning: int = 1,
        randomize_order: bool = True
    ):
        """
        Initialize the Gibbs sampler.

        Args:
            num_sweeps: Total number of sweeps through all variables
            burn_in: Number of initial sweeps to discard
            thinning: Keep every nth sample
            randomize_order: Whether to randomize variable order each sweep
        """
        self.num_sweeps = num_sweeps
        self.burn_in = burn_in
        self.thinning = thinning
        self.randomize_order = randomize_order

    @property
    def parameters(self):
        return {
            'num_sweeps': ['Number of full sweeps through variables'],
            'burn_in': ['Number of initial sweeps to discard'],
            'thinning': ['Thinning factor for samples'],
            'randomize_order': ['Whether to randomize variable update order'],
            'num_reads': ['Number of independent chains to run']
        }

    @property
    def properties(self):
        return {
            'description': 'Gibbs MCMC Sampler for binary quadratic models'
        }

    def sample(
        self,
        bqm: BinaryQuadraticModel,
        num_reads: int = 1,
        num_sweeps: Optional[int] = None,
        burn_in: Optional[int] = None,
        thinning: Optional[int] = None,
        randomize_order: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> SampleSet:
        """
        Sample from a binary quadratic model using Gibbs sampling.

        Args:
            bqm: Binary quadratic model to sample from
            num_reads: Number of independent Markov chains to run
            num_sweeps: Override default number of sweeps
            burn_in: Override default burn-in period
            thinning: Override default thinning factor
            randomize_order: Override default variable order randomization
            seed: Random seed for reproducibility
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            SampleSet with samples and their energies
        """
        # Use provided parameters or fall back to instance defaults
        num_sweeps = num_sweeps if num_sweeps is not None else self.num_sweeps
        burn_in = burn_in if burn_in is not None else self.burn_in
        thinning = thinning if thinning is not None else self.thinning
        randomize_order = randomize_order if randomize_order is not None else self.randomize_order

        # Set random seed
        rng = np.random.RandomState(seed)

        # Get variables and convert to list for indexing
        variables = list(bqm.variables)
        n_vars = len(variables)

        # Build variable index mapping
        var_to_idx = {v: i for i, v in enumerate(variables)}

        # Extract linear and quadratic terms for efficient access
        h = np.array([bqm.linear.get(v, 0.0) for v in variables])

        # Build interaction matrix J
        J = np.zeros((n_vars, n_vars))
        for (u, v), bias in bqm.quadratic.items():
            i, j = var_to_idx[u], var_to_idx[v]
            J[i, j] = bias
            J[j, i] = bias  # Symmetric

        # Offset
        offset = bqm.offset

        # Store samples from all chains
        all_samples = []
        all_energies = []

        # Run multiple independent chains
        for _ in range(num_reads):
            # Initialize random state (binary: 0 or 1)
            state = rng.randint(0, 2, size=n_vars)

            # Convert to spin: {-1, +1} if needed for energy calculation
            # BQM uses {0, 1} encoding by default, so we keep it

            samples = []

            # MCMC loop
            total_sweeps = burn_in + num_sweeps
            for sweep in range(total_sweeps):
                # Determine variable update order
                if randomize_order:
                    var_order = rng.permutation(n_vars)
                else:
                    var_order = np.arange(n_vars)

                # Update each variable
                for idx in var_order:
                    # Compute energy change for flipping variable idx
                    # E = sum_i h_i * x_i + sum_{i<j} J_{ij} * x_i * x_j + offset

                    # Current state contribution
                    current_val = state[idx]

                    # Energy contribution from this variable
                    # Linear term: h[idx] * x[idx]
                    # Quadratic terms: sum_j J[idx, j] * x[idx] * x[j]

                    # Field acting on variable idx (interaction with neighbors)
                    field = h[idx] + np.dot(J[idx], state)

                    # Energy difference when x[idx] = 1 vs x[idx] = 0
                    # E(x[idx]=1) - E(x[idx]=0) = field * (1 - 0) = field
                    delta_E = field

                    # Gibbs sampling: p(x[idx]=1 | x_{-idx}) = 1 / (1 + exp(delta_E))
                    # This is for minimization (BQM convention)
                    prob_1 = 1.0 / (1.0 + np.exp(delta_E))

                    # Sample new value
                    state[idx] = 1 if rng.random() < prob_1 else 0

                # Store sample after burn-in, with thinning
                if sweep >= burn_in and (sweep - burn_in) % thinning == 0:
                    samples.append(state.copy())

            # Convert samples to the format expected by SampleSet
            for sample_state in samples:
                # Create sample dict
                sample_dict = {variables[i]: int(sample_state[i]) for i in range(n_vars)}

                # Compute energy
                energy = bqm.energy(sample_dict)

                all_samples.append(sample_dict)
                all_energies.append(energy)

        # Create SampleSet
        sampleset = SampleSet.from_samples(
            all_samples,
            vartype=bqm.vartype,
            energy=all_energies
        )

        return sampleset


class GibbsSamplerSpin(GibbsSampler):
    """
    Gibbs MCMC Sampler optimized for spin variables {-1, +1}.

    This is a specialized version for Boltzmann Machines that use spin encoding.
    """

    def sample(
        self,
        bqm: BinaryQuadraticModel,
        num_reads: int = 1,
        num_sweeps: Optional[int] = None,
        burn_in: Optional[int] = None,
        thinning: Optional[int] = None,
        randomize_order: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> SampleSet:
        """
        Sample from a spin-based binary quadratic model using Gibbs sampling.

        For spin variables s_i ∈ {-1, +1}:
        E(s) = -sum_i h_i * s_i - sum_{i<j} J_{ij} * s_i * s_j

        Args:
            bqm: Binary quadratic model with SPIN vartype
            num_reads: Number of independent Markov chains
            num_sweeps: Override default number of sweeps
            burn_in: Override default burn-in period
            thinning: Override default thinning factor
            randomize_order: Override default variable order randomization
            seed: Random seed
            **kwargs: Additional arguments (ignored)

        Returns:
            SampleSet with spin samples and energies
        """
        # Convert to SPIN if not already
        if bqm.vartype != dimod.SPIN:
            bqm = bqm.change_vartype(dimod.SPIN)

        # Use provided parameters or fall back to instance defaults
        num_sweeps = num_sweeps if num_sweeps is not None else self.num_sweeps
        burn_in = burn_in if burn_in is not None else self.burn_in
        thinning = thinning if thinning is not None else self.thinning
        randomize_order = randomize_order if randomize_order is not None else self.randomize_order

        # Set random seed
        rng = np.random.RandomState(seed)

        # Get variables
        variables = list(bqm.variables)
        n_vars = len(variables)
        var_to_idx = {v: i for i, v in enumerate(variables)}

        # Extract parameters
        h = np.array([bqm.linear.get(v, 0.0) for v in variables])

        J = np.zeros((n_vars, n_vars))
        for (u, v), bias in bqm.quadratic.items():
            i, j = var_to_idx[u], var_to_idx[v]
            J[i, j] = bias
            J[j, i] = bias

        all_samples = []
        all_energies = []

        # Run chains
        for _ in range(num_reads):
            # Initialize random spin state {-1, +1}
            state = rng.choice([-1, 1], size=n_vars)

            samples = []

            total_sweeps = burn_in + num_sweeps
            for sweep in range(total_sweeps):
                # Variable update order
                if randomize_order:
                    var_order = rng.permutation(n_vars)
                else:
                    var_order = np.arange(n_vars)

                # Gibbs update for spin variables
                for idx in var_order:
                    # Compute local field acting on variable idx
                    # For Ising model: E = -sum h_i s_i - sum J_ij s_i s_j
                    # Local field: h[idx] + sum_{j != idx} J[idx,j] * s[j]
                    # Note: J diagonal should be zero, so we can safely include s[idx]
                    field = h[idx] + np.dot(J[idx], state)

                    # Conditional probability for spin {-1, +1}:
                    # p(s_i = +1 | s_{-i}) = 1 / (1 + exp(-2*field))
                    # This comes from the ratio of Boltzmann weights
                    prob_plus = 1.0 / (1.0 + np.exp(-2.0 * field))

                    state[idx] = 1 if rng.random() < prob_plus else -1

                # Store sample
                if sweep >= burn_in and (sweep - burn_in) % thinning == 0:
                    samples.append(state.copy())

            # Convert to SampleSet format
            # Compute energies in batch for all collected samples
            if samples:
                for sample_state in samples:
                    sample_dict = {variables[i]: int(sample_state[i]) for i in range(n_vars)}
                    all_samples.append(sample_dict)

                # Batch compute energies efficiently
                sample_array = np.array(samples)
                # E = -h^T s - s^T J s (with factor of 0.5 to avoid double counting)
                linear_energy = -np.dot(sample_array, h)
                quadratic_energy = -0.5 * np.sum(sample_array * np.dot(sample_array, J), axis=1)
                energies_batch = linear_energy + quadratic_energy + bqm.offset
                all_energies.extend(energies_batch.tolist())

        sampleset = SampleSet.from_samples(
            all_samples,
            vartype=dimod.SPIN,
            energy=all_energies
        )

        return sampleset


# Convenience function
def gibbs_sample(
    bqm: BinaryQuadraticModel,
    num_reads: int = 1,
    num_sweeps: int = 1000,
    burn_in: int = 100,
    thinning: int = 1,
    randomize_order: bool = True,
    seed: Optional[int] = None
) -> SampleSet:
    """
    Convenience function for Gibbs sampling.

    Args:
        bqm: Binary quadratic model
        num_reads: Number of independent chains
        num_sweeps: Number of sweeps per chain
        burn_in: Burn-in period
        thinning: Thinning factor
        randomize_order: Randomize variable order
        seed: Random seed

    Returns:
        SampleSet with samples
    """
    if bqm.vartype == dimod.SPIN:
        sampler = GibbsSamplerSpin(num_sweeps, burn_in, thinning, randomize_order)
    else:
        sampler = GibbsSampler(num_sweeps, burn_in, thinning, randomize_order)

    return sampler.sample(bqm, num_reads=num_reads, seed=seed)
