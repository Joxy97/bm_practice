"""
GPU-accelerated sampling algorithms for Boltzmann Machines.

Uses PyTorch for massively parallel sampling on CUDA GPUs.
All samplers support multi-chain parallelization for improved throughput.
"""

import torch
import numpy as np
from typing import Callable, Optional
from .base import BaseSampler, register_sampler
from .gpu_utils import get_device, compute_energy_batch, batch_to_numpy


@register_sampler('metropolis_gpu')
class MetropolisGPUSampler(BaseSampler):
    """
    GPU-accelerated Metropolis-Hastings sampler running many chains in parallel.

    Runs multiple independent Metropolis chains simultaneously on GPU for
    high throughput sampling.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        num_sweeps: int = 1000,
        burn_in: int = 100,
        thinning: int = 1,
        num_chains: int = 32,
        use_cuda: bool = True,
        **params
    ):
        """
        Initialize GPU Metropolis sampler.

        Args:
            temperature: Temperature T for acceptance probability
            num_sweeps: Number of MCMC sweeps per chain
            burn_in: Number of burn-in sweeps
            thinning: Keep every nth sample
            num_chains: Number of parallel chains to run
            use_cuda: Whether to use CUDA if available
        """
        super().__init__(
            temperature=temperature,
            num_sweeps=num_sweeps,
            burn_in=burn_in,
            thinning=thinning,
            num_chains=num_chains,
            use_cuda=use_cuda,
            **params
        )
        self.device = get_device(use_cuda)

    def _extract_energy_params(self, energy_fn: Callable, n_variables: int):
        """Extract h, J parameters from energy function by probing."""
        # Probe with basis states
        zero_state = np.zeros(n_variables)
        offset = energy_fn(zero_state.reshape(1, -1))[0]

        h = np.zeros(n_variables)
        for i in range(n_variables):
            state = np.zeros(n_variables)
            state[i] = 1.0
            E = energy_fn(state.reshape(1, -1))[0]
            h[i] = -(E - offset)

        J = np.zeros((n_variables, n_variables))
        for i in range(n_variables):
            for j in range(i + 1, n_variables):
                state_ij = np.zeros(n_variables)
                state_ij[i] = 1.0
                state_ij[j] = 1.0
                state_i = np.zeros(n_variables)
                state_i[i] = 1.0
                state_j = np.zeros(n_variables)
                state_j[j] = 1.0

                E_ij = energy_fn(state_ij.reshape(1, -1))[0]
                E_i = energy_fn(state_i.reshape(1, -1))[0]
                E_j = energy_fn(state_j.reshape(1, -1))[0]

                J_val = -(E_ij - E_i - E_j + offset)
                J[i, j] = J_val
                J[j, i] = J_val

        return h, J, offset

    def sample(
        self,
        energy_fn: Callable[[np.ndarray], np.ndarray],
        n_variables: int,
        num_samples: int,
        initial_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """Generate samples using parallel GPU Metropolis chains.

        Args:
            energy_fn: Energy function
            n_variables: Number of variables
            num_samples: Number of samples to generate
            initial_states: Optional initial states for PCD (num_chains, n_variables)
            **kwargs: Additional parameters

        Returns:
            Samples array (num_samples, n_variables)
        """
        T = self.params.get('temperature', 1.0)
        num_sweeps = self.params.get('num_sweeps', 1000)
        burn_in = self.params.get('burn_in', 100)
        thinning = self.params.get('thinning', 1)
        num_chains = self.params.get('num_chains', 32)

        # Extract energy parameters
        h, J, offset = self._extract_energy_params(energy_fn, n_variables)

        # Convert to torch tensors
        h_torch = torch.from_numpy(h).float().to(self.device)
        J_torch = torch.from_numpy(J).float().to(self.device)

        # Initialize chains: from initial_states if provided, else randomly
        if initial_states is not None:
            states = torch.from_numpy(initial_states).float().to(self.device)
            num_chains = states.shape[0]
        else:
            states = torch.randint(0, 2, (num_chains, n_variables), device=self.device).float() * 2 - 1

        # Compute initial energies
        energies = compute_energy_batch(states, h_torch, J_torch, offset)

        samples = []
        total_sweeps = burn_in + int(np.ceil(num_samples / num_chains)) * thinning

        for sweep in range(total_sweeps):
            # One sweep = N single-bit proposals per chain
            for _ in range(n_variables):
                # Pick random bit to flip for each chain
                flip_indices = torch.randint(0, n_variables, (num_chains,), device=self.device)

                # Create proposed states (flip selected bits)
                states_proposed = states.clone()
                states_proposed[torch.arange(num_chains, device=self.device), flip_indices] *= -1

                # Compute proposed energies
                energies_proposed = compute_energy_batch(states_proposed, h_torch, J_torch, offset)
                delta_E = energies_proposed - energies

                # Metropolis acceptance
                acceptance_probs = torch.clamp(torch.exp(-delta_E / T), max=1.0)
                accept = torch.rand(num_chains, device=self.device) < acceptance_probs

                # Update states and energies
                states[accept] = states_proposed[accept]
                energies[accept] = energies_proposed[accept]

            # Collect samples after burn-in
            if sweep >= burn_in and (sweep - burn_in) % thinning == 0:
                samples.append(states.clone())

        # Concatenate all samples and convert to numpy
        all_samples = torch.cat(samples, dim=0)
        all_samples_np = batch_to_numpy(all_samples)

        # Return exactly num_samples
        return all_samples_np[:num_samples]


@register_sampler('parallel_tempering_gpu')
class ParallelTemperingGPUSampler(BaseSampler):
    """
    GPU-accelerated Parallel Tempering with vectorized energy computations.

    Runs replica exchange MCMC with all replicas computed in parallel on GPU.
    Vectorized ΔE calculations for efficient swap proposals.
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
        use_cuda: bool = True,
        **params
    ):
        """
        Initialize GPU Parallel Tempering sampler.

        Args:
            num_replicas: Number of temperature replicas
            T_min: Minimum temperature
            T_max: Maximum temperature
            swap_interval: Sweeps between swap attempts
            num_sweeps: Total number of sweeps
            burn_in: Burn-in period
            thinning: Keep every nth sample
            use_cuda: Whether to use CUDA if available
        """
        super().__init__(
            num_replicas=num_replicas,
            T_min=T_min,
            T_max=T_max,
            swap_interval=swap_interval,
            num_sweeps=num_sweeps,
            burn_in=burn_in,
            thinning=thinning,
            use_cuda=use_cuda,
            **params
        )
        self.device = get_device(use_cuda)

        # Compute temperature ladder
        if num_replicas > 1:
            self.temperatures = T_min * np.power(
                T_max / T_min,
                np.arange(num_replicas) / (num_replicas - 1)
            )
        else:
            self.temperatures = np.array([T_min])

    def _extract_energy_params(self, energy_fn: Callable, n_variables: int):
        """Extract h, J parameters from energy function."""
        zero_state = np.zeros(n_variables)
        offset = energy_fn(zero_state.reshape(1, -1))[0]

        h = np.zeros(n_variables)
        for i in range(n_variables):
            state = np.zeros(n_variables)
            state[i] = 1.0
            E = energy_fn(state.reshape(1, -1))[0]
            h[i] = -(E - offset)

        J = np.zeros((n_variables, n_variables))
        for i in range(n_variables):
            for j in range(i + 1, n_variables):
                state_ij = np.zeros(n_variables)
                state_ij[i] = 1.0
                state_ij[j] = 1.0
                state_i = np.zeros(n_variables)
                state_i[i] = 1.0
                state_j = np.zeros(n_variables)
                state_j[j] = 1.0

                E_ij = energy_fn(state_ij.reshape(1, -1))[0]
                E_i = energy_fn(state_i.reshape(1, -1))[0]
                E_j = energy_fn(state_j.reshape(1, -1))[0]

                J_val = -(E_ij - E_i - E_j + offset)
                J[i, j] = J_val
                J[j, i] = J_val

        return h, J, offset

    def sample(
        self,
        energy_fn: Callable[[np.ndarray], np.ndarray],
        n_variables: int,
        num_samples: int,
        initial_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """Generate samples using GPU Parallel Tempering.

        Args:
            energy_fn: Energy function
            n_variables: Number of variables
            num_samples: Number of samples to generate
            initial_states: Optional initial states for PCD (num_replicas, n_variables)
            **kwargs: Additional parameters

        Returns:
            Samples array (num_samples, n_variables)
        """
        num_replicas = self.params.get('num_replicas', 8)
        swap_interval = self.params.get('swap_interval', 10)
        num_sweeps = self.params.get('num_sweeps', 1000)
        burn_in = self.params.get('burn_in', 100)
        thinning = self.params.get('thinning', 1)

        # Extract energy parameters
        h, J, offset = self._extract_energy_params(energy_fn, n_variables)

        # Convert to torch tensors
        h_torch = torch.from_numpy(h).float().to(self.device)
        J_torch = torch.from_numpy(J).float().to(self.device)
        temps_torch = torch.from_numpy(self.temperatures).float().to(self.device)

        # Initialize replicas: from initial_states if provided, else randomly
        if initial_states is not None:
            states = torch.from_numpy(initial_states).float().to(self.device)
            num_replicas = states.shape[0]
        else:
            states = torch.randint(0, 2, (num_replicas, n_variables), device=self.device).float() * 2 - 1

        # Compute initial energies for all replicas (vectorized)
        energies = compute_energy_batch(states, h_torch, J_torch, offset)

        samples = []
        total_sweeps = burn_in + num_samples * thinning

        for sweep in range(total_sweeps):
            # Update all replicas in parallel
            for _ in range(n_variables):
                # Propose flips for all replicas simultaneously
                flip_indices = torch.randint(0, n_variables, (num_replicas,), device=self.device)
                states_proposed = states.clone()
                states_proposed[torch.arange(num_replicas, device=self.device), flip_indices] *= -1

                # Vectorized energy computation
                energies_proposed = compute_energy_batch(states_proposed, h_torch, J_torch, offset)
                delta_E = energies_proposed - energies

                # Vectorized Metropolis acceptance for all temperatures
                acceptance_probs = torch.clamp(torch.exp(-delta_E / temps_torch), max=1.0)
                accept = torch.rand(num_replicas, device=self.device) < acceptance_probs

                states[accept] = states_proposed[accept]
                energies[accept] = energies_proposed[accept]

            # Attempt swaps between adjacent replicas (vectorized ΔE)
            if sweep % swap_interval == 0:
                for k in range(num_replicas - 1):
                    T_k = temps_torch[k]
                    T_k1 = temps_torch[k + 1]
                    E_k = energies[k]
                    E_k1 = energies[k + 1]

                    # Vectorized swap calculation
                    delta = (1.0 / T_k - 1.0 / T_k1) * (E_k1 - E_k)
                    swap_prob = torch.clamp(torch.exp(-delta), max=1.0)

                    if torch.rand(1, device=self.device) < swap_prob:
                        # Swap states and energies
                        states[[k, k + 1]] = states[[k + 1, k]]
                        energies[[k, k + 1]] = energies[[k + 1, k]]

            # Collect sample from lowest temperature replica
            if sweep >= burn_in and (sweep - burn_in) % thinning == 0:
                samples.append(states[0:1].clone())

        # Concatenate and convert to numpy
        all_samples = torch.cat(samples, dim=0)
        all_samples_np = batch_to_numpy(all_samples)

        return all_samples_np[:num_samples]


@register_sampler('gibbs_gpu')
class GibbsGPUSampler(BaseSampler):
    """
    GPU-accelerated Gibbs sampler with multi-chain parallelization.

    Runs multiple independent Gibbs chains in parallel on GPU.
    Can use block-Gibbs updates for improved vectorization.
    """

    def __init__(
        self,
        num_sweeps: int = 1000,
        burn_in: int = 100,
        thinning: int = 1,
        randomize_order: bool = True,
        num_chains: int = 32,
        use_cuda: bool = True,
        **params
    ):
        """
        Initialize GPU Gibbs sampler.

        Args:
            num_sweeps: Number of MCMC sweeps
            burn_in: Burn-in period
            thinning: Keep every nth sample
            randomize_order: Randomize variable update order
            num_chains: Number of parallel chains
            use_cuda: Whether to use CUDA if available
        """
        super().__init__(
            num_sweeps=num_sweeps,
            burn_in=burn_in,
            thinning=thinning,
            randomize_order=randomize_order,
            num_chains=num_chains,
            use_cuda=use_cuda,
            **params
        )
        self.device = get_device(use_cuda)

    def _extract_energy_params(self, energy_fn: Callable, n_variables: int):
        """Extract h, J parameters from energy function."""
        zero_state = np.zeros(n_variables)
        offset = energy_fn(zero_state.reshape(1, -1))[0]

        h = np.zeros(n_variables)
        for i in range(n_variables):
            state = np.zeros(n_variables)
            state[i] = 1.0
            E = energy_fn(state.reshape(1, -1))[0]
            h[i] = -(E - offset)

        J = np.zeros((n_variables, n_variables))
        for i in range(n_variables):
            for j in range(i + 1, n_variables):
                state_ij = np.zeros(n_variables)
                state_ij[i] = 1.0
                state_ij[j] = 1.0
                state_i = np.zeros(n_variables)
                state_i[i] = 1.0
                state_j = np.zeros(n_variables)
                state_j[j] = 1.0

                E_ij = energy_fn(state_ij.reshape(1, -1))[0]
                E_i = energy_fn(state_i.reshape(1, -1))[0]
                E_j = energy_fn(state_j.reshape(1, -1))[0]

                J_val = -(E_ij - E_i - E_j + offset)
                J[i, j] = J_val
                J[j, i] = J_val

        return h, J, offset

    def sample(
        self,
        energy_fn: Callable[[np.ndarray], np.ndarray],
        n_variables: int,
        num_samples: int,
        initial_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """Generate samples using parallel GPU Gibbs chains.

        Args:
            energy_fn: Energy function
            n_variables: Number of variables
            num_samples: Number of samples to generate
            initial_states: Optional initial states for PCD (num_chains, n_variables)
            **kwargs: Additional parameters

        Returns:
            Samples array (num_samples, n_variables)
        """
        num_sweeps = self.params.get('num_sweeps', 1000)
        burn_in = self.params.get('burn_in', 100)
        thinning = self.params.get('thinning', 1)
        randomize_order = self.params.get('randomize_order', True)
        num_chains = self.params.get('num_chains', 32)

        # Extract energy parameters
        h, J, offset = self._extract_energy_params(energy_fn, n_variables)

        # Convert to torch tensors
        h_torch = torch.from_numpy(h).float().to(self.device)
        J_torch = torch.from_numpy(J).float().to(self.device)

        # Initialize chains: from initial_states if provided, else randomly
        if initial_states is not None:
            states = torch.from_numpy(initial_states).float().to(self.device)
            num_chains = states.shape[0]
        else:
            states = torch.randint(0, 2, (num_chains, n_variables), device=self.device).float() * 2 - 1

        samples = []
        total_sweeps = burn_in + int(np.ceil(num_samples / num_chains)) * thinning

        for sweep in range(total_sweeps):
            # Determine update order
            if randomize_order:
                order = torch.randperm(n_variables, device=self.device)
            else:
                order = torch.arange(n_variables, device=self.device)

            # Update each variable for all chains
            for i in order:
                # Compute flip states
                states_flip = states.clone()
                states_flip[:, i] *= -1

                # Compute energies (vectorized)
                E_current = compute_energy_batch(states, h_torch, J_torch, offset)
                E_flip = compute_energy_batch(states_flip, h_torch, J_torch, offset)

                # Gibbs update probability (vectorized)
                delta_E = E_current - E_flip
                prob_flip = torch.sigmoid(delta_E)

                # Sample flips
                do_flip = torch.rand(num_chains, device=self.device) < prob_flip
                states[do_flip, i] *= -1

            # Collect samples after burn-in
            if sweep >= burn_in and (sweep - burn_in) % thinning == 0:
                samples.append(states.clone())

        # Concatenate and convert to numpy
        all_samples = torch.cat(samples, dim=0)
        all_samples_np = batch_to_numpy(all_samples)

        return all_samples_np[:num_samples]


@register_sampler('simulated_annealing_gpu')
class SimulatedAnnealingGPUSampler(BaseSampler):
    """
    GPU-accelerated Simulated Annealing sampler.

    Same as Metropolis GPU but with temperature schedule.
    Runs multiple annealing chains in parallel.
    """

    def __init__(
        self,
        beta_range: tuple = (1.0, 10.0),
        proposal_acceptance_criteria: str = 'Metropolis',
        num_sweeps: int = 1000,
        num_chains: int = 32,
        use_cuda: bool = True,
        **params
    ):
        """
        Initialize GPU Simulated Annealing sampler.

        Args:
            beta_range: [beta_min, beta_max] inverse temperature schedule
            proposal_acceptance_criteria: 'Gibbs' or 'Metropolis'
            num_sweeps: Number of sweeps
            num_chains: Number of parallel annealing chains
            use_cuda: Whether to use CUDA if available
        """
        super().__init__(
            beta_range=beta_range,
            proposal_acceptance_criteria=proposal_acceptance_criteria,
            num_sweeps=num_sweeps,
            num_chains=num_chains,
            use_cuda=use_cuda,
            **params
        )
        self.device = get_device(use_cuda)

    def _extract_energy_params(self, energy_fn: Callable, n_variables: int):
        """Extract h, J parameters from energy function."""
        zero_state = np.zeros(n_variables)
        offset = energy_fn(zero_state.reshape(1, -1))[0]

        h = np.zeros(n_variables)
        for i in range(n_variables):
            state = np.zeros(n_variables)
            state[i] = 1.0
            E = energy_fn(state.reshape(1, -1))[0]
            h[i] = -(E - offset)

        J = np.zeros((n_variables, n_variables))
        for i in range(n_variables):
            for j in range(i + 1, n_variables):
                state_ij = np.zeros(n_variables)
                state_ij[i] = 1.0
                state_ij[j] = 1.0
                state_i = np.zeros(n_variables)
                state_i[i] = 1.0
                state_j = np.zeros(n_variables)
                state_j[j] = 1.0

                E_ij = energy_fn(state_ij.reshape(1, -1))[0]
                E_i = energy_fn(state_i.reshape(1, -1))[0]
                E_j = energy_fn(state_j.reshape(1, -1))[0]

                J_val = -(E_ij - E_i - E_j + offset)
                J[i, j] = J_val
                J[j, i] = J_val

        return h, J, offset

    def sample(
        self,
        energy_fn: Callable[[np.ndarray], np.ndarray],
        n_variables: int,
        num_samples: int,
        **kwargs
    ) -> np.ndarray:
        """Generate samples using parallel GPU Simulated Annealing."""
        beta_range = self.params.get('beta_range', [1.0, 10.0])
        criteria = self.params.get('proposal_acceptance_criteria', 'Metropolis')
        num_sweeps = self.params.get('num_sweeps', 1000)
        num_chains = self.params.get('num_chains', 32)

        # Adjust num_chains to match num_samples if needed
        num_chains = min(num_chains, num_samples)

        # Extract energy parameters
        h, J, offset = self._extract_energy_params(energy_fn, n_variables)

        # Convert to torch tensors
        h_torch = torch.from_numpy(h).float().to(self.device)
        J_torch = torch.from_numpy(J).float().to(self.device)

        beta_min, beta_max = beta_range
        all_samples = []

        # Run multiple chains in parallel batches
        num_batches = int(np.ceil(num_samples / num_chains))

        for batch_idx in range(num_batches):
            batch_size = min(num_chains, num_samples - len(all_samples))

            # Initialize chains
            states = torch.randint(0, 2, (batch_size, n_variables), device=self.device).float() * 2 - 1

            # Annealing schedule
            for sweep in range(num_sweeps):
                t = sweep / max(1, num_sweeps - 1)
                beta = beta_min + (beta_max - beta_min) * t
                T = 1.0 / beta

                # Update all chains
                for i in range(n_variables):
                    states_flip = states.clone()
                    states_flip[:, i] *= -1

                    E_current = compute_energy_batch(states, h_torch, J_torch, offset)
                    E_flip = compute_energy_batch(states_flip, h_torch, J_torch, offset)

                    if criteria == 'Gibbs':
                        delta_E = beta * (E_current - E_flip)
                        prob_flip = torch.sigmoid(delta_E)
                    else:  # Metropolis
                        delta_E = E_flip - E_current
                        prob_flip = torch.clamp(torch.exp(-beta * delta_E), max=1.0)

                    do_flip = torch.rand(batch_size, device=self.device) < prob_flip
                    states[do_flip, i] *= -1

            # Collect final states
            all_samples.append(states)

        # Concatenate and convert to numpy
        final_samples = torch.cat(all_samples, dim=0)
        final_samples_np = batch_to_numpy(final_samples)

        return final_samples_np[:num_samples]


@register_sampler('population_annealing_gpu')
class PopulationAnnealingGPUSampler(BaseSampler):
    """
    GPU-accelerated Population Annealing sampler.

    Maintains a massive parallel population that is resampled based on
    importance weights during temperature schedule. Ideal for GPU acceleration.
    """

    def __init__(
        self,
        population_size: int = 1000,
        num_sweeps: int = 100,
        beta_min: float = 0.1,
        beta_max: float = 10.0,
        resample_threshold: float = 0.5,
        use_cuda: bool = True,
        **params
    ):
        """
        Initialize GPU Population Annealing sampler.

        Args:
            population_size: Size of parallel population
            num_sweeps: Number of temperature steps
            beta_min: Initial inverse temperature
            beta_max: Final inverse temperature
            resample_threshold: Resample when ESS < threshold * population_size
            use_cuda: Whether to use CUDA if available
        """
        super().__init__(
            population_size=population_size,
            num_sweeps=num_sweeps,
            beta_min=beta_min,
            beta_max=beta_max,
            resample_threshold=resample_threshold,
            use_cuda=use_cuda,
            **params
        )
        self.device = get_device(use_cuda)

    def _extract_energy_params(self, energy_fn: Callable, n_variables: int):
        """Extract h, J parameters from energy function."""
        zero_state = np.zeros(n_variables)
        offset = energy_fn(zero_state.reshape(1, -1))[0]

        h = np.zeros(n_variables)
        for i in range(n_variables):
            state = np.zeros(n_variables)
            state[i] = 1.0
            E = energy_fn(state.reshape(1, -1))[0]
            h[i] = -(E - offset)

        J = np.zeros((n_variables, n_variables))
        for i in range(n_variables):
            for j in range(i + 1, n_variables):
                state_ij = np.zeros(n_variables)
                state_ij[i] = 1.0
                state_ij[j] = 1.0
                state_i = np.zeros(n_variables)
                state_i[i] = 1.0
                state_j = np.zeros(n_variables)
                state_j[j] = 1.0

                E_ij = energy_fn(state_ij.reshape(1, -1))[0]
                E_i = energy_fn(state_i.reshape(1, -1))[0]
                E_j = energy_fn(state_j.reshape(1, -1))[0]

                J_val = -(E_ij - E_i - E_j + offset)
                J[i, j] = J_val
                J[j, i] = J_val

        return h, J, offset

    def sample(
        self,
        energy_fn: Callable[[np.ndarray], np.ndarray],
        n_variables: int,
        num_samples: int,
        **kwargs
    ) -> np.ndarray:
        """Generate samples using GPU Population Annealing."""
        population_size = self.params.get('population_size', 1000)
        num_sweeps = self.params.get('num_sweeps', 100)
        beta_min = self.params.get('beta_min', 0.1)
        beta_max = self.params.get('beta_max', 10.0)
        resample_threshold = self.params.get('resample_threshold', 0.5)

        # Extract energy parameters
        h, J, offset = self._extract_energy_params(energy_fn, n_variables)

        # Convert to torch tensors
        h_torch = torch.from_numpy(h).float().to(self.device)
        J_torch = torch.from_numpy(J).float().to(self.device)

        # Initialize population randomly
        population = torch.randint(0, 2, (population_size, n_variables), device=self.device).float() * 2 - 1

        # Compute initial energies
        energies = compute_energy_batch(population, h_torch, J_torch, offset)

        # Temperature schedule
        betas = torch.linspace(beta_min, beta_max, num_sweeps, device=self.device)

        # Initialize weights (uniform)
        weights = torch.ones(population_size, device=self.device) / population_size

        for sweep_idx in range(1, num_sweeps):
            beta_old = betas[sweep_idx - 1]
            beta_new = betas[sweep_idx]
            delta_beta = beta_new - beta_old

            # Update weights based on energy (importance sampling)
            log_weights = weights.log() - delta_beta * energies
            log_weights = log_weights - log_weights.max()  # Numerical stability
            weights = torch.exp(log_weights)
            weights = weights / weights.sum()

            # Compute Effective Sample Size (ESS)
            ess = 1.0 / (weights ** 2).sum()

            # Resample if ESS is too low
            if ess < resample_threshold * population_size:
                # Multinomial resampling
                indices = torch.multinomial(weights, population_size, replacement=True)
                population = population[indices]
                energies = energies[indices]
                weights = torch.ones(population_size, device=self.device) / population_size

            # MCMC updates at current temperature
            T = 1.0 / beta_new
            for _ in range(n_variables):
                flip_indices = torch.randint(0, n_variables, (population_size,), device=self.device)
                population_proposed = population.clone()
                population_proposed[torch.arange(population_size, device=self.device), flip_indices] *= -1

                energies_proposed = compute_energy_batch(population_proposed, h_torch, J_torch, offset)
                delta_E = energies_proposed - energies

                acceptance_probs = torch.clamp(torch.exp(-delta_E / T), max=1.0)
                accept = torch.rand(population_size, device=self.device) < acceptance_probs

                population[accept] = population_proposed[accept]
                energies[accept] = energies_proposed[accept]

        # Sample from final population according to weights
        indices = torch.multinomial(weights, num_samples, replacement=True)
        final_samples = population[indices]

        return batch_to_numpy(final_samples)
