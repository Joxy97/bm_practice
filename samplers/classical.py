"""
Classical sampling algorithms for Boltzmann Machines.

Includes MCMC samplers (Gibbs, Metropolis, Parallel Tempering, Simulated Annealing),
exact methods, and baseline samplers.
"""

import numpy as np
import itertools
from typing import Callable, Optional
from .base import BaseSampler, register_sampler


@register_sampler('gibbs')
class GibbsSampler(BaseSampler):
    """
    Gibbs sampler for Boltzmann Machines.

    Samples by iteratively updating each variable conditioned on others.
    """

    def __init__(
        self,
        num_sweeps: int = 1000,
        burn_in: int = 100,
        thinning: int = 1,
        randomize_order: bool = True,
        **params
    ):
        """
        Initialize Gibbs sampler.

        Args:
            num_sweeps: Number of MCMC sweeps
            burn_in: Number of burn-in sweeps
            thinning: Keep every nth sample
            randomize_order: Randomize variable update order
        """
        super().__init__(
            num_sweeps=num_sweeps,
            burn_in=burn_in,
            thinning=thinning,
            randomize_order=randomize_order,
            **params
        )

    def sample(
        self,
        energy_fn: Callable[[np.ndarray], np.ndarray],
        n_variables: int,
        num_samples: int,
        **kwargs
    ) -> np.ndarray:
        """Generate samples using Gibbs sampling."""
        num_sweeps = self.params.get('num_sweeps', 1000)
        burn_in = self.params.get('burn_in', 100)
        thinning = self.params.get('thinning', 1)
        randomize_order = self.params.get('randomize_order', True)

        # Initialize random state
        x = np.random.choice([-1, 1], size=n_variables)

        samples = []
        total_sweeps = burn_in + num_samples * thinning

        for sweep in range(total_sweeps):
            # Determine update order
            if randomize_order:
                order = np.random.permutation(n_variables)
            else:
                order = np.arange(n_variables)

            # Update each variable
            for i in order:
                # Flip variable i
                x_flip = x.copy()
                x_flip[i] *= -1

                # Compute energies
                E_current = energy_fn(x.reshape(1, -1))[0]
                E_flip = energy_fn(x_flip.reshape(1, -1))[0]

                # Gibbs update: p(x_i=+1 | x_{\i}) = σ(ΔE)
                delta_E = E_current - E_flip
                prob_flip = 1.0 / (1.0 + np.exp(-delta_E))

                if np.random.random() < prob_flip:
                    x = x_flip

            # Collect sample after burn-in
            if sweep >= burn_in and (sweep - burn_in) % thinning == 0:
                samples.append(x.copy())

        return np.array(samples)


@register_sampler('metropolis')
class MetropolisSampler(BaseSampler):
    """
    Metropolis-Hastings sampler with single-bit flip proposals.

    Standard Metropolis on discrete {-1,+1}^N with symmetric proposals.
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
        super().__init__(
            temperature=temperature,
            num_sweeps=num_sweeps,
            burn_in=burn_in,
            thinning=thinning,
            **params
        )

    def sample(
        self,
        energy_fn: Callable[[np.ndarray], np.ndarray],
        n_variables: int,
        num_samples: int,
        **kwargs
    ) -> np.ndarray:
        """Generate samples using Metropolis-Hastings."""
        T = self.params.get('temperature', 1.0)
        num_sweeps = self.params.get('num_sweeps', 1000)
        burn_in = self.params.get('burn_in', 100)
        thinning = self.params.get('thinning', 1)

        # Initialize random state
        x = np.random.choice([-1, 1], size=n_variables)

        samples = []
        total_sweeps = burn_in + num_samples * thinning

        for sweep in range(total_sweeps):
            # One sweep = N single-bit proposals
            for _ in range(n_variables):
                # Pick random bit to flip
                i = np.random.randint(n_variables)

                # Propose flip
                x_proposed = x.copy()
                x_proposed[i] *= -1

                # Compute energy change
                E_current = energy_fn(x.reshape(1, -1))[0]
                E_proposed = energy_fn(x_proposed.reshape(1, -1))[0]
                delta_E = E_proposed - E_current

                # Metropolis acceptance: min(1, exp(-ΔE/T))
                acceptance_prob = min(1.0, np.exp(-delta_E / T))

                if np.random.random() < acceptance_prob:
                    x = x_proposed

            # Collect sample after burn-in
            if sweep >= burn_in and (sweep - burn_in) % thinning == 0:
                samples.append(x.copy())

        return np.array(samples)


@register_sampler('parallel_tempering')
class ParallelTemperingSampler(BaseSampler):
    """
    Parallel Tempering (Replica Exchange) MCMC sampler.

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
        super().__init__(
            num_replicas=num_replicas,
            T_min=T_min,
            T_max=T_max,
            swap_interval=swap_interval,
            num_sweeps=num_sweeps,
            burn_in=burn_in,
            thinning=thinning,
            **params
        )

        # Compute temperature ladder (geometric)
        if num_replicas > 1:
            self.temperatures = T_min * np.power(
                T_max / T_min,
                np.arange(num_replicas) / (num_replicas - 1)
            )
        else:
            self.temperatures = np.array([T_min])

    def sample(
        self,
        energy_fn: Callable[[np.ndarray], np.ndarray],
        n_variables: int,
        num_samples: int,
        **kwargs
    ) -> np.ndarray:
        """Generate samples using Parallel Tempering."""
        num_replicas = self.params.get('num_replicas', 8)
        swap_interval = self.params.get('swap_interval', 10)
        num_sweeps = self.params.get('num_sweeps', 1000)
        burn_in = self.params.get('burn_in', 100)
        thinning = self.params.get('thinning', 1)

        # Initialize all replicas
        states = np.random.choice([-1, 1], size=(num_replicas, n_variables))

        samples = []
        total_sweeps = burn_in + num_samples * thinning

        for sweep in range(total_sweeps):
            # Update each replica with Metropolis
            for k in range(num_replicas):
                T_k = self.temperatures[k]

                # One sweep = N single-bit proposals
                for _ in range(n_variables):
                    i = np.random.randint(n_variables)

                    # Propose flip
                    x_proposed = states[k].copy()
                    x_proposed[i] *= -1

                    # Compute energy change
                    E_current = energy_fn(states[k].reshape(1, -1))[0]
                    E_proposed = energy_fn(x_proposed.reshape(1, -1))[0]
                    delta_E = E_proposed - E_current

                    # Metropolis acceptance
                    acceptance_prob = min(1.0, np.exp(-delta_E / T_k))

                    if np.random.random() < acceptance_prob:
                        states[k] = x_proposed

            # Attempt swaps between adjacent replicas
            if sweep % swap_interval == 0:
                for k in range(num_replicas - 1):
                    # Compute energies
                    E_k = energy_fn(states[k].reshape(1, -1))[0]
                    E_k1 = energy_fn(states[k + 1].reshape(1, -1))[0]

                    # Swap acceptance probability
                    T_k = self.temperatures[k]
                    T_k1 = self.temperatures[k + 1]
                    delta = (1.0 / T_k - 1.0 / T_k1) * (E_k1 - E_k)
                    swap_prob = min(1.0, np.exp(-delta))

                    if np.random.random() < swap_prob:
                        # Swap states
                        states[k], states[k + 1] = states[k + 1].copy(), states[k].copy()

            # Collect sample from lowest temperature replica (T_min)
            if sweep >= burn_in and (sweep - burn_in) % thinning == 0:
                samples.append(states[0].copy())

        return np.array(samples)


@register_sampler('simulated_annealing')
class SimulatedAnnealingSampler(BaseSampler):
    """
    Simulated Annealing sampler with temperature schedule.

    Gradually decreases temperature to find low-energy states.
    """

    def __init__(
        self,
        beta_range: tuple = (1.0, 1.0),
        proposal_acceptance_criteria: str = 'Gibbs',
        num_sweeps: int = 1000,
        **params
    ):
        """
        Initialize Simulated Annealing sampler.

        Args:
            beta_range: [beta_min, beta_max] inverse temperature range
            proposal_acceptance_criteria: 'Gibbs' or 'Metropolis'
            num_sweeps: Number of sweeps
        """
        super().__init__(
            beta_range=beta_range,
            proposal_acceptance_criteria=proposal_acceptance_criteria,
            num_sweeps=num_sweeps,
            **params
        )

    def sample(
        self,
        energy_fn: Callable[[np.ndarray], np.ndarray],
        n_variables: int,
        num_samples: int,
        **kwargs
    ) -> np.ndarray:
        """Generate samples using Simulated Annealing."""
        beta_range = self.params.get('beta_range', [1.0, 1.0])
        criteria = self.params.get('proposal_acceptance_criteria', 'Gibbs')
        num_sweeps = self.params.get('num_sweeps', 1000)

        beta_min, beta_max = beta_range
        samples = []

        for _ in range(num_samples):
            # Initialize random state
            x = np.random.choice([-1, 1], size=n_variables)

            # Annealing schedule
            for sweep in range(num_sweeps):
                # Linear temperature schedule
                t = sweep / max(1, num_sweeps - 1)
                beta = beta_min + (beta_max - beta_min) * t

                # Update variables
                for i in range(n_variables):
                    x_flip = x.copy()
                    x_flip[i] *= -1

                    E_current = energy_fn(x.reshape(1, -1))[0]
                    E_flip = energy_fn(x_flip.reshape(1, -1))[0]

                    if criteria == 'Gibbs':
                        delta_E = beta * (E_current - E_flip)
                        prob_flip = 1.0 / (1.0 + np.exp(-delta_E))
                    else:  # Metropolis
                        delta_E = E_flip - E_current
                        prob_flip = min(1.0, np.exp(-beta * delta_E))

                    if np.random.random() < prob_flip:
                        x = x_flip

            samples.append(x.copy())

        return np.array(samples)


@register_sampler('random')
class RandomSampler(BaseSampler):
    """
    Random baseline sampler.

    Generates uniformly random binary configurations.
    """

    def __init__(self, **params):
        """Initialize random sampler."""
        super().__init__(**params)

    def sample(
        self,
        energy_fn: Callable[[np.ndarray], np.ndarray],
        n_variables: int,
        num_samples: int,
        **kwargs
    ) -> np.ndarray:
        """Generate random samples."""
        return np.random.choice([-1, 1], size=(num_samples, n_variables))


@register_sampler('exact')
class ExactSampler(BaseSampler):
    """
    Exact sampler via enumeration and direct sampling.

    Enumerates all 2^N states and samples from the exact Boltzmann distribution.
    Only feasible for small N (N <= 20).
    """

    def __init__(self, max_variables: int = 20, **params):
        """
        Initialize exact sampler.

        Args:
            max_variables: Maximum number of variables (default: 20)
        """
        super().__init__(max_variables=max_variables, **params)

    def sample(
        self,
        energy_fn: Callable[[np.ndarray], np.ndarray],
        n_variables: int,
        num_samples: int,
        **kwargs
    ) -> np.ndarray:
        """Generate exact samples from Boltzmann distribution."""
        max_vars = self.params.get('max_variables', 20)

        if n_variables > max_vars:
            raise ValueError(
                f"Exact sampler requires N <= {max_vars}, got N = {n_variables}. "
                f"Use approximate samplers for larger problems."
            )

        # Enumerate all states
        all_states = np.array(list(itertools.product([-1, 1], repeat=n_variables)))

        # Compute all energies
        energies = energy_fn(all_states)

        # Compute Boltzmann probabilities
        log_unnormalized = -energies
        log_Z = np.logaddexp.reduce(log_unnormalized)
        log_probs = log_unnormalized - log_Z
        probs = np.exp(log_probs)

        # Sample from the distribution
        indices = np.random.choice(len(all_states), size=num_samples, p=probs)
        samples = all_states[indices]

        return samples


@register_sampler('gumbel_max')
class GumbelMaxSampler(BaseSampler):
    """
    Gumbel-max trick for exact sampling.

    Uses the Gumbel-max reparameterization to generate independent
    exact samples from the Boltzmann distribution.
    """

    def __init__(self, max_variables: int = 20, **params):
        """
        Initialize Gumbel-max sampler.

        Args:
            max_variables: Maximum number of variables (default: 20)
        """
        super().__init__(max_variables=max_variables, **params)

    def sample(
        self,
        energy_fn: Callable[[np.ndarray], np.ndarray],
        n_variables: int,
        num_samples: int,
        **kwargs
    ) -> np.ndarray:
        """Generate samples using Gumbel-max trick."""
        max_vars = self.params.get('max_variables', 20)

        if n_variables > max_vars:
            raise ValueError(
                f"Gumbel-max sampler requires N <= {max_vars}, got N = {n_variables}. "
                f"Use approximate samplers for larger problems."
            )

        # Enumerate all states
        all_states = np.array(list(itertools.product([-1, 1], repeat=n_variables)))
        n_states = len(all_states)

        # Compute all energies once
        energies = energy_fn(all_states)

        samples = []
        for _ in range(num_samples):
            # Draw i.i.d. Gumbel noise
            gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, size=n_states)))

            # Compute perturbed scores: -E(x) + g_x
            scores = -energies + gumbel_noise

            # Take argmax
            max_idx = np.argmax(scores)
            samples.append(all_states[max_idx])

        return np.array(samples)
