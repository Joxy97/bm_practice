"""
Sampler factory for creating D-Wave samplers based on configuration.
"""

from typing import Optional, Dict, Any
from dimod import Sampler


def create_sampler(sampler_type: str, params: Optional[Dict[str, Any]] = None) -> Sampler:
    """
    Create a sampler based on the specified type and parameters.

    Args:
        sampler_type: Type of sampler to create. Options:
            Classical Samplers:
            - "simulated_annealing": Classical MCMC-based sampler (default)
            - "tabu": Tabu search metaheuristic
            - "steepest_descent": Steepest descent local search
            - "greedy": Greedy heuristic sampler
            - "exact": Exact solver (brute force, only for small problems)
            - "random": Random sampler

            D-Wave Quantum Samplers (requires Leap access):
            - "dwave": D-Wave quantum annealer (QPU)
            - "advantage": D-Wave Advantage quantum system (alias for dwave)

            D-Wave Hybrid Samplers (requires Leap access):
            - "hybrid": LeapHybridSampler (general purpose)
            - "hybrid_bqm": LeapHybridBQMSampler (for BQM problems)
            - "hybrid_dqm": LeapHybridDQMSampler (for DQM problems)
            - "kerberos": KerberosSampler (QPU + classical hybrid)

        params: Dictionary of sampler-specific parameters

    Returns:
        A dimod.Sampler instance

    Raises:
        ValueError: If sampler_type is not recognized
        ImportError: If required sampler package is not installed
    """
    if params is None:
        params = {}

    sampler_type = sampler_type.lower().replace("-", "_").replace(" ", "_")

    # Classical samplers
    if sampler_type == "simulated_annealing":
        from dwave.samplers import SimulatedAnnealingSampler
        return SimulatedAnnealingSampler()

    elif sampler_type == "tabu":
        from dwave.samplers import TabuSampler
        # Extract tabu-specific parameters
        tenure = params.get("tenure", None)
        timeout = params.get("timeout", 20)
        return TabuSampler(tenure=tenure, timeout=timeout)

    elif sampler_type == "steepest_descent":
        from dwave.samplers import SteepestDescentSampler
        return SteepestDescentSampler()

    elif sampler_type == "greedy":
        from dwave.samplers import GreedySampler
        return GreedySampler()

    elif sampler_type == "exact":
        from samplers.dimod_bridge import ExactSamplerBridge
        max_variables = params.get("max_variables", 20)
        print("WARNING: Exact sampler uses brute force enumeration. Only suitable for ~20 variables or less.")
        return ExactSamplerBridge(max_variables=max_variables)

    elif sampler_type == "random":
        from dimod import RandomSampler
        return RandomSampler()

    elif sampler_type == "gibbs":
        from utils.gibbs_sampler import GibbsSamplerSpin
        # Extract Gibbs-specific parameters
        num_sweeps = params.get("num_sweeps", 1000)
        burn_in = params.get("burn_in", 100)
        thinning = params.get("thinning", 1)
        randomize_order = params.get("randomize_order", True)
        return GibbsSamplerSpin(
            num_sweeps=num_sweeps,
            burn_in=burn_in,
            thinning=thinning,
            randomize_order=randomize_order
        )

    elif sampler_type == "metropolis":
        from samplers.dimod_bridge import MetropolisSampler
        temperature = params.get("temperature", 1.0)
        num_sweeps = params.get("num_sweeps", 1000)
        burn_in = params.get("burn_in", 100)
        thinning = params.get("thinning", 1)
        return MetropolisSampler(
            temperature=temperature,
            num_sweeps=num_sweeps,
            burn_in=burn_in,
            thinning=thinning
        )

    elif sampler_type == "parallel_tempering":
        from samplers.dimod_bridge import ParallelTemperingSampler
        num_replicas = params.get("num_replicas", 8)
        T_min = params.get("T_min", 1.0)
        T_max = params.get("T_max", 4.0)
        swap_interval = params.get("swap_interval", 10)
        num_sweeps = params.get("num_sweeps", 1000)
        burn_in = params.get("burn_in", 100)
        thinning = params.get("thinning", 1)
        return ParallelTemperingSampler(
            num_replicas=num_replicas,
            T_min=T_min,
            T_max=T_max,
            swap_interval=swap_interval,
            num_sweeps=num_sweeps,
            burn_in=burn_in,
            thinning=thinning
        )

    elif sampler_type == "gumbel_max":
        from samplers.dimod_bridge import GumbelMaxSampler
        max_variables = params.get("max_variables", 20)
        print("INFO: Gumbel-max sampler uses exact enumeration. Only suitable for ~20 variables or less.")
        return GumbelMaxSampler(max_variables=max_variables)

    # D-Wave Quantum samplers
    elif sampler_type in ["dwave", "advantage"]:
        try:
            from dwave.system import DWaveSampler, EmbeddingComposite
            print("INFO: Connecting to D-Wave quantum annealer via Leap...")
            print("      This requires a valid API token and Leap access.")

            # Get solver name if specified
            solver = params.get("solver", None)

            # Create base sampler
            if solver:
                base_sampler = DWaveSampler(solver=solver)
            else:
                base_sampler = DWaveSampler()

            print(f"      Connected to: {base_sampler.solver.name}")

            # Wrap with embedding composite for automatic minor embedding
            sampler = EmbeddingComposite(base_sampler)
            return sampler

        except ImportError:
            raise ImportError(
                "D-Wave quantum sampler requires dwave-system. Install with:\n"
                "  pip install dwave-system"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to D-Wave quantum annealer: {e}\n"
                "Make sure you have:\n"
                "  1. Valid D-Wave Leap API token (set DWAVE_API_TOKEN env var or configure with 'dwave config create')\n"
                "  2. Active Leap account with QPU access\n"
                "  3. Internet connection"
            )

    # D-Wave Hybrid samplers
    elif sampler_type in ["hybrid", "hybrid_bqm"]:
        try:
            from dwave.system import LeapHybridSampler
            print("INFO: Connecting to D-Wave Leap Hybrid solver...")
            print("      This requires a valid API token and Leap access.")

            # Get solver name if specified
            solver = params.get("solver", None)

            if solver:
                sampler = LeapHybridSampler(solver=solver)
            else:
                sampler = LeapHybridSampler()

            print(f"      Connected to: {sampler.solver.name}")
            return sampler

        except ImportError:
            raise ImportError(
                "D-Wave Leap Hybrid sampler requires dwave-system. Install with:\n"
                "  pip install dwave-system"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to D-Wave Leap Hybrid solver: {e}\n"
                "Make sure you have:\n"
                "  1. Valid D-Wave Leap API token (set DWAVE_API_TOKEN env var or configure with 'dwave config create')\n"
                "  2. Active Leap account\n"
                "  3. Internet connection"
            )

    elif sampler_type == "hybrid_dqm":
        try:
            from dwave.system import LeapHybridDQMSampler
            print("INFO: Connecting to D-Wave Leap Hybrid DQM solver...")
            sampler = LeapHybridDQMSampler()
            print(f"      Connected to: {sampler.solver.name}")
            return sampler
        except ImportError:
            raise ImportError(
                "D-Wave Leap Hybrid DQM sampler requires dwave-system. Install with:\n"
                "  pip install dwave-system"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Leap Hybrid DQM solver: {e}")

    elif sampler_type == "kerberos":
        try:
            from hybrid.reference.kerberos import KerberosSampler

            print("INFO: Creating Kerberos hybrid sampler (QPU + classical)...")

            # Extract parameters
            max_iter = params.get("max_iter", 10)
            max_time = params.get("max_time", None)

            # KerberosSampler will automatically connect to QPU
            sampler = KerberosSampler(max_iter=max_iter, max_time=max_time)
            return sampler

        except ImportError:
            raise ImportError(
                "Kerberos sampler requires dwave-system and dwave-hybrid. Install with:\n"
                "  pip install dwave-system dwave-hybrid"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create Kerberos sampler: {e}")

    else:
        available_samplers = [
            "Classical MCMC: gibbs, metropolis, parallel_tempering, simulated_annealing",
            "Exact/Quasi-Exact: exact, gumbel_max",
            "Local Search: steepest_descent, tabu, greedy",
            "Baseline: random",
            "Quantum: dwave, advantage",
            "Hybrid: hybrid, hybrid_bqm, hybrid_dqm, kerberos"
        ]
        raise ValueError(
            f"Unknown sampler type: '{sampler_type}'. "
            f"Available options:\n" + "\n".join(f"  - {s}" for s in available_samplers)
        )


def get_sampler_info(sampler_type: str) -> str:
    """
    Get information about a specific sampler type.

    Args:
        sampler_type: Type of sampler

    Returns:
        String with sampler description and characteristics
    """
    info = {
        "simulated_annealing": """
Simulated Annealing Sampler (Classical MCMC)
- Type: Stochastic metaheuristic
- Best for: General-purpose sampling, medium to large problems
- Speed: Moderate
- Quality: Good quality samples with proper parameters
- Parameters: beta_range, proposal_acceptance_criteria, num_reads
- Cost: Free (runs locally)
""",
        "tabu": """
Tabu Sampler (Tabu Search)
- Type: Deterministic metaheuristic with memory
- Best for: Optimization problems, finding low-energy states
- Speed: Fast
- Quality: Good for optimization, less diverse samples
- Parameters: tenure (memory length), timeout
- Cost: Free (runs locally)
""",
        "steepest_descent": """
Steepest Descent Sampler (Local Search)
- Type: Deterministic hill climbing
- Best for: Fast local optimization, refinement
- Speed: Very fast
- Quality: Local optima only, not diverse
- Parameters: None
- Cost: Free (runs locally)
""",
        "greedy": """
Greedy Sampler
- Type: Deterministic greedy algorithm
- Best for: Quick approximate solutions
- Speed: Very fast
- Quality: Low, greedy local decisions
- Parameters: None
- Cost: Free (runs locally)
""",
        "exact": """
Exact Solver (Brute Force)
- Type: Exhaustive enumeration
- Best for: Small problems (~20 variables or less), verification
- Speed: Exponential in problem size (VERY SLOW for large problems)
- Quality: Perfect, explores entire state space
- Parameters: None
- Cost: Free (runs locally)
WARNING: Only use for very small problems!
""",
        "random": """
Random Sampler
- Type: Uniform random sampling
- Best for: Baseline comparison, testing
- Speed: Very fast
- Quality: Poor, no optimization
- Parameters: None
- Cost: Free (runs locally)
""",
        "gibbs": """
Gibbs MCMC Sampler
- Type: Markov Chain Monte Carlo with Gibbs updates
- Best for: High-quality sampling from Boltzmann distributions, accurate probability estimation
- Speed: Moderate (depends on num_sweeps and burn_in)
- Quality: Excellent for converged chains, theoretically correct sampling
- Parameters: num_sweeps (default: 1000), burn_in (default: 100), thinning (default: 1), randomize_order (default: True)
- Cost: Free (runs locally)
- Note: True MCMC sampler that samples from the exact Boltzmann distribution (after sufficient burn-in)
""",
        "metropolis": """
Metropolis-Hastings MCMC Sampler
- Type: Markov Chain Monte Carlo with single-bit flip proposals
- Best for: General MCMC sampling with temperature control
- Speed: Moderate (similar to Gibbs)
- Quality: Excellent for converged chains, theoretically correct
- Parameters: temperature (default: 1.0), num_sweeps (default: 1000), burn_in (default: 100), thinning (default: 1)
- Cost: Free (runs locally)
- Note: Standard Metropolis with symmetric proposals, temperature parameter allows exploration control
""",
        "parallel_tempering": """
Parallel Tempering (Replica Exchange) MCMC Sampler
- Type: Advanced MCMC with multiple temperature replicas
- Best for: Complex energy landscapes, avoiding local minima, high-quality sampling
- Speed: Slower than single-chain MCMC (runs multiple replicas)
- Quality: Excellent, superior mixing properties compared to single-temperature MCMC
- Parameters: num_replicas (default: 8), T_min (default: 1.0), T_max (default: 4.0), swap_interval (default: 10), num_sweeps, burn_in, thinning
- Cost: Free (runs locally)
- Note: Best for difficult sampling problems with multiple modes or barriers
""",
        "gumbel_max": """
Gumbel-Max Exact Sampler
- Type: Exact sampling via Gumbel-max reparameterization
- Best for: Small problems requiring exact independent samples
- Speed: O(2^N) enumeration (only feasible for N <= 20)
- Quality: Perfect - generates exact independent samples from Boltzmann distribution
- Parameters: max_variables (default: 20), num_reads
- Cost: Free (runs locally)
- Note: Generates truly independent samples (no autocorrelation), but limited to small N
""",
        "dwave": """
D-Wave Quantum Annealer (QPU)
- Type: Quantum annealing hardware
- Best for: Large optimization problems, quantum advantage exploration
- Speed: Fast (~20μs annealing time, but includes overhead)
- Quality: Excellent for optimization, quantum sampling
- Parameters: solver (optional, specify QPU), num_reads
- Requirements: D-Wave Leap account, API token, QPU access
- Cost: Paid (requires Leap subscription or free tier quota)
- Note: Automatically uses EmbeddingComposite for graph embedding
""",
        "advantage": """
D-Wave Advantage Quantum System (Alias for 'dwave')
- Type: Quantum annealing hardware (latest generation)
- Best for: Large-scale optimization, >5000 variables
- Speed: Fast quantum annealing
- Quality: State-of-the-art quantum performance
- Parameters: solver (optional), num_reads
- Requirements: D-Wave Leap account with Advantage access
- Cost: Paid (requires Leap subscription)
""",
        "hybrid": """
D-Wave Leap Hybrid Solver (BQM)
- Type: Cloud-based hybrid classical-quantum solver
- Best for: Large problems (up to millions of variables)
- Speed: Typically seconds to minutes (problem-dependent)
- Quality: Excellent, combines quantum and classical strengths
- Parameters: solver (optional), time_limit (optional)
- Requirements: D-Wave Leap account, API token
- Cost: Paid (requires Leap subscription or free tier quota)
- Note: Ideal for production workloads
""",
        "hybrid_bqm": """
D-Wave Leap Hybrid BQM Solver (Alias for 'hybrid')
- Type: Cloud-based hybrid solver for Binary Quadratic Models
- Best for: Large-scale BQM problems
- Speed: Fast (typically < 1 minute for most problems)
- Quality: Near-optimal solutions
- Parameters: solver (optional), time_limit
- Requirements: D-Wave Leap account
- Cost: Paid
""",
        "hybrid_dqm": """
D-Wave Leap Hybrid DQM Solver
- Type: Cloud-based hybrid solver for Discrete Quadratic Models
- Best for: Discrete optimization problems
- Speed: Problem-dependent (seconds to minutes)
- Quality: High-quality discrete solutions
- Parameters: time_limit (optional)
- Requirements: D-Wave Leap account
- Cost: Paid
- Note: For discrete variable problems only
""",
        "kerberos": """
Kerberos Hybrid Sampler (QPU + Classical)
- Type: Hybrid workflow combining QPU and classical methods
- Best for: Problems benefiting from iterative QPU refinement
- Speed: Multiple QPU calls (slower but higher quality)
- Quality: Very high, iterative improvement
- Parameters: max_iter (iterations), max_time (timeout)
- Requirements: D-Wave Leap account with QPU access, dwave-hybrid
- Cost: Paid (uses QPU time)
- Note: Advanced hybrid algorithm for complex problems
"""
    }

    sampler_type = sampler_type.lower().replace("-", "_").replace(" ", "_")
    return info.get(sampler_type, f"No information available for sampler type: {sampler_type}")


def list_available_samplers() -> None:
    """Print information about all available samplers."""
    mcmc_samplers = [
        "gibbs",
        "metropolis",
        "parallel_tempering",
        "simulated_annealing"
    ]

    exact_samplers = [
        "exact",
        "gumbel_max"
    ]

    local_search_samplers = [
        "steepest_descent",
        "tabu",
        "greedy"
    ]

    baseline_samplers = [
        "random"
    ]

    quantum_samplers = [
        "dwave",
        "advantage"
    ]

    hybrid_samplers = [
        "hybrid",
        "hybrid_bqm",
        "hybrid_dqm",
        "kerberos"
    ]

    print("=" * 80)
    print("CLASSICAL MCMC SAMPLERS (Free, run locally)")
    print("=" * 80)
    for sampler_type in mcmc_samplers:
        print(get_sampler_info(sampler_type))
        print("-" * 80)

    print("\n" + "=" * 80)
    print("EXACT/QUASI-EXACT SAMPLERS (Free, run locally, small N only)")
    print("=" * 80)
    for sampler_type in exact_samplers:
        print(get_sampler_info(sampler_type))
        print("-" * 80)

    print("\n" + "=" * 80)
    print("LOCAL SEARCH / OPTIMIZATION (Free, run locally)")
    print("=" * 80)
    for sampler_type in local_search_samplers:
        print(get_sampler_info(sampler_type))
        print("-" * 80)

    print("\n" + "=" * 80)
    print("BASELINE SAMPLERS (Free, run locally)")
    print("=" * 80)
    for sampler_type in baseline_samplers:
        print(get_sampler_info(sampler_type))
        print("-" * 80)

    print("\n" + "=" * 80)
    print("D-WAVE QUANTUM SAMPLERS (Requires Leap account)")
    print("=" * 80)
    for sampler_type in quantum_samplers:
        print(get_sampler_info(sampler_type))
        print("-" * 80)

    print("\n" + "=" * 80)
    print("D-WAVE HYBRID SAMPLERS (Requires Leap account)")
    print("=" * 80)
    for sampler_type in hybrid_samplers:
        print(get_sampler_info(sampler_type))
        print("-" * 80)


if __name__ == "__main__":
    # Demo: list all available samplers
    list_available_samplers()
