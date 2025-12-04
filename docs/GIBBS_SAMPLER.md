# Gibbs MCMC Sampler

A pure Gibbs sampling implementation for Boltzmann Machines, compatible with D-Wave's dimod interface.

## Overview

The Gibbs sampler implements Markov Chain Monte Carlo (MCMC) sampling using the Gibbs algorithm. Unlike optimization-focused samplers (like simulated annealing or steepest descent), Gibbs sampling is designed to **accurately sample from the Boltzmann distribution**.

## Algorithm

Gibbs sampling updates one variable at a time by sampling from its conditional distribution:

```
For each iteration:
  For each variable i:
    Sample x_i ~ p(x_i | x_{-i})
```

where the conditional probability for spin variables is:

```
p(s_i = +1 | s_{-i}) = 1 / (1 + exp(-2 * h_i))
```

and `h_i = linear_bias[i] + sum_j J[i,j] * s[j]` is the local field.

## Usage

### Basic Usage

```python
from utils.gibbs_sampler import GibbsSamplerSpin

# Create sampler
sampler = GibbsSamplerSpin(
    num_sweeps=1000,    # Number of full sweeps through variables
    burn_in=100,        # Discard first N sweeps
    thinning=1,         # Keep every nth sample
    randomize_order=True  # Randomize variable update order
)

# Sample from a BQM
sampleset = sampler.sample(bqm, num_reads=10)
```

### Using with GRBM (D-Wave PyTorch Plugin)

```python
from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine as GRBM
from utils.sampler_factory import create_sampler

# Create a Boltzmann Machine
model = GRBM(nodes, edges, hidden_nodes, linear, quadratic)

# Create Gibbs sampler via factory
sampler = create_sampler("gibbs", {
    "num_sweeps": 1000,
    "burn_in": 100,
    "thinning": 1,
    "randomize_order": True
})

# Sample from the model
samples = model.sample(
    sampler=sampler,
    sample_params={"num_reads": 1000},
    as_tensor=True
)
```

### In Configuration Files

```yaml
sampler:
  type: "gibbs"
  params:
    num_sweeps: 1000
    burn_in: 100
    thinning: 1
    randomize_order: true
    num_reads: 1000
```

## Parameters

### `num_sweeps` (default: 1000)
Number of complete sweeps through all variables after burn-in. More sweeps = more samples but longer runtime.

**Recommendation:**
- Small problems (N < 10): 1000-5000 sweeps
- Medium problems (10 ≤ N < 20): 5000-10000 sweeps
- Large problems (N ≥ 20): 10000+ sweeps

### `burn_in` (default: 100)
Number of initial sweeps to discard before collecting samples. This allows the chain to reach equilibrium.

**Recommendation:**
- Start with 100-1000 burn-in sweeps
- For complex distributions, increase to 1000-5000
- Monitor autocorrelation to verify convergence

### `thinning` (default: 1)
Keep every nth sample to reduce autocorrelation. Higher values = more independent samples but fewer total samples.

**Recommendation:**
- Default (1): Keep all samples
- For highly autocorrelated chains: Use 5-10
- For critical applications: Monitor autocorrelation time and set thinning accordingly

### `randomize_order` (default: True)
Whether to randomly permute the variable update order each sweep.

**Recommendation:**
- True (default): Better mixing, reduces systematic bias
- False: Faster but may have systematic patterns

### `num_reads` (default: 1)
Number of independent Markov chains to run. Each chain starts from a random initial state.

**Recommendation:**
- Use multiple chains (10-100) for better coverage
- Each chain is independent and can be run in parallel (future optimization)

## Performance

### Benchmark Results

From the test benchmark (2-4 variables, 1000 samples):

| Sampler | Mean KL Divergence | Quality |
|---------|-------------------|---------|
| **Gibbs** | **0.13** | Excellent |
| Random | 1.49 | Poor |
| Simulated Annealing | 13.84 | Very Poor |

**Key Insights:**
- Gibbs provides **10x better** sampling quality than random
- Gibbs provides **100x better** sampling quality than simulated annealing (for this task)
- Gibbs produces theoretically correct samples from the Boltzmann distribution

### Runtime Complexity

- **Per sweep:** O(N²) where N is the number of variables
  - Each variable update: O(N) to compute local field
  - N variables per sweep: O(N²)

- **Total time:** O(N² × num_sweeps × num_reads)

**Example runtimes (approximate):**
- N=4, 1000 sweeps, 1 read: ~0.1 seconds
- N=10, 1000 sweeps, 1 read: ~0.5 seconds
- N=20, 1000 sweeps, 1 read: ~2 seconds
- N=50, 10000 sweeps, 10 reads: ~60 seconds

## When to Use Gibbs Sampling

### ✅ Use Gibbs When:
- You need **accurate probability sampling** from the Boltzmann distribution
- You want to estimate **expectation values** or **marginal probabilities**
- You're benchmarking other samplers (Gibbs is the gold standard)
- You need **theoretically correct** MCMC samples
- Problem size is small to medium (N < 50 for reasonable runtime)

### ❌ Don't Use Gibbs When:
- You only need **low-energy states** (use optimization samplers instead)
- You need **very fast** sampling (use random or greedy)
- Problem is very large (N > 100) and you need many samples (runtime becomes prohibitive)
- You have access to quantum hardware and want to leverage it

## Comparison with Other Samplers

| Sampler | Purpose | Quality | Speed | Use Case |
|---------|---------|---------|-------|----------|
| **Gibbs** | MCMC sampling | Excellent | Moderate | Accurate probability estimation |
| Simulated Annealing | Optimization | Good* | Moderate | Finding low-energy states |
| Steepest Descent | Optimization | Local optima | Fast | Quick local optimization |
| Exact | Exhaustive | Perfect | Very Slow | Small problems only (N < 20) |
| Random | Baseline | Poor | Very Fast | Testing baseline |

*Note: Simulated annealing is good for optimization but poor for probability sampling (as shown in benchmarks)

## Implementation Details

### Spin Encoding
The sampler uses **spin encoding** {-1, +1} which is natural for Boltzmann Machines:

```
Energy: E(s) = -sum_i h_i * s_i - sum_{i<j} J_{i,j} * s_i * s_j
```

### Sampling Strategy
1. **Initialize:** Random spin configuration
2. **Burn-in:** Run `burn_in` sweeps to reach equilibrium
3. **Sample:** Run `num_sweeps` sweeps, collecting samples with `thinning`
4. **Return:** SampleSet compatible with dimod interface

### Convergence Diagnostics
To verify convergence, monitor:
- **Autocorrelation time:** Should be << num_sweeps
- **Multiple chains:** Should produce similar distributions
- **Energy trace:** Should stabilize after burn-in

## Examples

### Example 1: Basic Sampling

```python
from utils.gibbs_sampler import gibbs_sample
import dimod

# Create a simple BQM
bqm = dimod.BinaryQuadraticModel(
    {0: -1, 1: -1},  # linear biases
    {(0, 1): -1},     # quadratic bias
    0.0,              # offset
    dimod.SPIN
)

# Sample
sampleset = gibbs_sample(
    bqm,
    num_reads=100,
    num_sweeps=1000,
    burn_in=100
)

print(f"Samples: {len(sampleset)}")
print(f"Lowest energy: {sampleset.first.energy}")
```

### Example 2: Benchmarking

See [BENCHMARK_README.md](../BENCHMARK_README.md) for complete benchmarking setup.

```bash
# Quick benchmark with Gibbs
python main.py --mode benchmark --config benchmark_configs/config_benchmark_test.yaml
```

### Example 3: Data Generation

Use Gibbs for generating high-quality training data:

```yaml
data_generation:
  sampler:
    type: "gibbs"
    params:
      num_sweeps: 5000
      burn_in: 500
      thinning: 1
      num_reads: 10000
```

## Theoretical Background

Gibbs sampling is a MCMC method that is **guaranteed to converge** to the target distribution (Boltzmann distribution) under mild conditions:

1. **Ergodicity:** The chain can reach any state from any other state
2. **Detailed Balance:** The chain satisfies detailed balance condition
3. **Sufficient burn-in:** Enough iterations to reach equilibrium

For Boltzmann Machines, these conditions are satisfied, making Gibbs sampling **theoretically correct**.

## References

- Geman, S., & Geman, D. (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. IEEE Transactions on Pattern Analysis and Machine Intelligence.
- Neal, R. M. (1993). Probabilistic inference using Markov chain Monte Carlo methods. Technical Report CRG-TR-93-1, University of Toronto.

## Advanced Usage

### Custom Burn-in Determination

Monitor energy convergence to determine optimal burn-in:

```python
import matplotlib.pyplot as plt

# Track energy during sampling
energies = []
for sweep in range(total_sweeps):
    # ... perform sweep ...
    energies.append(current_energy)

# Plot to identify burn-in period
plt.plot(energies)
plt.axvline(x=burn_in, color='r', linestyle='--', label='Burn-in')
plt.xlabel('Sweep')
plt.ylabel('Energy')
plt.legend()
plt.show()
```

### Autocorrelation Analysis

Compute autocorrelation to optimize thinning:

```python
import numpy as np

def autocorrelation(samples, max_lag=100):
    """Compute autocorrelation of samples."""
    acf = np.correlate(samples - samples.mean(),
                      samples - samples.mean(),
                      mode='full')
    acf = acf[len(acf)//2:]
    acf /= acf[0]
    return acf[:max_lag]

# Use to determine optimal thinning
acf = autocorrelation(sample_energies)
decorrelation_time = np.argmax(acf < 0.1)  # Time to decorrelate
recommended_thinning = max(1, decorrelation_time // 2)
```

## Troubleshooting

### Issue: High KL Divergence
**Solution:** Increase `num_sweeps` or `burn_in`

### Issue: Slow Convergence
**Solution:**
- Increase `burn_in`
- Use `randomize_order=True`
- Check for multimodal distributions

### Issue: Too Slow
**Solution:**
- Reduce `num_sweeps`
- Increase `thinning`
- Reduce `num_reads`
- Consider approximate samplers for large problems

## License

This implementation is part of the BM Practice project and follows the same license.
