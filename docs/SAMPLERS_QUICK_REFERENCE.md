# Samplers Quick Reference

Quick reference guide for all available samplers in the BM Practice project.

## Available Samplers

### Classical Samplers (Free, Local)

| Sampler | Type | Quality | Speed | Best For |
|---------|------|---------|-------|----------|
| **gibbs** | MCMC | ⭐⭐⭐⭐⭐ | Medium | **Probability sampling** |
| **simulated_annealing** | MCMC | ⭐⭐⭐⭐ | Medium | Optimization |
| **steepest_descent** | Local Search | ⭐⭐ | Fast | Quick local optimization |
| **exact** | Brute Force | ⭐⭐⭐⭐⭐ | Very Slow* | Small problems (N ≤ 20) |
| **random** | Uniform | ⭐ | Very Fast | Baseline comparison |
| ~~tabu~~ | Tabu Search | ⭐⭐⭐ | Fast | (has parameter issues) |
| ~~greedy~~ | Greedy | ⭐ | Very Fast | (not available) |

*Exponential complexity O(2^N)

### Quantum Samplers (Requires D-Wave Leap)

| Sampler | Description | Access Required |
|---------|-------------|-----------------|
| **dwave** | D-Wave quantum annealer (QPU) | Leap account + QPU access |
| **advantage** | D-Wave Advantage (alias for dwave) | Leap account + QPU access |

### Hybrid Samplers (Requires D-Wave Leap)

| Sampler | Description | Access Required |
|---------|-------------|-----------------|
| **hybrid** | LeapHybridSampler (general) | Leap account |
| **hybrid_bqm** | LeapHybridBQMSampler (alias) | Leap account |
| **hybrid_dqm** | LeapHybridDQMSampler (discrete) | Leap account |
| **kerberos** | QPU + classical hybrid | Leap account + QPU access |

## Usage in Config

### Basic Configuration

```yaml
sampler:
  type: "gibbs"  # Choose from list above
  params:
    num_reads: 1000
```

### Gibbs Sampler (Recommended for Probability Sampling)

```yaml
sampler:
  type: "gibbs"
  params:
    num_reads: 1000      # Number of samples
    num_sweeps: 1000     # MCMC sweeps per chain
    burn_in: 100         # Equilibration period
    thinning: 1          # Keep every nth sample
    randomize_order: true  # Shuffle variable updates
```

**When to use:** Accurate sampling from Boltzmann distribution, KL divergence minimization, probability estimation

### Simulated Annealing

```yaml
sampler:
  type: "simulated_annealing"
  params:
    num_reads: 1000
    beta_range: [1.0, 1.0]  # Temperature schedule [min, max]
    proposal_acceptance_criteria: "Gibbs"  # or "Metropolis"
```

**When to use:** Finding low-energy states, optimization problems, general-purpose sampling

### Steepest Descent

```yaml
sampler:
  type: "steepest_descent"
  params:
    num_reads: 1000
```

**When to use:** Fast local optimization, quick approximate solutions

### Exact Solver

```yaml
sampler:
  type: "exact"
  params:
    num_reads: 1000  # Will return all 2^N states
```

**When to use:** Small problems (N ≤ 20), verification, perfect sampling

**WARNING:** Exponential complexity! Only for very small problems.

### Random Sampler

```yaml
sampler:
  type: "random"
  params:
    num_reads: 1000
```

**When to use:** Baseline comparison, testing

### D-Wave Quantum

```yaml
sampler:
  type: "dwave"  # or "advantage"
  params:
    num_reads: 1000
    solver: null  # Optional: specify QPU name
```

**Requirements:**
- D-Wave Leap account
- API token configured (`dwave config create`)
- QPU access

### D-Wave Hybrid

```yaml
sampler:
  type: "hybrid"  # or "hybrid_bqm"
  params:
    time_limit: 5  # Seconds
```

**Requirements:**
- D-Wave Leap account
- API token configured

## Quick Decision Guide

### I want to...

**...accurately sample from the Boltzmann distribution**
→ Use `gibbs` (best quality, theoretically correct)

**...find low-energy states / optimize**
→ Use `simulated_annealing` or `steepest_descent`

**...benchmark my model / compute KL divergence**
→ Use `gibbs` (gold standard) or `exact` (small problems only)

**...generate training data**
→ Use `gibbs` (high quality) or `simulated_annealing` (good balance)

**...test my code quickly**
→ Use `random` (very fast) or `steepest_descent` (fast)

**...verify correctness on small problem**
→ Use `exact` (perfect but slow)

**...access quantum hardware**
→ Use `dwave` or `advantage` (requires Leap account)

**...solve very large problems (N > 1000)**
→ Use `hybrid` or `hybrid_bqm` (requires Leap account)

## Benchmark Results

From empirical testing (2-4 variables, 1000 samples):

| Sampler | Mean KL Divergence | Relative Quality |
|---------|-------------------:|------------------|
| **Gibbs** | **0.13** | 🥇 Best (10x better than random) |
| Random | 1.49 | 🥉 Baseline |
| Simulated Annealing | 13.84 | ❌ Poor for sampling |

**Key Finding:** For probability sampling tasks, Gibbs is ~100x better than SA!

## Common Parameters

### All Samplers
- `num_reads` - Number of samples to generate

### Gibbs Only
- `num_sweeps` (default: 1000) - MCMC iterations
- `burn_in` (default: 100) - Equilibration period
- `thinning` (default: 1) - Sample thinning factor
- `randomize_order` (default: true) - Randomize updates

### Simulated Annealing Only
- `beta_range` - Temperature schedule [min, max]
- `proposal_acceptance_criteria` - "Gibbs" or "Metropolis"

### Tabu Only
- `tenure` - Tabu list size
- `timeout` - Time limit in seconds

### D-Wave Samplers
- `solver` - Specific solver name (optional)
- `time_limit` - For hybrid samplers (seconds)

## Performance Comparison

| Sampler | Runtime (N=10) | Quality | Memory | Parallelizable |
|---------|----------------|---------|--------|----------------|
| Gibbs | ~0.5s | ⭐⭐⭐⭐⭐ | Low | ✅ (chains) |
| SA | ~0.3s | ⭐⭐⭐ | Low | ✅ |
| Steepest | ~0.1s | ⭐⭐ | Low | ✅ |
| Exact | ~1s | ⭐⭐⭐⭐⭐ | O(2^N) | ❌ |
| Random | ~0.01s | ⭐ | Low | ✅ |

## Command Line Usage

List all available samplers:
```bash
python -m utils.sampler_factory
```

Get info on specific sampler:
```python
from utils.sampler_factory import get_sampler_info
print(get_sampler_info("gibbs"))
```

## See Also

- [Full Samplers Documentation](SAMPLERS.md)
- [Gibbs Sampler Guide](GIBBS_SAMPLER.md)
- [Benchmark README](../BENCHMARK_README.md)
- [Sampler Factory Code](../utils/sampler_factory.py)
