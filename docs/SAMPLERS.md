# Sampler Configuration Guide

This guide explains how to configure and use different samplers in the Boltzmann Machine training pipeline.

## Overview

The pipeline supports multiple sampler types from the D-Wave Ocean SDK for both data generation and training. Samplers are configurable via the YAML configuration file.

## Available Samplers

### Classical Samplers (Free, Run Locally)

#### 1. Simulated Annealing (Default)
**Type:** `simulated_annealing`

Classical MCMC-based sampler using simulated annealing algorithm.

**Best for:**
- General-purpose sampling
- Medium to large problems
- Good quality samples with proper parameters

**Speed:** Moderate
**Quality:** Good quality samples

**Configuration:**
```yaml
sampler:
  type: "simulated_annealing"
  params:
    beta_range: [1.0, 1.0]
    proposal_acceptance_criteria: "Gibbs"
    num_reads: 5000
```

#### 2. Tabu Search
**Type:** `tabu`

Deterministic metaheuristic with memory to avoid revisiting recent solutions.

**Best for:**
- Optimization problems
- Finding low-energy states quickly
- When you need fast approximate solutions

**Speed:** Fast
**Quality:** Good for optimization, less diverse samples

**Configuration:**
```yaml
sampler:
  type: "tabu"
  params:
    tenure: 20          # Memory length (optional)
    timeout: 20         # Max time in seconds
    num_reads: 1000
```

### 3. Steepest Descent
**Type:** `steepest_descent`

Deterministic hill climbing algorithm that always moves to the best neighboring solution.

**Best for:**
- Fast local optimization
- Refinement of existing solutions
- When you need very fast approximate answers

**Speed:** Very fast
**Quality:** Local optima only, not diverse

**Configuration:**
```yaml
sampler:
  type: "steepest_descent"
  params:
    num_reads: 100
```

### 4. Greedy
**Type:** `greedy`

Deterministic greedy algorithm making locally optimal choices.

**Best for:**
- Quick approximate solutions
- Baseline comparisons
- Very fast sampling

**Speed:** Very fast
**Quality:** Low, greedy local decisions

**Configuration:**
```yaml
sampler:
  type: "greedy"
  params:
    num_reads: 100
```

### 5. Exact Solver
**Type:** `exact`

Exhaustive enumeration of all possible states (brute force).

**Best for:**
- **Very small problems only** (~20 variables or less)
- Verification and testing
- When you need the exact solution

**Speed:** Exponential in problem size (**VERY SLOW** for large problems)
**Quality:** Perfect, explores entire state space

**⚠️ WARNING:** Only use for very small problems! Computational cost grows exponentially.

**Configuration:**
```yaml
sampler:
  type: "exact"
  params: {}
```

#### 6. Random Sampler
**Type:** `random`

Uniform random sampling from the state space.

**Best for:**
- Baseline comparison
- Testing
- When you need uniform random samples

**Speed:** Very fast
**Quality:** Poor, no optimization

**Configuration:**
```yaml
sampler:
  type: "random"
  params:
    num_reads: 1000
```

---

### D-Wave Quantum Samplers (Requires Leap Account)

#### 7. D-Wave Quantum Annealer (QPU)
**Type:** `dwave` or `advantage`

Real quantum annealing hardware from D-Wave Systems.

**Best for:**
- Large optimization problems
- Quantum advantage exploration
- Production workloads with quantum access

**Speed:** Fast (~20μs annealing time, plus overhead for embedding and communication)
**Quality:** Excellent for optimization, quantum sampling distribution

**Requirements:**
- D-Wave Leap account
- API token configured (`dwave config create` or `DWAVE_API_TOKEN` env var)
- QPU access (paid subscription or free tier quota)
- Internet connection

**Configuration:**
```yaml
sampler:
  type: "dwave"  # or "advantage" for Advantage system
  params:
    solver: null  # Optional: specify solver name (e.g., "Advantage_system6.4")
    num_reads: 1000
```

**Notes:**
- Automatically uses `EmbeddingComposite` for graph embedding to QPU topology
- QPU time is metered and costs money (or uses free tier quota)
- Best for problems with >100 variables where quantum speedup is possible

**Example with specific solver:**
```yaml
sampler:
  type: "advantage"
  params:
    solver: "Advantage_system6.4"
    num_reads: 5000
```

---

### D-Wave Hybrid Samplers (Requires Leap Account)

#### 8. Leap Hybrid Solver
**Type:** `hybrid` or `hybrid_bqm`

Cloud-based hybrid classical-quantum solver that automatically partitions problems.

**Best for:**
- Large problems (up to millions of variables)
- Production workloads
- When you need high-quality solutions without manual tuning

**Speed:** Typically seconds to minutes (problem-dependent)
**Quality:** Excellent, combines classical and quantum strengths

**Requirements:**
- D-Wave Leap account with hybrid solver access
- API token configured
- Internet connection

**Configuration:**
```yaml
sampler:
  type: "hybrid"
  params:
    time_limit: 5  # Optional: max runtime in seconds
```

**Notes:**
- Ideal for BM training where graph size is large
- No embedding required - handles topology automatically
- More cost-effective than pure QPU for large problems

#### 9. Kerberos Hybrid Sampler
**Type:** `kerberos`

Advanced hybrid workflow combining QPU with classical refinement through iterative calls.

**Best for:**
- Problems benefiting from iterative QPU refinement
- High-quality solutions worth multiple QPU calls
- Research and experimentation

**Speed:** Multiple QPU calls (slower but higher quality)
**Quality:** Very high through iterative improvement

**Requirements:**
- D-Wave Leap account with QPU access
- `dwave-hybrid` package installed
- Internet connection

**Configuration:**
```yaml
sampler:
  type: "kerberos"
  params:
    max_iter: 10       # Maximum iterations
    max_time: 60       # Maximum time in seconds
```

**Notes:**
- Uses more QPU time than single-shot sampling
- Best for final production runs where quality matters
- Requires `pip install dwave-hybrid`

#### 10. Hybrid DQM Solver
**Type:** `hybrid_dqm`

Specialized hybrid solver for Discrete Quadratic Models (not typically used for BMs).

**Configuration:**
```yaml
sampler:
  type: "hybrid_dqm"
  params:
    time_limit: 5
```

---

## Configuration Locations

Samplers can be configured in two places in `configs/config.yaml`:

### 1. Data Generation Sampler

Used when generating training data from the true model:

```yaml
data:
  dataset_name: "bm_toy_dataset"
  n_samples: 10000
  prefactor: 1.0

  sampler:
    type: "simulated_annealing"
    params:
      beta_range: [1.0, 1.0]
      proposal_acceptance_criteria: "Gibbs"
      num_reads: 5000
```

### 2. Training Sampler

Used during model training for sampling from the learned model:

```yaml
training:
  batch_size: 1000
  n_epochs: 500
  learning_rate: 0.01

  sampler:
    type: "simulated_annealing"
    params:
      beta_range: [1.0, 1.0]
      proposal_acceptance_criteria: "Gibbs"
```

## Example Configurations

### Fast Training (Tabu Search)

For quick experiments or debugging:

```yaml
data:
  sampler:
    type: "tabu"
    params:
      timeout: 10
      num_reads: 1000

training:
  sampler:
    type: "tabu"
    params:
      timeout: 10
```

### High-Quality Sampling (Simulated Annealing)

For production or final experiments:

```yaml
data:
  sampler:
    type: "simulated_annealing"
    params:
      beta_range: [1.0, 1.0]
      proposal_acceptance_criteria: "Gibbs"
      num_reads: 10000

training:
  sampler:
    type: "simulated_annealing"
    params:
      beta_range: [1.0, 1.0]
      proposal_acceptance_criteria: "Gibbs"
```

### Small Problem Verification (Exact Solver)

For tiny models only (e.g., 4-6 variables):

```yaml
true_model:
  n_visible: 4
  n_hidden: 0

data:
  sampler:
    type: "exact"
    params: {}

training:
  sampler:
    type: "exact"
    params: {}
```

### Quantum Sampling (D-Wave QPU)

For large problems with access to D-Wave quantum hardware:

```yaml
data:
  sampler:
    type: "dwave"
    params:
      num_reads: 1000

training:
  sampler:
    type: "advantage"  # Use latest Advantage system
    params:
      num_reads: 5000
```

### Hybrid Sampling (Production)

For large-scale production workloads:

```yaml
data:
  sampler:
    type: "hybrid"
    params:
      time_limit: 3  # 3 seconds max

training:
  sampler:
    type: "hybrid"
    params:
      time_limit: 5  # 5 seconds max for training
```

### Mixed Classical-Quantum

Use fast classical for data generation, quantum for training:

```yaml
data:
  sampler:
    type: "tabu"  # Fast classical for data generation
    params:
      timeout: 10

training:
  sampler:
    type: "hybrid"  # High-quality hybrid for training
    params:
      time_limit: 5
```

## Programmatic Usage

You can also create samplers programmatically:

```python
from utils import create_sampler, get_sampler_info, list_available_samplers

# Create a specific sampler
sampler = create_sampler('tabu', {'timeout': 20})

# Get information about a sampler
info = get_sampler_info('tabu')
print(info)

# List all available samplers
list_available_samplers()
```

## Choosing the Right Sampler

### Decision Tree:

**Classical Samplers:**
1. **Problem size <= 20 variables** → Consider `exact` for verification
2. **Need very fast sampling** → Use `tabu` or `steepest_descent`
3. **Need high-quality samples** → Use `simulated_annealing` (default)
4. **Baseline comparison** → Use `random`
5. **Quick debugging** → Use `greedy` or `tabu`

**D-Wave Samplers (if you have Leap access):**
6. **Large problems (>100 variables) with QPU access** → Use `dwave` or `advantage`
7. **Very large problems (>1000 variables)** → Use `hybrid`
8. **Production workloads, want best quality** → Use `hybrid` or `kerberos`
9. **Want quantum exploration** → Use `dwave` for direct QPU access

### Sampler Performance Comparison

| Sampler | Speed | Quality | Diversity | Cost | Use Case |
|---------|-------|---------|-----------|------|----------|
| **Classical** | | | | | |
| Exact | Very Slow | Perfect | Complete | Free | Small problems only |
| Simulated Annealing | Moderate | Good | High | Free | General purpose (default) |
| Tabu | Fast | Good | Medium | Free | Fast optimization |
| Steepest Descent | Very Fast | Low | Very Low | Free | Quick local search |
| Greedy | Very Fast | Low | Very Low | Free | Quick approximation |
| Random | Very Fast | Poor | Very High | Free | Baseline/testing |
| **Quantum** | | | | | |
| D-Wave QPU | Fast | Excellent | Medium | Paid | Large problems, quantum research |
| Advantage | Fast | Excellent | Medium | Paid | Large-scale optimization |
| **Hybrid** | | | | | |
| Leap Hybrid | Moderate | Excellent | High | Paid | Production, large problems |
| Kerberos | Slow | Very High | Medium | Paid | Iterative refinement |

## Common Parameters

### num_reads
Number of samples to generate. Higher values give better statistics but take longer.
- Small problems: 100-1000
- Medium problems: 1000-5000
- Large problems: 5000-10000

### beta_range (Simulated Annealing)
Temperature schedule for annealing. `[1.0, 1.0]` means constant temperature.
- Lower values: More exploration
- Higher values: More exploitation

### proposal_acceptance_criteria (Simulated Annealing)
- `"Gibbs"`: Gibbs sampling (default)
- `"Metropolis"`: Metropolis-Hastings criterion

## Troubleshooting

### Sampler ImportError

If you get an import error for a specific sampler:

```bash
pip install dwave-ocean-sdk
```

### Poor Sample Quality

Try:
1. Increase `num_reads`
2. Switch to `simulated_annealing` if using faster samplers
3. Adjust `beta_range` for simulated annealing

### Too Slow

Try:
1. Reduce `num_reads`
2. Switch to `tabu` or `steepest_descent`
3. Ensure problem size is reasonable

### Out of Memory

Reduce `num_reads` or problem size.

### D-Wave Connection Errors

If you get connection errors with D-Wave samplers:

1. **Check API Token:**
   ```bash
   # Configure D-Wave API token
   dwave config create

   # Or set environment variable
   export DWAVE_API_TOKEN="your-token-here"
   ```

2. **Verify Leap Account:**
   - Go to https://cloud.dwavesys.com/leap/
   - Ensure you have an active account
   - Check your quota/subscription status

3. **Test Connection:**
   ```python
   from dwave.system import DWaveSampler
   sampler = DWaveSampler()
   print(f"Connected to: {sampler.solver.name}")
   ```

4. **Install Required Packages:**
   ```bash
   # For QPU and Hybrid samplers
   pip install dwave-system

   # For Kerberos sampler
   pip install dwave-hybrid
   ```

### Solver Not Found

If you get "solver not found" errors:

```python
# List available solvers
from dwave.system import DWaveSampler
print(DWaveSampler().solver.available_solvers())
```

## D-Wave Leap Setup Guide

### 1. Create Leap Account

1. Go to https://cloud.dwavesys.com/leap/
2. Sign up for a free account (includes free QPU/Hybrid minutes)
3. Verify your email

### 2. Get API Token

1. Log in to Leap dashboard
2. Click on your profile → API Token
3. Copy your API token

### 3. Configure Local Environment

**Option A: Using dwave command**
```bash
dwave config create

# Follow prompts:
# - API token: [paste your token]
# - Default solver: [leave blank or specify]
# - Default region: [leave blank]
```

**Option B: Using environment variable**
```bash
# Linux/Mac
export DWAVE_API_TOKEN="DEV-your-token-here"

# Windows (PowerShell)
$env:DWAVE_API_TOKEN="DEV-your-token-here"

# Windows (CMD)
set DWAVE_API_TOKEN=DEV-your-token-here
```

**Option C: In Python code** (not recommended for production)
```python
import os
os.environ['DWAVE_API_TOKEN'] = 'DEV-your-token-here'
```

### 4. Verify Setup

```python
from dwave.system import LeapHybridSampler

# Test connection
sampler = LeapHybridSampler()
print(f"Connected to: {sampler.solver.name}")
print("Setup successful!")
```

### 5. Check Quota

Monitor your usage:
1. Go to https://cloud.dwavesys.com/leap/
2. Check Dashboard → Usage
3. Free tier typically includes:
   - 20 seconds/month of QPU time
   - 20 minutes/month of Hybrid solver time

## References

- [D-Wave Ocean SDK Documentation](https://docs.ocean.dwavesys.com/)
- [D-Wave Leap](https://cloud.dwavesys.com/leap/)
- [Ocean SDK Installation](https://docs.ocean.dwavesys.com/en/stable/overview/install.html)
- [dimod Samplers](https://docs.ocean.dwavesys.com/en/stable/docs_dimod/reference/samplers.html)
- [Hybrid Solvers](https://docs.ocean.dwavesys.com/en/stable/docs_system/reference/samplers.html#leap-hybrid-samplers)
- [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing)
- [Tabu Search](https://en.wikipedia.org/wiki/Tabu_search)
