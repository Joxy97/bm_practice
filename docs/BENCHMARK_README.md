# Sampler Benchmarking Tool

This tool benchmarks different sampling methods for Boltzmann Machines by computing the KL divergence between the empirical distribution from samples and the exact Boltzmann distribution.

## Overview

The benchmarking pipeline:
1. Creates a fully visible dense Boltzmann Machine with N variables
2. Computes the exact Boltzmann distribution p(x) in closed form
3. Generates samples using different samplers
4. Computes the empirical distribution p-hat(x) from samples
5. Calculates KL divergence: D_KL(p || p-hat) = Σ p(x) log(p(x) / p-hat(x))
6. Repeats across multiple problem sizes
7. Creates comprehensive visualizations

## Usage

### Quick Test Run

For a quick test with smaller problem sizes:

```bash
python main.py --mode benchmark --config configs/config_benchmark_test.yaml
```

### Full Benchmark Run

For the full benchmark across all problem sizes (2-10 variables):

```bash
python main.py --mode benchmark --config configs/config.yaml
```

## Configuration

Edit the `benchmark` section in your config file:

```yaml
benchmark:
  # Samplers to benchmark
  samplers:
    - "simulated_annealing"
    - "steepest_descent"
    - "exact"
    - "random"

  # Range of problem sizes (number of variables)
  n_variables_range: [2, 3, 4, 5, 6, 7, 8, 9, 10]

  # Number of samples per benchmark run
  n_samples: 10000

  # Model configuration for benchmarking (always dense FVBM)
  model_type: "fvbm"
  connectivity: "dense"
  linear_bias_scale: 1.0
  quadratic_weight_scale: 1.5
```

## Available Samplers

### Working Samplers:
- `"simulated_annealing"` - MCMC-based annealing (good quality)
- `"steepest_descent"` - Local search optimization
- `"exact"` - Brute force enumeration (only feasible for small N)
- `"random"` - Baseline uniform random sampling

### Not Currently Supported:
- `"tabu"` - Issues with parameter compatibility
- `"greedy"` - Not available in current D-Wave version

## Output

The benchmark produces:

### 1. Data Files
- `benchmark_results.csv` - Full results table
- `benchmark_summary.csv` - Summary statistics per sampler

### 2. Visualizations
- `kl_vs_variables.png` - Line plot of KL divergence vs problem size
- `kl_heatmap.png` - Heatmap of KL divergence across samplers and sizes
- `average_kl_per_sampler.png` - Bar chart of average performance
- `kl_distribution.png` - Box plot of KL divergence distributions
- `kl_vs_variables_logscale.png` - Log-scale comparison
- `relative_performance.png` - Relative performance and ranking

## Metrics

### KL Divergence
The primary metric is KL divergence from the true distribution:
- **Lower is better** (0 is perfect)
- Measures how well the sampler approximates the true Boltzmann distribution
- Computed as: D_KL(p || p-hat) = Σ p(x) log(p(x) / p-hat(x))

### Interpretation
- KL ≈ 0: Excellent sampling quality
- KL < 1: Good sampling quality
- KL > 5: Poor sampling quality
- KL increases with problem size as the distribution becomes more complex

## Implementation Details

### Exact Distribution Computation
For a Boltzmann Machine with energy E(x) = -Σ h_i x_i - Σ J_ij x_i x_j:

```
p(x) = exp(-E(x)) / Z
```

where Z is the partition function computed by enumerating all 2^N states.

**Computational Complexity:** O(2^N)
- Feasible for N ≤ 20 variables
- For the default benchmark (N ≤ 10), this is very fast

### Empirical Distribution Computation
```
p-hat(x) = (1/M) Σ 1(x^(m) = x)
```

where M is the number of samples.

## Example Results

Typical results show:
- **Simulated Annealing**: Best quality, KL divergence grows slowly with N
- **Steepest Descent**: Fast but can get stuck in local minima
- **Random**: Worst quality, used as baseline
- **Exact**: Perfect sampling (KL ≈ 0) but slow for large N

## Notes

- All benchmarks use dense fully visible Boltzmann Machines (FVBM)
- Random seeds are fixed for reproducibility
- Each benchmark run creates a new timestamped output directory
- The exact sampler becomes infeasible for N > 20 due to exponential complexity

## Troubleshooting

### Out of Memory
If you run out of memory for large N:
- Reduce `n_variables_range` to smaller values
- Reduce `n_samples` (though this may affect KL divergence accuracy)

### Slow Performance
- The exact sampler is exponentially slow; avoid for N > 15
- Reduce the number of samplers or problem sizes
- Use the test config for quick validation

## Citation

If using this benchmarking tool, please cite the following:

- D-Wave Ocean SDK: https://github.com/dwavesystems/dwave-ocean-sdk
- PyTorch: https://pytorch.org/
