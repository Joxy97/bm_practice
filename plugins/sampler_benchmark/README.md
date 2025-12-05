# Sampler Benchmark Plugin

Benchmarks and compares different MCMC samplers for Boltzmann Machines.

## Purpose

This plugin is useful for:
- **Performance Testing** - Compare sampling speed across different samplers
- **Accuracy Analysis** - Measure KL divergence from true distribution
- **Convergence Study** - Analyze autocorrelation and effective sample size
- **Sampler Selection** - Choose the best sampler for your use case

## Usage

### Standalone CLI

```bash
python -m plugins.sampler_benchmark.run_benchmark \
  --config plugins/sampler_benchmark/benchmark_config.yaml \
  --output-dir outputs/benchmark/
```

### As Python Module

```python
from plugins.sampler_benchmark import SamplerBenchmark
from plugins.sampler_factory import SamplerFactory
import yaml

# Load config
with open('plugins/sampler_benchmark/benchmark_config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize
factory = SamplerFactory()
benchmark = SamplerBenchmark(config, factory.get_sampler_dict())

# Run benchmark
results = benchmark.run_benchmark()

# Save results and create plots
benchmark.save_results('outputs/benchmark/')
benchmark.create_visualizations('outputs/benchmark/plots/')
```

## Configuration

Edit `benchmark_config.yaml`:

```yaml
benchmark:
  # Samplers to test
  samplers:
    - "gibbs"
    - "metropolis"
    - "gibbs_gpu"
    - "metropolis_gpu"
    - "exact_sampler"

  # Problem sizes
  n_variables_range: [2, 3, 4, 5, 6, 7, 8, 9, 10]

  # Number of samples
  n_samples: 10000

  # Model configuration
  model_type: "fvbm"

# Sampler-specific parameters
sampler_params:
  gibbs:
    num_sweeps: 1000
    burn_in: 100
  gibbs_gpu:
    num_sweeps: 1000
    burn_in: 100
    num_chains: 32
```

## Metrics Computed

### Performance Metrics
- **Sampling Time** - Total time to generate samples
- **Throughput** - Samples per second
- **Effective Sample Size (ESS)** - Independent samples after accounting for autocorrelation

### Accuracy Metrics (for small problems)
- **KL Divergence** - Distance from true distribution
- **Total Variation Distance** - L1 distance between distributions

### Convergence Metrics
- **Autocorrelation** - Correlation between samples at different lags
- **Integrated Autocorrelation Time** - Time to decorrelate

## Output

### Results File
`benchmark_results.json` - Complete metrics for all samplers and problem sizes

### Plots

1. **sampling_time_comparison.png** - Sampling time vs problem size
2. **throughput_comparison.png** - Samples/sec vs problem size
3. **kl_divergence_comparison.png** - KL divergence vs problem size
4. **effective_sample_size.png** - ESS vs problem size
5. **autocorrelation_nX.png** - Autocorrelation plots for each problem size
6. **distribution_comparison_nX.png** - Exact vs empirical distributions (small problems)

### Summary Table
`benchmark_summary.csv` - Tabular summary of all results

## Examples

### Quick Performance Test

```yaml
benchmark:
  samplers: ["gibbs", "gibbs_gpu"]
  n_variables_range: [5, 10, 15, 20]
  n_samples: 5000
  model_type: "fvbm"
```

### Detailed Accuracy Test

```yaml
benchmark:
  samplers: ["gibbs", "metropolis", "parallel_tempering", "exact_sampler"]
  n_variables_range: [2, 3, 4, 5, 6, 7, 8]  # Small for exact comparison
  n_samples: 20000  # More samples for better statistics
  model_type: "fvbm"

sampler_params:
  gibbs:
    num_sweeps: 2000
    burn_in: 500
```

### GPU Sampler Comparison

```yaml
benchmark:
  samplers: ["gibbs_gpu", "metropolis_gpu", "pt_gpu"]
  n_variables_range: [10, 20, 50, 100, 200]
  n_samples: 10000
  model_type: "fvbm"

sampler_params:
  gibbs_gpu:
    num_sweeps: 1000
    burn_in: 100
    num_chains: 64
```

## Tips

1. **Small problems (n <= 10)**: Use exact sampler for ground truth comparison
2. **Medium problems (10 < n <= 50)**: Compare classical and GPU samplers
3. **Large problems (n > 50)**: Use GPU samplers only
4. **Accuracy vs Speed**: Increase `num_sweeps` and `burn_in` for better accuracy
5. **Multiple runs**: Run benchmark multiple times for statistical significance

## Interpreting Results

### Sampling Time
- Lower is better
- GPU samplers should be faster for large problems
- Classical samplers may be faster for very small problems (overhead)

### KL Divergence
- Lower is better (closer to true distribution)
- Should decrease with more sweeps/burn-in
- Exact sampler should have ~0 KL divergence

### Effective Sample Size
- Higher is better (more independent samples)
- ESS << n_samples indicates high autocorrelation
- Good samplers: ESS > 0.5 * n_samples

### Autocorrelation
- Faster decay is better
- Should approach 0 after sufficient lags
- High autocorrelation → need more thinning or longer sweeps

## Integration with Projects

Use benchmark results to choose samplers for your project:

```bash
# 1. Run benchmark
python -m plugins.sampler_benchmark.run_benchmark \
  --config plugins/sampler_benchmark/benchmark_config.yaml \
  --output-dir outputs/benchmark/

# 2. Analyze results, choose best sampler

# 3. Use in your project config
# Edit: projects/my_project/project_config.py
config = BMConfig(
    training=TrainingConfig(
        sampler_name='gibbs_gpu',  # Based on benchmark results
        sampler_params={
            'num_sweeps': 1000,
            'burn_in': 100,
            'num_chains': 32
        }
    )
)
```

## Advanced Usage

### Custom Metrics

Add custom metrics by extending `BenchmarkMetrics`:

```python
from plugins.sampler_benchmark.utils import BenchmarkMetrics

class MyMetrics(BenchmarkMetrics):
    @staticmethod
    def compute_custom_metric(samples):
        # Your metric implementation
        return metric_value
```

### Custom Visualizations

Create custom plots using `BenchmarkVisualizer`:

```python
from plugins.sampler_benchmark.utils import BenchmarkVisualizer

visualizer = BenchmarkVisualizer('outputs/plots/')
visualizer.plot_custom_comparison(results, ...)
```

## Troubleshooting

**GPU samplers fail:**
- Check CUDA availability: `torch.cuda.is_available()`
- Reduce `num_chains` if out of memory
- Use CPU samplers for testing

**Exact sampler too slow:**
- Only use for n_variables <= 10
- Disable exact sampler for larger problems
- Set `compute_exact: false` in config

**High KL divergence:**
- Increase `num_sweeps` and `burn_in`
- Check sampler convergence with autocorrelation plots
- Try different samplers (e.g., parallel tempering)
