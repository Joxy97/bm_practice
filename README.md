# Boltzmann Machine Tutorial: Reverse Engineering a Quadratic Model

A pedagogical implementation for learning how to build and train Boltzmann Machines using D-Wave's sampling capabilities.

## Overview

This tutorial demonstrates:

1. **Define TRUE model**: Manually set biases and weights for a 4-variable fully-connected BM
2. **Sample training data**: Use D-Wave's simulated annealing to sample from the true model
3. **Train NEW model**: Initialize a random BM and train it on the sampled data
4. **Compare results**: Visualize how well the learned model recovered the true parameters

## Installation

```bash
pip install -r requirements.txt
```

## Running the Tutorial

```bash
python boltzmann_machine_tutorial.py
```

## What You'll Learn

### Boltzmann Machine Basics

A Boltzmann Machine defines an energy function over binary variables:

```
E(v) = -Σᵢ bᵢvᵢ - Σᵢ<ⱼ Wᵢⱼvᵢvⱼ
```

Where:
- `vᵢ ∈ {0,1}` are binary variables
- `bᵢ` are biases (linear terms)
- `Wᵢⱼ` are interaction weights (quadratic terms)

### Training Process

The code uses **maximum likelihood estimation** for a fully visible BM:

1. **Data statistics**: Calculate empirical means and correlations from training data
2. **Model statistics**: Sample from current model to estimate expectations
3. **Gradient update**: Move parameters to match data statistics

```
∂L/∂bᵢ = ⟨vᵢ⟩_data - ⟨vᵢ⟩_model
∂L/∂Wᵢⱼ = ⟨vᵢvⱼ⟩_data - ⟨vᵢvⱼ⟩_model
```

### D-Wave Integration

The tutorial uses D-Wave's simulated annealing sampler to:
- Sample from the true model (generate training data)
- Sample from the learned model (estimate model statistics during training)

This is converted to QUBO (Quadratic Unconstrained Binary Optimization) format that D-Wave can process.

## Output Files

The script generates 5 visualizations:

1. `true_bm_parameters.png` - The manually defined true model
2. `training_data_statistics.png` - Analysis of sampled training data
3. `learned_bm_parameters.png` - The trained model parameters
4. `training_history.png` - Loss and parameter evolution during training
5. `true_vs_learned_comparison.png` - Side-by-side comparison of true vs learned

## Key Parameters

You can adjust these in the `main()` function:

- `num_training_samples`: Number of samples to generate (default: 500)
- `learning_rate`: Gradient descent step size (default: 0.05)
- `num_epochs`: Training iterations (default: 50)
- `sample_size`: Samples per gradient estimate (default: 200)

## Understanding the Code

### BoltzmannMachine Class

- `set_parameters()`: Manually define biases and weights
- `energy()`: Calculate energy of a configuration
- `to_qubo()`: Convert to D-Wave QUBO format
- `sample_dwave()`: Sample configurations using simulated annealing
- `visualize_parameters()`: Plot biases and weights

### BoltzmannMachineTrainer Class

- `train()`: Maximum likelihood training loop
- `plot_training_history()`: Visualize learning progress

## Pedagogical Notes

This implementation prioritizes clarity over efficiency:

- ✓ Explicit parameter setting
- ✓ Clear visualization at each step
- ✓ Full control over training process
- ✓ No hidden abstractions
- ✓ Comments explain the theory

Perfect for understanding BM fundamentals before moving to production libraries!
