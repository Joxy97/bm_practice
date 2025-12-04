# Architecture Terminology Guide

## Overview

This document explains the clean, explicit naming scheme for Boltzmann Machine architectures introduced in Phase 1 refactoring.

## The Naming Scheme

The configuration uses three independent, explicit dimensions:

### 1. Model Type (`model_type`)

Defines the **structure** of the Boltzmann Machine:

- **`"fvbm"`** - Fully Visible Boltzmann Machine
  - Only visible nodes (no hidden layer)
  - Requires `n_hidden = 0`
  - Edges: only v-v (visible-visible)

- **`"rbm"`** - Restricted Boltzmann Machine
  - Bipartite structure with visible and hidden layers
  - Requires `n_hidden > 0`
  - Edges: only v-h (visible-hidden), no intra-layer connections

- **`"sbm"`** - Standard/General Boltzmann Machine
  - Full general structure with both visible and hidden layers
  - Requires `n_hidden > 0`
  - Edges: all types allowed - v-v, v-h, and h-h (hidden-hidden)

### 2. Connectivity Pattern (`connectivity`)

Defines the **density** of edges within the allowed structure:

- **`"dense"`** - All allowed edges exist
  - For FVBM: Complete graph among visible nodes
  - For RBM: Complete bipartite graph

- **`"sparse"`** - Random subset of allowed edges
  - Controlled by `connectivity_density` parameter
  - Ensures graph connectivity (every node has ≥1 edge)

### 3. Connectivity Density (`connectivity_density`)

Only used when `connectivity = "sparse"`:
- Range: 0.0 to 1.0
- Fraction of allowed edges to include
- Example: `0.3` means 30% of possible edges

## Complete Configuration Example

```yaml
true_model:
  n_visible: 10
  n_hidden: 0
  model_type: "fvbm"
  connectivity: "dense"
  connectivity_density: 0.7  # Ignored for dense
```

## Model Type vs Connectivity Matrix

| Model Type | Connectivity | n_hidden | Edge Types | Result |
|------------|--------------|----------|------------|--------|
| `fvbm` | `dense` | 0 | v-v | Complete visible graph |
| `fvbm` | `sparse` | 0 | v-v | Sparse visible graph |
| `rbm` | `dense` | > 0 | v-h | Complete bipartite graph |
| `rbm` | `sparse` | > 0 | v-h | Sparse bipartite graph |
| `sbm` | `dense` | > 0 | v-v, v-h, h-h | Complete general graph |
| `sbm` | `sparse` | > 0 | v-v, v-h, h-h | Sparse general graph |

## Why This Naming?

### Problems with Old Scheme

The old scheme used `architecture: "fully-connected"` and `architecture: "restricted"`, which was confusing:

1. **Overloaded "restricted"**:
   - In ML: "Restricted BM" = bipartite structure (RBM)
   - In old code: "restricted" = sparse connectivity
   - These are DIFFERENT concepts!

2. **Implicit model type**:
   - Model structure (FVBM vs RBM) was inferred from `n_hidden`
   - Not immediately clear from config what you're building

3. **Confusing examples**:
   - "RBM" with `architecture: "fully-connected"` seemed contradictory
   - Actually correct but required explanation

### Advantages of New Scheme

1. **Explicit dimensions**: Model type and connectivity are separate
2. **Standard terminology**: "FVBM" and "RBM" match literature
3. **Self-documenting**: Config clearly states what you're building
4. **No ambiguity**: "sparse" vs "dense" is unambiguous

## Validation Rules

The `create_topology()` function enforces consistency:

```python
# VALID configurations
create_topology(n_visible=10, n_hidden=0, model_type="fvbm", ...)
create_topology(n_visible=10, n_hidden=5, model_type="rbm", ...)
create_topology(n_visible=10, n_hidden=5, model_type="sbm", ...)

# INVALID configurations (raises ValueError)
create_topology(n_visible=10, n_hidden=5, model_type="fvbm", ...)  # FVBM needs n_hidden=0
create_topology(n_visible=10, n_hidden=0, model_type="rbm", ...)   # RBM needs n_hidden>0
create_topology(n_visible=10, n_hidden=0, model_type="sbm", ...)   # SBM needs n_hidden>0
```

## Complete Examples

### Example 1: Dense FVBM
Complete graph with 10 visible nodes (45 edges).

```yaml
true_model:
  n_visible: 10
  n_hidden: 0
  model_type: "fvbm"
  connectivity: "dense"
```

**Result**: All 45 possible visible-visible edges exist.

### Example 2: Sparse FVBM
Sparse graph with ~30% of edges (13-14 edges expected).

```yaml
true_model:
  n_visible: 10
  n_hidden: 0
  model_type: "fvbm"
  connectivity: "sparse"
  connectivity_density: 0.3
```

**Result**: Random subset of ~30% of 45 possible edges.

### Example 3: Dense RBM (Standard)
Complete bipartite graph (6 visible, 3 hidden = 18 edges).

```yaml
true_model:
  n_visible: 6
  n_hidden: 3
  model_type: "rbm"
  connectivity: "dense"
```

**Result**: All 18 visible-to-hidden edges exist. Standard RBM.

### Example 4: Sparse RBM
Sparse bipartite graph with 50% of edges (~100 edges expected).

```yaml
true_model:
  n_visible: 20
  n_hidden: 10
  model_type: "rbm"
  connectivity: "sparse"
  connectivity_density: 0.5
```

**Result**: Random subset of ~50% of 200 possible bipartite edges.

### Example 5: Dense SBM
Complete general BM with all edge types (6 visible, 4 hidden = 51 edges total).

```yaml
true_model:
  n_visible: 6
  n_hidden: 4
  model_type: "sbm"
  connectivity: "dense"
```

**Result**:
- 15 v-v edges (6 choose 2)
- 24 v-h edges (6 × 4)
- 6 h-h edges (4 choose 2)
- Total: 45 edges

### Example 6: Sparse SBM
Sparse general BM with 40% of all possible edges.

```yaml
true_model:
  n_visible: 15
  n_hidden: 8
  model_type: "sbm"
  connectivity: "sparse"
  connectivity_density: 0.4
```

**Result**:
- Possible: 105 v-v + 120 v-h + 28 h-h = 253 total
- Expected: ~40% × 253 ≈ 101 edges

## Code Implementation

See [utils/topology.py](../utils/topology.py) for the implementation:

- `create_topology()` - Main function (lines 14-75)
- `_create_dense_edges()` - Dense edge generation (lines 78-98)
- `_create_sparse_edges()` - Sparse edge generation with connectivity guarantee (lines 101-148)

### Legacy Functions

For backward compatibility during migration, the old functions still exist:

```python
# DEPRECATED - use create_topology() instead
create_fully_connected_topology(n_visible, n_hidden)
create_restricted_topology(n_visible, n_hidden, connectivity, seed)
```

These internally call `create_topology()` with appropriate parameters.

## Standard Boltzmann Machine (SBM) - Phase 2 Implementation

**Status**: ✅ Implemented in Phase 2

The SBM model type is now fully supported:

```yaml
model_type: "sbm"  # Standard/General Boltzmann Machine
```

**Characteristics:**
- Has hidden units (`n_hidden > 0`)
- Allows ALL edge types: v-v, v-h, AND h-h
- Most general BM structure
- Works with both dense and sparse connectivity

**Note**: While topology creation fully supports SBM, ensure your sampler and training configuration can handle the full general BM structure with h-h connections.

## Migration from Old Scheme

If you have old configs using `architecture`:

| Old Config | New Config |
|------------|------------|
| `architecture: "fully-connected"`<br>`n_hidden: 0` | `model_type: "fvbm"`<br>`connectivity: "dense"` |
| `architecture: "fully-connected"`<br>`n_hidden: 3` | `model_type: "rbm"`<br>`connectivity: "dense"` |
| `architecture: "restricted"`<br>`connectivity: 0.3`<br>`n_hidden: 0` | `model_type: "fvbm"`<br>`connectivity: "sparse"`<br>`connectivity_density: 0.3` |
| `architecture: "restricted"`<br>`connectivity: 0.5`<br>`n_hidden: 5` | `model_type: "rbm"`<br>`connectivity: "sparse"`<br>`connectivity_density: 0.5` |

## Summary

**Three independent dimensions:**
1. **Model Type** (`model_type`): FVBM vs RBM vs SBM (structure type)
   - FVBM: Only v-v edges
   - RBM: Only v-h edges (bipartite)
   - SBM: All edge types (v-v, v-h, h-h)
2. **Connectivity** (`connectivity`): Dense vs Sparse (edge density)
3. **Connectivity Density** (`connectivity_density`): Sparsity level (0.0-1.0)

**Key principle**: Each dimension is explicit and self-documenting. No overloaded terminology, no implicit behavior.
