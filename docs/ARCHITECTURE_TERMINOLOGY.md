# Architecture Terminology Guide

## Overview

This document clarifies the potentially confusing terminology around "restricted" in the context of Boltzmann Machines.

## The Confusion

The term "restricted" appears in two different contexts with different meanings:

1. **In ML Literature**: "Restricted Boltzmann Machine (RBM)" refers to a specific architecture
2. **In This Codebase**: `architecture: "restricted"` refers to sparse connectivity

## Architecture Parameter Behavior

### `architecture: "fully-connected"`

Creates **all possible edges** within the specified structure:

- **When `n_hidden = 0`**:
  - Creates a **dense visible graph**
  - All visible nodes connect to each other
  - Total edges: n_visible × (n_visible - 1) / 2

- **When `n_hidden > 0`**:
  - Creates a **complete bipartite graph** (standard RBM)
  - All visible nodes connect to all hidden nodes
  - No connections within visible layer or within hidden layer
  - Total edges: n_visible × n_hidden
  - **This IS a "Restricted Boltzmann Machine" in ML terminology**

### `architecture: "restricted"`

Creates **a random subset of edges** (sparse connectivity):

- **When `n_hidden = 0`**:
  - Creates a **sparse visible graph**
  - Only `connectivity` fraction of possible visible-visible edges are included
  - Example: `connectivity: 0.3` means 30% of possible edges

- **When `n_hidden > 0`**:
  - Creates a **sparse bipartite graph**
  - Only `connectivity` fraction of possible visible-hidden edges are included
  - Still maintains bipartite structure (no intra-layer connections)

## Terminology Mapping

| You Want... | Configuration | ML Term |
|-------------|---------------|---------|
| Dense graph (all visible nodes connected) | `n_hidden: 0`<br>`architecture: "fully-connected"` | Fully Visible Boltzmann Machine |
| Standard RBM (complete bipartite) | `n_hidden: > 0`<br>`architecture: "fully-connected"` | Restricted Boltzmann Machine (RBM) |
| Sparse visible graph | `n_hidden: 0`<br>`architecture: "restricted"`<br>`connectivity: 0.3` | Sparse Boltzmann Machine |
| Sparse bipartite graph | `n_hidden: > 0`<br>`architecture: "restricted"`<br>`connectivity: 0.3` | Sparse RBM |

## Why "fully-connected" for RBMs?

This seems counterintuitive at first: why is a "Restricted" Boltzmann Machine created with `architecture: "fully-connected"`?

The answer is that **RBM refers to the restricted structure (bipartite), not restricted connectivity**:
- "Restricted" in RBM means restrictions on **which types** of connections exist (only between layers)
- "fully-connected" in our parameter means **all allowed connections** exist (all bipartite edges)

So `architecture: "fully-connected"` with `n_hidden > 0` creates a complete bipartite graph, which is exactly what a standard RBM is.

## Code Implementation

See [utils/topology.py](../utils/topology.py) for the implementation:

- `create_fully_connected_topology()`: Lines 9-45
  - Creates dense visible graph when n_hidden=0 (lines 34-38)
  - Creates complete bipartite graph when n_hidden>0 (lines 40-43)

- `create_restricted_topology()`: Lines 48-109
  - Creates sparse visible graph when n_hidden=0 (lines 76-81)
  - Creates sparse bipartite graph when n_hidden>0 (lines 83-87)
  - Ensures connectivity (lines 89-108)

## Recommendations

To avoid confusion in the future, consider:

1. **Using more descriptive parameter names**:
   - `architecture: "dense"` instead of `"fully-connected"`
   - `architecture: "sparse"` instead of `"restricted"`

2. **Explicitly documenting**:
   - When `n_hidden > 0` with any architecture, you get a bipartite structure
   - The architecture parameter only controls edge density, not structure type

3. **Always clarifying in examples**:
   - Explicitly state whether you're creating an RBM or not
   - Mention edge density (complete vs sparse)

## Examples with Clear Descriptions

### Example 1: Dense Fully Visible BM
```yaml
true_model:
  n_visible: 10
  n_hidden: 0              # No hidden units
  architecture: "fully-connected"  # All visible-visible edges
# Result: Dense graph with 45 edges
```

### Example 2: Standard RBM (Complete Bipartite)
```yaml
true_model:
  n_visible: 6
  n_hidden: 3              # Hidden units present
  architecture: "fully-connected"  # All bipartite edges
# Result: RBM with 18 edges (6×3)
```

### Example 3: Sparse Visible BM
```yaml
true_model:
  n_visible: 20
  n_hidden: 0              # No hidden units
  architecture: "restricted"  # Sparse connectivity
  connectivity: 0.3        # 30% of possible edges
# Result: Sparse graph with ~57 edges (out of 190 possible)
```

### Example 4: Sparse RBM
```yaml
true_model:
  n_visible: 20
  n_hidden: 10             # Hidden units present
  architecture: "restricted"  # Sparse connectivity
  connectivity: 0.3        # 30% of possible bipartite edges
# Result: Sparse bipartite graph with ~60 edges (out of 200 possible)
```

## Summary

- **RBM in ML = bipartite structure** (any model with `n_hidden > 0`)
- **`architecture: "restricted"` = sparse connectivity** (random edge subset)
- **`architecture: "fully-connected"` = dense connectivity** (all allowed edges)
- The examples in README.md are technically correct but require explanation
- This terminology is inherently confusing and worth documenting clearly
