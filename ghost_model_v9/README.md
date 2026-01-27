# Ghost Model v9

**MoD + Memory + Binary Mamba**

## Features

| Feature | Benefit |
|---------|---------|
| **Binary Mamba** | 20x compression |
| **Mixture of Depths** | 50% less compute |
| **Memory Mamba** | Infinite context |

## Usage

```python
from ghost_model_v9 import GhostWorkerV9

model = GhostWorkerV9(dim=256, num_layers=6)

# With all features
out = model(x)

# Check compute savings
stats = model.get_mod_stats()
print(f"Compute: {stats['compute_ratio']*100:.1f}%")
```

## Architecture

```
Input → Embedding
          ↓
    Memory Retrieval
          ↓
┌─────────────────────┐
│ MoD Router          │ → Easy tokens skip layers
└─────────────────────┘
          ↓
┌─────────────────────┐
│ Binary Mamba        │ ← 1-bit weights
│ + BitFFN            │
└─────────────────────┘
          ↓
    Memory Store
          ↓
       Output
```

## Comparison

| Metric | v7 | v8 | v9 |
|--------|----|----|-----|
| Memory | 25MB | 1.5MB | **1.5MB** |
| Compute | 100% | 100% | **~50%** |
| Context | Fixed | Fixed | **Infinite** |
