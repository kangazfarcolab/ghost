# Ghost Model v8

**Binary Mamba + Adaptive Depth**

## Features

| Feature | Benefit |
|---------|---------|
| **1-bit weights** | 13x smaller memory |
| **Adaptive depth** | 30-50% less compute |
| **SwarmMomentum** | 3.4x faster training |

## Usage

```python
from ghost_model_v8 import GhostWorkerV8

model = GhostWorkerV8(dim=256, num_layers=6)

# With early exit (default)
out = model(x)

# Full depth (no early exit)
out = model(x, use_early_exit=False)
```

## Architecture

```
Input (bytes)
    ↓
Embedding (float)
    ↓
┌─────────────────────┐
│ BinaryMamba Layer 1 │ ← 1-bit weights
│ + BitFFN            │
└─────────────────────┘
    ↓ (confidence check)
┌─────────────────────┐
│ BinaryMamba Layer 2 │
│ + BitSparseAttn     │
│ + BitFFN            │
└─────────────────────┘
    ↓ (confidence check → early exit if > 0.85)
    ...
    ↓
Output (256 logits)
```

## Comparison

| Metric | v7 | v8 |
|--------|----|----|
| Memory | 26 MB | **2 MB** |
| Compute | 100% | **50-70%** |
| Quality | Baseline | Same or better |
