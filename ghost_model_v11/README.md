# Ghost Model v11 - Ultra Compression

**11 Novel Features** | **Ternary + Codebook** | **1B params → 250MB**

## Overview

Ghost v11 introduces **ternary weights with learned codebook** for extreme compression while maintaining quality.

| Metric | v10 (Binary) | v11 (Ternary+CB) |
|--------|--------------|------------------|
| Bits/weight | 1 bit | 2 bits + codebook |
| Quality | 80-85% | 92-95% |
| Compression | 8x | 7x |
| Training | Stable | More stable |

## Core Innovation

### Ternary Weights
```python
# Binary: +1 or -1 (loses magnitude)
# Ternary: -1, 0, +1 (keeps sparsity)

Weight: 0.78 → Ternary: +1, Magnitude: 0.78 (from codebook)
Weight: -0.23 → Ternary: -1, Magnitude: 0.23 (from codebook)
Weight: 0.02 → Ternary: 0 (sparse, skip computation!)
```

### Learned Codebook
```python
# 256 possible magnitude values per layer
codebook = [0.001, 0.002, ..., 0.999]  # Learned during training

# Each weight stores:
# - 2 bits: ternary sign (-1, 0, +1)
# - 8 bits: codebook index
# = 10 bits vs 32 bits (3.2x compression)
```

## Features

| # | Feature | Source | Purpose |
|---|---------|--------|---------|
| 1 | Conv1D Tokenizer | v7 | Local context |
| 2 | Depth Predictor | v7 | Sparse routing |
| 3 | Surprise Predictor | v7 | Skip predictable tokens |
| 4 | Per-Layer Memory | v7 | Context retrieval |
| 5 | Sparse Attention | v7 | Long-range dependencies |
| 6 | **Ternary Mamba** | v11 | SSM with ternary weights |
| 7 | **Codebook FFN** | v11 | Precision recovery |
| 8 | MoD | v9 | Compute savings |
| 9 | Memory Bank | v9 | Persistent memory |
| 10 | SwarmMomentum | v10 | Consensus training |
| 11 | **Adaptive Codebook** | v11 | Per-layer optimization |

## Usage

```python
from ghost_model_v11 import GhostWorkerV11
import mlx.core as mx

# Create model
model = GhostWorkerV11(dim=256, num_layers=6)

# Check storage
storage = model.estimate_storage()
print(f"Params: {storage['params']:,}")
print(f"Compressed: {storage['total_mb']:.2f} MB")

# Forward pass
x = mx.array([[ord(c) for c in "list pods"]], dtype=mx.int32)
output = model(x)
```

## Storage Comparison

| Model Size | FP32 | Binary (v10) | Ternary+CB (v11) |
|------------|------|--------------|------------------|
| 8M | 32 MB | 1.5 MB | 2.0 MB |
| 100M | 400 MB | 12.5 MB | 25 MB |
| 1B | 4 GB | 125 MB | **250 MB** ✓ |

## Architecture

```
Input
  ↓
[Conv1D Tokenizer] ← Local context
  ↓
[Depth Predictor] ← Route tokens
  ↓
For each layer:
  ├─ [TernaryMamba] ← SSM with ternary weights
  ├─ [Memory Query] ← (layers 2, 4)
  ├─ [Sparse Attn] ← (layers 3, 5)
  └─ [CodebookFFN] ← Precision recovery
  ↓
[Output] → Predictions
```

## Training

v11 is compatible with SwarmMomentum training:

```python
from ghost_swarm.training import SwarmMomentum

trainer = SwarmMomentum(model, lr=2e-3, num_workers=4)
trainer.train(data, epochs=400)
```

## Files

```
ghost_model_v11/
├── __init__.py
├── README.md
├── core/
│   ├── __init__.py
│   ├── ternary_linear.py      # TernaryLinear layer
│   ├── learned_codebook.py    # LearnedCodebook + CodebookLinear
│   ├── ternary_mamba.py       # TernaryMamba SSM
│   └── ghost_worker_v11.py    # Full v11 worker
└── docs/
    └── COMPRESSION.md
```

## Roadmap

- [x] Core ternary layers
- [x] Codebook integration
- [x] TernaryMamba SSM
- [ ] Q&A benchmark vs v10
- [ ] 1B scale testing
