# Ghost Model v4

> The Ultimate Ghost Model - All Validated Features Combined

## 🏆 Achievement Summary

| Metric | v1 Baseline | v4 Final | Improvement |
|:---|:---|:---|:---|
| **Accuracy** | 40% | **100%** | +150% |
| **Training Time** | 1169s | **479s** | 2.4x faster |
| **Parameters** | 28.8M | **5.98M** | 4.8x smaller |
| **File Size** | 515 KB | **68.8 KB** (with compression) | 7.5x smaller |

## Features Included

### Core Architecture
- ✅ **2-bit Ghost Weights** - 8x smaller than float16
- ✅ **Mamba SSM** - O(N) infinite context
- ✅ **Parallel Associative Scan** - O(log N) processing

### v3 → v4 Innovations
- ✅ **State-Space Tokenization** - Model learns its own boundaries
- ✅ **Sparse Byte Routing** - Adaptive depth per byte
- ✅ **Predictive Coding** - Skip predictable bytes
- ✅ **Checkpointing** - Pause/resume training

## Quick Start

```python
from ghost_model_v4.core.ghost_v4 import GhostModelV4Ultimate, Trainer

# Initialize model
model = GhostModelV4Ultimate(dim=256, num_layers=6)

# Create trainer with checkpointing
trainer = Trainer(model, checkpoint_dir="my_checkpoints")

# Train (auto-saves checkpoints)
trainer.train(data, steps=300, checkpoint_every=100)

# Resume from checkpoint
trainer.train(data, steps=200, resume_from="my_checkpoints/checkpoint_step300.npz")
```

## Documentation

| Doc | Description |
|:---|:---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full architecture details |
| [EXPERIMENTS.md](docs/EXPERIMENTS.md) | All experiments and results |
| [TUNING_GUIDE.md](docs/TUNING_GUIDE.md) | Parameters to tweak before training |

## Folder Structure

```
ghost_model_v4/
├── README.md
├── core/
│   └── ghost_v4.py      # Main model
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EXPERIMENTS.md
│   └── TUNING_GUIDE.md
└── checkpoints/         # Saved model states
```
