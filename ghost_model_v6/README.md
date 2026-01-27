# Ghost Model v6

> Production-Ready Byte-Level Language Model

The culmination of all Ghost Model experiments, ready for real-world training.

---

## Quick Start

### 1. Environment Setup

We use the shared virtual environment in `learnable_tok/venv` because:
- All dependencies (MLX, numpy, etc.) are already installed
- Avoids duplicate package installations
- Consistent environment across all Ghost versions

```bash
# Activate the shared environment
source /Users/azfar.naufal/Documents/myprodjet/ex/learnable_tok/venv/bin/activate

# Verify MLX is available
python -c "import mlx.core as mx; print('MLX version:', mx.__version__)"
```

### 2. Test the Model

```bash
cd /Users/azfar.naufal/Documents/myprodjet/ex/ghost_model_v6/tests

# Run basic test
python test_basic.py
```

### 3. Train from Scratch

```bash
cd /Users/azfar.naufal/Documents/myprodjet/ex/ghost_model_v6/training

# Train on Q&A data
python train_v6.py
```

---

## Step-by-Step Training Guide

### Step 1: Activate Environment
```bash
source /Users/azfar.naufal/Documents/myprodjet/ex/learnable_tok/venv/bin/activate
```

### Step 2: Prepare Your Data
Create a text file with your training data. For Q&A:
```
Q: What is 2+2? A: 4
Q: Capital of France? A: Paris
...
```

### Step 3: Initialize Model
```python
from ghost_v6 import GhostV6
import mlx.core as mx

model = GhostV6(dim=256, num_layers=6)
mx.eval(model.parameters())
print(f"Parameters: {model.count_params():,}")
```

### Step 4: Store Facts (Optional Memory)
```python
# Store facts for memory augmentation
model.store_fact([ord(c) for c in "capital of france"], [ord(c) for c in "Paris"])
```

### Step 5: Train
```python
import mlx.optimizers as optim

optimizer = optim.AdamW(learning_rate=3e-4)

for step in range(300):
    # Your training loop here
    loss, grads = mx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
```

### Step 6: Generate
```python
def generate(model, prompt, max_tokens=20):
    x = mx.array([[ord(c) for c in prompt]], dtype=mx.int32)
    
    for _ in range(max_tokens):
        logits = model(x)
        probs = mx.softmax(logits[0, -1, :] / 0.3)
        val = int(mx.argmax(probs).item())
        if val == 10: break  # newline
        x = mx.concatenate([x, mx.array([[val]])], axis=1)
    
    return ''.join(chr(int(b)) for b in x[0].tolist()[len(prompt):])
```

---

## Complete Feature List (v1 → v6)

### v1: Foundation
| Feature | Status | Description |
|:---|:---|:---|
| Ghost Weights (2-bit) | ✅ Novel | 4-value learnable codebook |
| Mamba SSM | ✅ | O(N) state-space model |
| Parallel Scan | ✅ | O(log N) computation |
| MoE (8 experts) | ⚠️ Deprecated | Removed in later versions |

### v2: Extended Context
| Feature | Status | Description |
|:---|:---|:---|
| 2048 context | ✅ | Longer sequences |
| Byte-level | ✅ | No BPE tokenization |

### v3: Breakthroughs
| Feature | Status | Description |
|:---|:---|:---|
| State-Space Tokenization | ✅ Novel | Word boundaries from state velocity |
| Sparse Byte Routing | ✅ Novel | Adaptive depth per byte |
| Predictive Coding | ✅ Novel | Skip predictable bytes |
| Recursive Compression | ✅ Novel | 7.5x smaller checkpoints |

### v4: Memory
| Feature | Status | Description |
|:---|:---|:---|
| Memory Augmentation | ✅ Fixed | Cross-attention to stored facts |
| Checkpointing | ✅ | Pause/resume training |

### v5: Hybrid Architecture
| Feature | Status | Description |
|:---|:---|:---|
| Sparse Attention | ✅ | Global context (layers 3, 5) |
| Per-Layer Memory | ✅ | Query at layers 2, 4 |
| Self-Distillation | ✅ | 38x data expansion |

### v6: Production Ready
| Feature | Status | Description |
|:---|:---|:---|
| All v5 features | ✅ | Combined and tested |
| Documentation | ✅ | Complete setup guide |
| Ready for training | ✅ | Bash, code, custom domains |

---

## Architecture

```
Input: Raw Bytes [0-255]
       ↓
┌──────────────────────────────────────────┐
│ State-Space Tokenizer                    │
│ • Conv1d for local context               │
│ • Boundary detection from state velocity │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│ Routing Layer                            │
│ • Depth predictor (per-byte)             │
│ • Surprise detector (skip predictable)   │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│ Layer 0: Mamba + FFN                     │
│ Layer 1: Mamba + FFN                     │
│ Layer 2: Mamba + Memory + FFN            │
│ Layer 3: Mamba + Sparse Attention + FFN  │
│ Layer 4: Mamba + Memory + FFN            │
│ Layer 5: Mamba + Sparse Attention + FFN  │
└──────────────────────────────────────────┘
       ↓
Output: Next byte logits [256]
```

---

## Model Configuration

| Parameter | Default | Description |
|:---|:---|:---|
| `dim` | 256 | Hidden dimension |
| `num_layers` | 6 | Number of layers |
| `stride` | 128 | Sparse attention stride |
| Memory layers | 2, 4 | Where to query memory |
| Attention layers | 3, 5 | Where to use sparse attention |

---

## Files

```
ghost_model_v6/
├── README.md           # This file
├── core/
│   └── ghost_v6.py     # Main model (6.58M params)
├── training/
│   └── train_v6.py     # Training script
├── tests/
│   └── test_basic.py   # Basic benchmark
├── docs/
│   └── FEATURES.md     # Detailed feature docs
└── checkpoints/        # Saved models
```

---

## Performance

| Benchmark | Result |
|:---|:---|
| Basic Q&A (10 questions) | 100% |
| Hard Tasks (multi-hop) | 100% |
| Self-Distillation (38x) | 100% base, 58% novel |
| Parameters | 6.58M |
| Training Time (300 steps) | ~325s |

---

## Why Use learnable_tok/venv?

The virtual environment is located at:
```
/Users/azfar.naufal/Documents/myprodjet/ex/learnable_tok/venv/
```

**Reasons:**
1. **Historical**: Original ML experiments started in learnable_tok
2. **Dependencies**: MLX, numpy, and all required packages installed
3. **Consistency**: Same environment works for all Ghost versions
4. **Shared**: Avoids duplicate 500MB+ environments

**If you need a fresh environment:**
```bash
cd /Users/azfar.naufal/Documents/myprodjet/ex/ghost_model_v6
python3 -m venv venv
source venv/bin/activate
pip install mlx numpy
```
