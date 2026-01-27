# Ghost v4 - Tuning Guide

Complete guide to ALL tweakable parameters before training.

---

## Quick Reference: Recommended Configs

### For Speed (Fast Iteration)
```python
DIM = 128          # Smaller
LAYERS = 4         # Fewer
BATCH_SIZE = 32    # Larger
SEQ_LEN = 64       # Shorter
STEPS = 100
```

### For Accuracy (Production)
```python
DIM = 256          # Default
LAYERS = 6         # Default
BATCH_SIZE = 16    # Medium
SEQ_LEN = 256      # Longer
STEPS = 500+
```

### For Resource-Constrained (Phone/Pi)
```python
DIM = 64           # Tiny
LAYERS = 2         # Minimal
BATCH_SIZE = 4     # Small
SEQ_LEN = 32       # Very short
```

---

## Parameter Detailed Guide

### 1. Model Architecture

#### `DIM` - Hidden Dimension
| Value | Params | Speed | Accuracy | Use Case |
|:---|:---|:---|:---|:---|
| 64 | ~500K | Very fast | Lower | Mobile/embedded |
| 128 | ~2M | Fast | Good | Quick experiments |
| **256** | ~6M | Medium | **Best** | **Default** |
| 512 | ~24M | Slow | Overkill | Large datasets only |

**Trade-off:** Bigger = smarter but slower

---

#### `LAYERS` - Number of Mamba Blocks
| Value | Depth | Speed | Capability |
|:---|:---|:---|:---|
| 2 | Shallow | Very fast | Basic patterns |
| 4 | Medium | Fast | Most use cases |
| **6** | Deep | Medium | **Default, complex reasoning** |
| 12 | Very deep | Slow | Multi-step reasoning |

**Trade-off:** More layers = deeper understanding but slower

---

### 2. Training Configuration

#### `BATCH_SIZE` - Samples Per Step
| Value | Memory | Speed/Step | Quality/Step |
|:---|:---|:---|:---|
| 4 | ~1GB | Slow | Noisy |
| 8 | ~2GB | Medium | Better |
| **16** | ~4GB | **Good** | **Default** |
| 32 | ~8GB | Very good | Best |
| 64 | ~16GB | Fastest | Best (if fits) |

**Trade-off:** Bigger = faster training but more memory

**Rule:** Use the largest batch that fits in your RAM

---

#### `SEQ_LEN` - Context Window (Training)
| Value | Memory | Patterns Learned | Use Case |
|:---|:---|:---|:---|
| 32 | Very low | Very short | Character-level |
| 64 | Low | Short phrases | **Testing** |
| 128 | Medium | Sentences | Short Q&A |
| **256** | High | Paragraphs | **Default** |
| 512 | Very high | Documents | Long-form |
| 2048 | Extreme | Full files | Code, articles |

**Trade-off:** Longer = better long-range reasoning but slower

**Note:** Thanks to Mamba, inference has UNLIMITED context regardless of training SEQ_LEN

---

#### `LEARNING_RATE` - Step Size
| Value | Speed | Stability | When to Use |
|:---|:---|:---|:---|
| 1e-5 | Very slow | Very stable | Fine-tuning |
| 1e-4 | Slow | Stable | Conservative |
| **3e-4** | Good | **Default** | **Most cases** |
| 1e-3 | Fast | Less stable | Small models |
| 3e-3 | Very fast | Unstable | Rarely |

**Trade-off:** Higher = faster but risk of divergence

---

#### `STEPS` - Training Iterations
| Value | Time | Quality | Use Case |
|:---|:---|:---|:---|
| 50 | Fast | Quick test | Sanity check |
| 100 | 2 min | Okay | Experiments |
| **300** | 8 min | **Good** | **Default** |
| 500 | 15 min | Better | Small dataset |
| 1000+ | 30+ min | Best | Full training |

**Rule:** Train until loss plateaus (stops improving)

---

### 3. Sparse Routing Parameters

#### `max_depth` - Maximum Layers Per Byte
| Value | Compute | Accuracy |
|:---|:---|:---|
| 2 | Minimal | Lower |
| 4 | Medium | Good |
| **6** | Full | **Best** |

Set this equal to your LAYERS count.

---

#### `threshold` - Surprise Threshold
| Value | Behavior | Speed |
|:---|:---|:---|
| 0.3 | Most bytes get full compute | Slower, safer |
| **0.5** | Balanced | **Default** |
| 0.7 | Skip most bytes | Faster, riskier |

---

### 4. Checkpointing

#### `checkpoint_every` - Save Frequency
| Value | Disk Usage | Safety |
|:---|:---|:---|
| 10 | Very high | Maximum safety |
| 50 | High | Safe |
| **100** | Medium | **Default** |
| 500 | Low | Risky |

**Rule:** More frequent = safer but more disk space

---

### 5. Memory-Speed Trade-offs Summary

| Want... | Increase | Decrease |
|:---|:---|:---|
| Faster training | BATCH_SIZE | SEQ_LEN, LAYERS |
| Better accuracy | DIM, LAYERS, STEPS | LEARNING_RATE |
| Less memory | SEQ_LEN, BATCH_SIZE | DIM |
| Smaller model | DIM, LAYERS | - |
| Longer context | SEQ_LEN | BATCH_SIZE |

---

## Example Configurations

### Config A: Quick Test (5 min)
```python
config = {
    "dim": 128,
    "layers": 4,
    "batch_size": 16,
    "seq_len": 64,
    "learning_rate": 3e-4,
    "steps": 100,
}
```

### Config B: Production (15 min)
```python
config = {
    "dim": 256,
    "layers": 6,
    "batch_size": 16,
    "seq_len": 256,
    "learning_rate": 3e-4,
    "steps": 500,
}
```

### Config C: Maximum Quality (1+ hour)
```python
config = {
    "dim": 512,
    "layers": 8,
    "batch_size": 8,
    "seq_len": 512,
    "learning_rate": 1e-4,
    "steps": 2000,
}
```

---

## GPU/Memory Requirements

| Config | RAM Required | M4 Base | M4 Pro | M4 Max |
|:---|:---|:---|:---|:---|
| Quick Test | 2-4 GB | ✅ | ✅ | ✅ |
| Production | 6-8 GB | ⚠️ | ✅ | ✅ |
| Maximum | 16-24 GB | ❌ | ⚠️ | ✅ |
