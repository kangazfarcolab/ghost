# Ghost Model Changelog

Complete history of changes from v1 to v5.

---

## v5 (Current) - The Next Evolution ✅ COMPLETE

### Results Achieved
- **v5 Fast**: 10/10 (100%) on basic + hard tasks
- **Training Time**: 771s
- **Parameters**: 6.58M
- **Self-Distillation**: 10→70 Q&A pairs, 100% base accuracy

### New Features
- **Sparse Attention**: Global context via efficient attention at layers 3, 5
- **Per-Layer Memory Query**: Each layer can query memory (layers 2, 4)
- **Hybrid Architecture**: Mamba (local) + Sparse Attention (global) + Memory

### Architecture Changes
```
v4: Mamba → Mamba → Mamba → Mamba → Mamba → Mamba → Memory → Output
v5: Mamba → Mamba+Mem → Mamba+Attn → Mamba+Mem → Mamba → Mamba+Attn → Output
```

### Why These Changes?
1. **Sparse Attention**: Mamba is great for local patterns but limited for global reasoning. Adding sparse attention every 2 layers gives global context at minimal cost.

2. **Per-Layer Memory**: In v4, memory was only queried at the end. Different layers may need different information - early layers for surface patterns, late layers for semantic matching.

---

## v4 - Unified Model + Memory Fix

### Features Added
- Combined all v3 experiments into one model
- **Fixed Memory Augmentation**: Actually queries memory in forward pass
- Added cross-attention to memory with gating
- Checkpointing support

### Key Insight
The memory in v3 holographic experiments never worked because:
1. Memory was stored but never queried in forward pass
2. Circular convolution was oversimplified (x*y doesn't work)

**Fix**: Use soft attention over stored facts, let model learn when to query via gating.

### Results
- 100% accuracy
- 5.98M parameters
- 479s training time

---

## v3 - Revolutionary Experiments

### v3.1: State-Space Tokenization (SST) ✅
- **Problem**: ByteGrouper with GROUP_SIZE=4 caused character confusion
- **Solution**: Let Mamba hidden state detect natural word boundaries
- **Key Insight**: State velocity (|state_t - state_{t-1}|) is high at word boundaries
- **Result**: 40% → 100% accuracy

### v3.2: Sparse Byte Routing ✅
- **Problem**: All bytes got same compute regardless of importance
- **Solution**: Route important bytes through more layers
- **Key Insight**: Punctuation/numbers need more layers than common letters
- **Result**: 2.3x faster, 5x fewer params

### v3.3: Predictive Coding ✅
- **Problem**: Wasting compute on predictable bytes
- **Solution**: Quick prediction first, skip if confident
- **Key Insight**: "The " is predictable, save compute for hard bytes
- **Result**: 3.4x faster

### v3.4: Holographic Memory ❌
- **Problem**: Wanted to store facts without training
- **Attempt**: Hyperdimensional computing (circular convolution)
- **Result**: 0% - memory never queried, operations oversimplified
- **Status**: Fixed in v4 with attention-based memory

### v3.5: Recursive Self-Compression ✅
- **Problem**: Model files too large
- **Solution**: Train model to predict its own weights, store residuals
- **Result**: 7.5x smaller files

---

## v2 - Extended Context

### Changes
- Extended context from 512 to 2048 tokens
- Improved training data formatting

### Results
- 60% accuracy (up from 40%)
- Same 29M parameters

---

## v1 - Foundation

### Core Architecture
- **Ghost Weights**: 2-bit palette quantization with learnable codebook
- **Mamba SSM**: O(N) state-space model for infinite context
- **Parallel Scan**: O(log N) efficient Mamba computation
- **MoE**: 8 experts, top-2 routing

### Novel Contributions
1. Ghost Weights (2-bit palette with STE)
2. Byte-level processing without BPE

### Results
- 40% accuracy on 10-question benchmark
- 29M parameters

---

## Feature Evolution Matrix

| Feature | v1 | v2 | v3 | v4 | v5 |
|:---|:---:|:---:|:---:|:---:|:---:|
| Ghost Weights (2-bit) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Mamba SSM | ✅ | ✅ | ✅ | ✅ | ✅ |
| Parallel Scan | ✅ | ✅ | ✅ | ✅ | ✅ |
| MoE | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ |
| ByteGrouper | ✅ | ✅ | ❌ | ❌ | ❌ |
| State-Space Tokenization | - | - | ✅ | ✅ | ✅ |
| Sparse Byte Routing | - | - | ✅ | ✅ | ✅ |
| Predictive Coding | - | - | ✅ | ✅ | ✅ |
| Memory Augmentation | - | - | ❌ | ✅ | ✅ |
| Sparse Attention | - | - | - | - | ✅ |
| Per-Layer Memory | - | - | - | - | ✅ |

---

## Accuracy Progress

```
v1:  ████░░░░░░ 40%
v2:  ██████░░░░ 60%
v3:  ██████████ 100%
v4:  ██████████ 100%
v5:  ██████████ 100% (target)
```

---

## Parameter Efficiency Progress

```
v1:  29.0M  ████████████████████████████████████████
v2:  29.0M  ████████████████████████████████████████
v3:  5.9M   ████████
v4:  5.98M  ████████
v5:  ~7M    ██████████ (estimate with sparse attention)
```

---

## Training Speed Progress

```
v1:  ~600s   ████████████
v2:  ~800s   ████████████████
v3:  340s    ███████
v4:  479s    ██████████
v5:  ~500s   ██████████ (target)
```
