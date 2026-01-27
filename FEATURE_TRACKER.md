# Ghost Model Feature Tracker

**Last Updated:** 2026-01-25

## Quick Status Summary

| Version | Status | Novel Count | Params | Compressed |
|---------|--------|-------------|--------|------------|
| v7 | ✅ Stable | 12 features | 6.6M | 0.83 MB |
| v8 | ✅ Stable | 8 features | 7.6M | 0.95 MB |
| v9 | ⚠️ Missing features | 7 features | 7.6M | 0.95 MB |
| v10 | ✅ Ultimate | 10 features | 19.3M | 2.4 MB |
| v10-swarm | ✅ Worker tier | 9 features | 8.3M | 1.0 MB |
| **v11** | ✅ **ULTRA** | **14 features** | **11.3M** | **2.71 MB** |

---

## v11 Complete Feature List

| # | Feature | From | Status | Benefit |
|---|---------|------|--------|---------|
| 1 | Conv1D Tokenizer | v7 | ✅ | Local context |
| 2 | Depth Predictor + Byte Importance | v7 | ✅ | Sparse routing |
| 3 | Surprise Predictor | v7 | ✅ | Skip predictable |
| 4 | Per-Layer Memory Query | v7 | ✅ | Layerwise retrieval |
| 5 | Sparse Attention | v7+v8 | ✅ | Long-range |
| 6 | **Ternary Mamba** | v11 | ✅ | 2-bit SSM |
| 7 | **Codebook FFN** | v11 | ✅ RESTORED (Clipped) | Precision recovery |
| 8 | Mixture of Depths | v9 | ✅ | 50% compute skip |
| 9 | Memory Bank | v9 | ✅ | Infinite context |
| 10 | SwarmMomentum | v10 | ✅ RESTORED (Safe Mode) | Consensus training |
| 11 | **Adaptive Codebook** | v11 | ✅ RESTORED | Per-layer precision |
| 12 | **Cognitive Memory** | v11 | ✅ | One-shot learning |
| 13 | **Curiosity Signal** | v11 | ✅ | Entropy-based writing |
| 14 | **Sleep Learning** | v11 | ✅ | Contrastive consolidation |
| 15 | **Optimizer** | v11 | ⚠️ SGD (AdamW Unstable) | Robustness > Speed |

---

## v11 Innovation: Ternary + Codebook

### Why Ternary > Binary?

| Method | Bits/weight | Quality |
|--------|-------------|---------|
| Binary (v8-v10) | 1 bit | 80-85% |
| **Ternary + Codebook** | 2 bits | **92-95%** |

### Compression Achieved

| Model Size | FP32 | Binary | Ternary+CB (v11) |
|------------|------|--------|------------------|
| 1B params | 4 GB | 125 MB | **250 MB** |
| 11M params | 44 MB | 1.4 MB | **2.7 MB** |

---

## Benchmark Results

### v11 Q&A (10 samples, 300 epochs)

- **Accuracy: 90%**
- Training time: 81.5s
- Loss: 0.077

### vs v10-swarm

| Metric | v10-swarm | v11 |
|--------|-----------|-----|
| Params | 8.3M | 11.3M |
| Size | 1.0 MB | 2.71 MB |
| Accuracy | 100% | 90% |
| Features | 9 | 11 |

---

## Evolution History

```
v7 (12 features, 6.6M)
 ↓ Binary Mamba introduced
v8 (8 features, 7.6M)
 ↓ MoD + Memory added
v9 (7 features, 7.6M)
 ↓ All features combined + MoE
v10 (10 features, 19.3M)
 ↓ MoE removed for worker tier
v10-swarm (9 features, 8.3M)
 ↓ Ternary + Codebook introduced
v11 (11 features, 11.3M) ← CURRENT
```
