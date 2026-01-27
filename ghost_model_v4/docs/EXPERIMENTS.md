# Ghost v4 - Complete Experiments Documentation

This document catalogs ALL experiments conducted from v1 to v4.

---

## Experiment Timeline

| Version | Date | Key Experiment | Result |
|:---|:---|:---|:---|
| v1 | Initial | ByteGrouper (GROUP_SIZE=4) | 40% accuracy |
| v2 | +1 day | Longer context (2048) | 60% accuracy |
| v3.1 | +1 day | State-Space Tokenization | **100% accuracy** |
| v3.2 | +1 day | Sparse Byte Routing | 100% in 500s |
| v3.3 | +1 day | Predictive Coding | 100% in 340s |
| v3.4 | +1 day | Holographic Memory | 0% (needs work) |
| v3.5 | +1 day | Recursive Compression | 7.5x smaller |
| **v4** | +1 day | All combined | **100% in 479s, 6M params** |

---

## Validated Experiments (Used in v4)

### 1. State-Space Tokenization ✅

**Problem Solved:** ByteGrouper with GROUP_SIZE=4 caused character confusion ("Paris" → "Pythoth")

**Solution:** Let Mamba hidden state detect natural word boundaries

```python
class StateSpaceTokenizer:
    # Detects boundaries from state "velocity"
    state_velocity = abs(state_t - state_{t-1})
    boundaries = sigmoid(velocity_score)
```

**Result:** 40% → **100% accuracy**

---

### 2. Sparse Byte Routing ✅

**Problem Solved:** All bytes got same compute regardless of importance

**Solution:** Route important bytes through more layers, skip layers for easy bytes

```python
class DepthRouter:
    # Punctuation, numbers: 6 layers
    # Common letters: 2-3 layers
    depth = sigmoid(context_score + byte_importance) * max_layers
```

**Result:** 2.3x faster, 5x fewer params

---

### 3. Predictive Coding ✅

**Problem Solved:** Wasting compute on predictable bytes ("The " is obvious)

**Solution:** Quick prediction first, only full compute if wrong

```python
class SurpriseDetector:
    # Quick prediction
    pred = quick_model(bytes)
    # Only process if surprising
    surprise_mask = (prediction_error > threshold)
```

**Result:** 3.4x faster than baseline

---

### 4. Checkpointing ✅

**Problem Solved:** Can't pause training when closing laptop

**Solution:** Save model state periodically, resume later

```python
# Save
trainer.save_checkpoint("checkpoint_step100.npz")

# Resume next day
trainer.train(data, resume_from="checkpoint_step100.npz")
```

**Result:** Practical training for real-world use

---

### 5. Recursive Self-Compression ✅

**Problem Solved:** Model files too large (515 KB)

**Solution:** Model predicts its own weights, store only residuals

```python
# Train predictor to predict weight values
predictor = train_on(weight_positions → weight_values)
# Store only residuals
residuals = actual_weights - predicted_weights
```

**Result:** 7.5x smaller files

---

## Experimental Features (Needs Work)

### Holographic Memory 🧪

**Concept:** Store facts in hyperdimensional vectors, query by similarity

**Status:** Memory storage works, but model doesn't learn to query it

**Next Steps:**
- Add cross-attention between hidden states and memory
- Train model to learn when/how to query
- Make retrieval differentiable

---

## Deprecated Features

### ByteGrouper (v1-v2)

**Replaced by:** State-Space Tokenization

**Reason:** Fixed grouping caused character confusion

---

## Feature Comparison Table

| Feature | Novel? | Status | Impact |
|:---|:---|:---|:---|
| Ghost Weights (2-bit) | 🆕 Yes | ✅ Used | 8x smaller |
| Mamba SSM | Known | ✅ Used | O(N) context |
| Parallel Scan | Known | ✅ Used | O(log N) |
| MoE | Known | ⚠️ Partial | 4x capacity |
| State-Space Tokenization | 🆕 Yes | ✅ Used | +60% accuracy |
| Sparse Byte Routing | 🆕 Yes | ✅ Used | 2.3x faster |
| Predictive Coding | Semi-novel | ✅ Used | 3.4x faster |
| Recursive Compression | 🆕 Yes | ✅ Works | 7.5x smaller |
| Holographic Memory | 🆕 Yes | 🧪 Experimental | - |

---

## Benchmark Results

### 10-Question Accuracy Test

| Model | Accuracy | Time | Params |
|:---|:---|:---|:---|
| v1 (ByteGrouper) | 40% | - | 29M |
| v2 (Long Context) | 60% | 806s | 29M |
| v3 SST Only | 100% | 1169s | 28.8M |
| v3 + Sparse | 100% | 500s | 5.9M |
| v3 + Predictive | 100% | 340s | 4.1M |
| **v4 Combined** | **100%** | **479s** | **5.98M** |

Note: v4 is slightly slower than Predictive alone because it combines all features for robustness.
