# Ghost Model - Complete Feature List

**Version:** v12 (Stable Base)  
**Date:** 2026-01-28  
**Status:** ✅ Verified Working (100% Q&A Accuracy)

---

## 📊 Benchmark Results

| Metric | Value |
|--------|-------|
| Parameters | 11,343,755 |
| Model Size | 2.71 MB (compressed) |
| Training Speed | 9,233 tok/s |
| Inference Speed | 2,568 tok/s |
| Inference Latency | 7.79 ms |
| Q&A Accuracy | 100% (20/20) |

---

## 🧠 Core Architecture Features

### 1. **Ternary Mamba SSM** (v11)
- State Space Model with 2-bit weights (-1, 0, +1)
- Linear-time sequence modeling
- Replaces traditional attention for efficiency

### 2. **Codebook FFN** (v11)
- Feed-Forward Network with learned codebook quantization
- 256 codewords for precision recovery
- SiLU activation with gated projection

### 3. **TernaryLinear** (v11)
- Linear layers quantized to ternary weights
- Learned threshold and scale per channel
- Straight-Through Estimator for gradient flow

### 4. **RMSNorm** (Standard)
- Root Mean Square Layer Normalization
- More stable than LayerNorm for quantized networks

---

## 🔧 Efficiency Features

### 5. **Conv1D Tokenizer** (v7)
- Byte-level embedding + depthwise Conv1D
- Captures local context patterns
- Kernel size: 4, padding: 3

### 6. **Mixture of Depths (MoD)** (v9)
- Learned router skips "easy" tokens
- 50% compute reduction on average
- Capacity factor: 0.5

### 7. **Sparse Attention** (v7)
- Stride-based sparse attention at layers 3, 5
- Long-range context without O(n²) cost
- Stride: 64 tokens

---

## 🧬 Learning Features

### 8. **Depth Predictor** (v7)
- Predicts processing depth per token
- Enables early exit for simple tokens

### 9. **Byte Importance** (v7)
- Per-byte importance embedding
- Guides attention to important tokens

### 10. **Surprise Predictor** (v7)
- Predicts next-token distribution
- High surprise = important token

---

## 💾 Memory Features

### 11. **Cognitive Memory Bank** (v9)
- Key-Value store for one-shot learning
- Max entries: 512
- Associative recall with cosine similarity

### 12. **Per-Layer Memory Query** (v7)
- Memory queries at layers 2, 4
- Gated memory injection

### 13. **Sleep Learning** (v11)
- Contrastive consolidation during idle
- Strengthens memory associations

### 14. **Curiosity Signal** (v11)
- High prediction entropy triggers memory write
- "I don't know this, remember it"

---

## 🏋️ Training Features

### 15. **SwarmMomentum** (v10)
- Multi-worker consensus training
- 4 workers compare gradients
- ⚠️ Currently unstable (NaN bug) - Use simple AdamW instead

### 16. **Adaptive Codebook** (v11)
- Per-layer precision adjustment
- Scales based on gradient magnitude

---

## 📁 File Structure (v12)

```
ghost_model_v12/
├── core/
│   ├── ghost_worker_v12.py    # Main model
│   ├── ternary_linear.py      # Ternary layers
│   ├── ternary_mamba.py       # Ternary SSM
│   ├── learned_codebook.py    # Codebook quantization
│   ├── mixture_of_depths.py   # MoD router
│   └── cognitive_memory.py    # Memory system
├── training/
│   ├── train_qna_debug.py     # Simple training (WORKING)
│   ├── train_perceptual.py    # Swarm training (BUGGY)
│   ├── benchmark.py           # Benchmark script
│   ├── dataset_qna.py         # Q&A dataset
│   └── muon.py                # Muon optimizer
├── tests/
├── docs/
├── checkpoints/
└── README.md
```

---

## ✅ What Works

1. Model initialization and forward pass
2. Ternary + Codebook compression
3. MoD routing
4. Sparse attention
5. Cognitive memory (when importance > 0.7)
6. Simple AdamW training
7. Q&A generation (100% accuracy)

## ⚠️ Known Issues

1. **SwarmMomentum NaN** - Consensus calculation causes NaN at ~350 steps
2. **Memory threshold** - Needs trained weights to store (importance check)

---

## 🔜 Next Steps for v13

1. Fix SwarmMomentum bug
2. Add LoRA for efficient fine-tuning
3. Create specialized workers (Tinker, Coder, Writer)
4. Add checkpoint resume
