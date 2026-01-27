# Ghost v13: Enhanced Swarm Intelligence 🧠

**Base:** Forked from v12 (Stable)  
**Status:** Development  
**Date:** 2026-01-28

---

## 🎯 Goals for v13

1. **Fix SwarmMomentum** - Resolve NaN bug in consensus training
2. **Add LoRA** - Low-Rank Adaptation for efficient fine-tuning
3. **Swarm Workers** - Specialized personas (Tinker, Coder, Writer, DevOps)
4. **Checkpoint Resume** - Save/load training state

---

## ✅ Inherited from v12

- 11.3M parameters, 2.71 MB compressed
- 100% Q&A accuracy
- 9,233 tok/s training speed
- All 16 novel features working

---

## 📁 Structure

```
ghost_model_v13/
├── core/
│   ├── ghost_worker_v13.py    # Main model
│   ├── ternary_linear.py      # Ternary layers
│   ├── ternary_mamba.py       # Ternary SSM
│   ├── learned_codebook.py    # Codebook quantization
│   ├── mixture_of_depths.py   # MoD router
│   ├── cognitive_memory.py    # Memory system
│   └── lora.py                # [NEW] LoRA adapters
├── training/
│   ├── train_qna_debug.py     # Simple training
│   ├── train_perceptual.py    # Swarm training (fixing)
│   └── benchmark.py           # Benchmark script
├── swarm/                     # [NEW] Worker system
│   ├── router.py              # Task router
│   └── workers.py             # Specialized workers
└── README.md
```

---

## 🚀 Quick Start

```bash
# Run benchmark
python ghost_model_v13/training/benchmark.py

# Train on Q&A
python ghost_model_v13/training/train_qna_debug.py
```
