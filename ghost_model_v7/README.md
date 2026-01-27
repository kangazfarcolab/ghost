# Ghost v7: Tiered Architecture

> Multi-tier neural swarm with Worker, Expert, and Thinker agents.

---

## 🏗️ Architecture

| Tier | Params | Memory | Best For |
|:---|:---|:---|:---|
| **Worker** | 6.58M | ~25 MB | Commands, automation, parallel tasks |
| **Expert** | 26.47M | ~101 MB | Explanations, domain expertise |
| **Thinker** | 44.40M | ~169 MB | Planning, synthesis, long generation |

---

## 🎯 Swarm Capacity (16GB Mac)

```
Tiered Swarm:
  2 Thinkers  →   338 MB (planning, synthesis)
  10 Experts  → 1,010 MB (domain specialists)
  100 Workers → 2,500 MB (parallel execution)
  ─────────────────────────
  Total       → 3,848 MB (3.8 GB)
  
  ✅ Fits easily with room to spare!
```

---

## 🚀 Quick Start

```bash
# Activate environment
source /path/to/venv/bin/activate

# Test all tiers
cd ghost_model_v7
python tests/test_tiers.py

# Use individual tiers
from core.ghost_worker import GhostWorker
from core.ghost_expert import GhostExpert
from core.ghost_thinker import GhostThinker

worker = GhostWorker()   # 6.58M params
expert = GhostExpert()   # 26.47M params
thinker = GhostThinker() # 44.40M params
```

---

## 📊 Tier Comparison

| Feature | Worker | Expert | Thinker |
|:---|:---|:---|:---|
| Layers | 6 | 8 | 10 |
| Dimension | 256 | 448 | 512 |
| Attention heads | N/A | 8 | 8 |
| Attention stride | 128 | 64 | 32 |
| Memory layers | 2, 4 | 1,3,5,7 | Every odd |
| Attn layers | 3, 5 | 2,4,6 | Every even |
| Init time | 0.01s | 0.03s | 0.05s |
| Forward time | 0.02s | 0.04s | 0.05s |

---

## 🔬 Unique Features per Tier

### Worker (6M)
- Identical to Ghost v6
- Optimized for speed and parallelism
- 100+ can run simultaneously

### Expert (25M)
- Multi-head sparse attention (8 heads)
- Denser attention stride (64)
- Enhanced memory with key projection
- Good for 50-100 token outputs

### Thinker (50M)
- Hierarchical memory (short-term + long-term)
- Rotary position embeddings
- Densest attention (stride 32)
- Good for 200+ token outputs

---

## 🐝 War Room Pattern

```
User: "Create production EKS infrastructure"
                     ↓
┌─────────────────────────────────────────┐
│ THINKER (Planner)                       │
│ "Breaking down task..."                 │
│ Plan: VPC → EKS → IAM → Monitoring      │
└─────────────────────────────────────────┘
        ↓          ↓          ↓
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ Expert  │ │ Expert  │ │ Expert  │
   │ (VPC)   │ │ (EKS)   │ │ (IAM)   │
   └─────────┘ └─────────┘ └─────────┘
        ↓          ↓          ↓
   Workers    Workers    Workers
   (10 each)  (10 each)  (10 each)
        ↓          ↓          ↓
┌─────────────────────────────────────────┐
│ THINKER (Synthesizer)                   │
│ "Combining outputs, validating refs..." │
└─────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ghost_model_v7/
├── core/
│   ├── ghost_worker.py   # 6.58M Worker tier
│   ├── ghost_expert.py   # 26.47M Expert tier
│   └── ghost_thinker.py  # 44.40M Thinker tier
├── tests/
│   └── test_tiers.py     # Verification script
├── training/             # (Coming soon)
└── docs/
    └── README.md
```

---

## 🔗 Related

- [Ghost Infinite](../ghost_swarm/docs/GHOST_INFINITE.md) - 5 pillars of unified generation
- [Ghost Swarm](../ghost_swarm/docs/SWARM_ARCHITECTURE.md) - Multi-agent orchestration
- [Ghost v6](../ghost_model_v6/README.md) - Production base model
