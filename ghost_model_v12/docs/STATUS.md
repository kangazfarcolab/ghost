# Ghost v11 - The "Cortex" Model 🧠

## Overview
Ghost v11 is a **ternary-weight State Space Model (SSM)** designed for extreme efficiency and "true learning" capability. It combines a fast Mamba backbone with a persistent Cognitive Memory system.

## 🏗 Architecture
### 1. **Backbone: Ternary Mamba**
- **Weights:** Quantized to {-1, 0, +1} (2 bits/weight).
- **Core:** Simplified Mamba SSM (State Space Model) for linear-time sequence modeling.
- **Layers:** 6 Ternary Layers.
- **Dim:** 256 hidden dimension.

### 2. **Optimization: SwarmMomentum**
- **Distributed Consensus:** 4 "Workers" compute gradients on different data batches.
- **Consensus Metric:** Cosine similarity of gradients determines the "truth" of a direction.
- **Result:** Adaptive learning rate that accelerates when workers agree and slows down when they disagree.
- **Speed:** ~14,000 tokens/sec (Vectorized).

### 3. **Memory: Cognitive Loop**
- **One-Shot Learning:** Can key-value store information without backprop.
- **Curiosity Signal:** High prediction entropy triggers memory writes ("I am confused, I need to remember this").
- **Sleep Learning:** Contrastive consolidation during idle time (`sleep()` method).

## 📊 Current Status (Phase 13)
- [x] **Model Code:** Stable & Verified.
- [x] **Trainer:** Optimized `train_perceptual.py` (vmap enabled).
- [x] **Dataset:** TinyStories pipeline ready (Offline mode supported).
- [/] **Pre-training:** Ready to run.
- [ ] **Validation:** Pending 1.0 loss.

## 🔮 Roadmap to v12
Ghost v12 will build upon this foundation by adding:
1.  **Multi-Modal Inputs:** (Text + Vision?)
2.  **Hierarchical Planning:** (Goal - Subgoal)
3.  **Recursive Self-Improvement:** (Rewriting its own code)
