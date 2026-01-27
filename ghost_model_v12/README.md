# Ghost v12: The Self-Evolving Brain 🧬
**Protocol: STABLE BASE**

> "First we build a stable mind. Then we let it evolve."

Ghost v12 is the **base platform** for the next phase of evolution. It consolidates all the successful experiments from v11 into a robust, crash-proof architecture.

## 🏗️ Architecture (The "Safe Mode" Config)
1.  **Backbone**: Ternary Mamba (2-bit weights, `-1, 0, +1`).
2.  **FFN**: Codebook Linear (Clipped `[-3, 3]`) - **Enabled & Stable**.
3.  **Optimizer**: SGD + Momentum (robust to quantization noise).
4.  **Learning**:
    *   **Perceptual**: Swarm Training (4 workers).
    *   **Cognitive**: Hippocampus (One-Shot Learning).

## 🚀 Speed & performance
-   **Training Speed**: ~14,000 tokens/sec (Swarm Optimized).
-   **Convergence**: 10x faster loss drop with SGD (lr=3e-3).
-   **Stability**: 100% (No NaN gradients).

## 🔮 Next Steps (Self-Evolution)
Now that the brain is stable, we will implement:
1.  **Self-Rewrite**: The ability for the model to modify its own Python code.
2.  **Vision**: Adding a retinal layer.
3.  **Dreaming**: Offline memory consolidation.

---
*Created from Ghost v11-Stable on 2026-01-27*
