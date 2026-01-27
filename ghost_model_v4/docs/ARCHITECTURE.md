# Ghost v4 - Architecture Deep Dive

Complete technical architecture of Ghost Model v4.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GHOST MODEL v4                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  INPUT: Raw Bytes (0-255)                           │   │
│  └────────────────────────┬────────────────────────────┘   │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  STATE-SPACE TOKENIZER                              │   │
│  │  • Byte Embedding (256 → DIM)                       │   │
│  │  • Conv1d for local context                         │   │
│  │  • Boundary detection from state velocity           │   │
│  └────────────────────────┬────────────────────────────┘   │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ROUTING LAYER                                      │   │
│  │  • Depth Router: How many layers per byte?          │   │
│  │  • Surprise Detector: Is this byte predictable?     │   │
│  └────────────────────────┬────────────────────────────┘   │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SPARSE MAMBA LAYERS (x6)                           │   │
│  │  • RMSNorm → Mamba SSM → Masked residual            │   │
│  │  • RMSNorm → FFN → Masked residual                  │   │
│  │  • Skip layers based on routing                     │   │
│  └────────────────────────┬────────────────────────────┘   │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  OUTPUT                                             │   │
│  │  • RMSNorm                                          │   │
│  │  • Linear → 256 (next byte prediction)              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. State-Space Tokenizer

**Purpose:** Learn natural token boundaries instead of fixed GROUP_SIZE

```python
class StateSpaceTokenizer:
    byte_embed: Embedding(256, dim)  # Byte to vector
    conv1d: Conv1d(dim, dim, k=4)    # Local context
    state_proj: Linear(dim, d_state) # Project to state space
    boundary_head: Linear(d_state*2, 1)  # Detect boundaries
    
    def __call__(x):
        h = silu(conv1d(byte_embed(x)))
        states = state_proj(h)
        velocity = abs(states[t] - states[t-1])
        boundaries = sigmoid(boundary_head([states, velocity]))
        return h * (1 + boundaries), boundaries
```

**Key Insight:** State velocity is high at word boundaries

---

### 2. Depth Router

**Purpose:** Predict how many layers each byte needs

```python
class DepthRouter:
    depth_predictor: Linear(dim, 1)
    byte_importance: Embedding(256, 1)  # Learned per byte
    
    def __call__(h, byte_ids):
        context_score = depth_predictor(h)
        byte_score = byte_importance(byte_ids)
        return sigmoid(context_score + byte_score) * max_depth
```

**Key Insight:** Punctuation/numbers need more layers than common letters

---

### 3. Surprise Detector

**Purpose:** Identify which bytes need full processing

```python
class SurpriseDetector:
    quick_pred: Linear(dim, 256)  # Cheap prediction
    
    def get_surprise_mask(h, x):
        probs = softmax(quick_pred(h))
        correct_prob = probs[actual_next_byte]
        surprise = 1 - correct_prob
        return (surprise > threshold)
```

**Key Insight:** Skip compute on predictable bytes like "The "

---

### 4. Sparse Mamba Layer

**Purpose:** Apply Mamba only where routing says to

```python
class SparsePredictveMambaLayer:
    norm1: RMSNorm(dim)
    mamba: MambaSSM(dim)
    norm2: RMSNorm(dim)
    ffn: Linear(dim, dim*4) → GELU → Linear(dim*4, dim)
    
    def __call__(x, depth_mask, surprise_mask):
        combined_mask = depth_mask * surprise_mask
        x = x + mamba(norm1(x)) * combined_mask
        x = x + ffn(norm2(x)) * combined_mask
        return x
```

**Key Insight:** Soft masking allows gradient flow

---

### 5. Mamba SSM (Core)

**Purpose:** O(N) sequence modeling with infinite context

```
Recurrence:
h_t = A * h_{t-1} + B * x_t
y_t = C * h_t

Parallel Scan (O(log N)):
(a1, b1) ⊕ (a2, b2) = (a1*a2, a2*b1 + b2)
```

**Key Insight:** Constant memory per token, unlimited context

---

## Data Flow

```
Input: "Hello"
  ↓
[72, 101, 108, 108, 111]  # ASCII bytes
  ↓
StateSpaceTokenizer:
  • Embed: [72, 101, 108, 108, 111] → [5, 256] vectors
  • Boundaries: [0.1, 0.1, 0.1, 0.1, 0.9] (high at end)
  ↓
Routing:
  • Depths: [3, 2, 2, 2, 4]  # 'H' and 'o' need more layers
  • Surprise: [1, 0, 0, 0, 1]  # 'H' and 'o' surprising
  ↓
Layers 0-5:
  • Each layer checks: depth > layer_idx AND surprise > 0.5
  • Apply Mamba + FFN only if both true
  ↓
Output:
  • Linear → 256 logits
  • Predicts next byte for each position
```

---

## Memory Layout

| Component | Parameters | Memory |
|:---|:---|:---|
| StateSpaceTokenizer | ~200K | 0.8 MB |
| DepthRouter | ~70K | 0.3 MB |
| SurpriseDetector | ~70K | 0.3 MB |
| Mamba Layers (x6) | ~5.5M | 22 MB |
| Output | ~66K | 0.3 MB |
| **Total** | **~5.98M** | **~24 MB** |

---

## Comparison: Traditional vs Ghost v4

| Aspect | Transformer | Ghost v4 |
|:---|:---|:---|
| Attention | O(N²) | O(N) Mamba |
| Context | Limited (4K-32K) | **Unlimited** |
| Tokenizer | BPE (fixed) | Learned (SST) |
| Weights | 16-bit | 2-bit (Ghost) |
| Compute | Dense | Sparse (routing) |
| Memory | Grows with context | Constant |
