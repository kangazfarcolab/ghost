# Ghost Model v5

> The Next Evolution - Everything We've Learned, Plus New Innovations

---

## 🎉 v5 Results (Achieved!)

| Model | Accuracy | Hard Tasks | Time | Params |
|:---|:---|:---|:---|:---|
| v5 Original | 100% | N/A | 783s | 7.17M |
| **v5 Fast** | **100%** | **100%** | **771s** | **6.58M** |

### New Features Validated:
- ✅ **Sparse Attention** (at layers 3, 5) - Global context
- ✅ **Per-Layer Memory** (at layers 2, 4) - Layerwise retrieval
- ✅ **Hard Tasks** - Multi-hop reasoning (5/5)
- ✅ **Self-Distillation v2** - 10→171 pairs (17x), 60% generalization

---

## Complete Feature History

### v1: Foundation (Baseline)

| Feature | Description | Status | Impact |
|:---|:---|:---|:---|
| **Ghost Weights (2-bit Palette)** | Learnable 4-value codebook, STE for gradients | ✅ Novel | 8x smaller storage |
| **Mamba SSM** | State-space model, O(N) complexity | ✅ Implemented | Infinite context |
| **Parallel Associative Scan** | O(log N) parallel Mamba computation | ✅ Implemented | GPU-efficient |
| **MoE (8 Experts, Top-2)** | Mixture of Experts routing | ✅ Implemented | 4x capacity |
| **ByteGrouper (GROUP_SIZE=4)** | Fixed byte grouping | ❌ Deprecated | Caused confusion |

**v1 Result:** 40% accuracy, 29M params

---

### v2: Extended Context

| Feature | Description | Status | Impact |
|:---|:---|:---|:---|
| **Longer Context (2048)** | Extended sequence length | ✅ Implemented | Better patterns |
| **Byte-Level Processing** | Direct byte input, no BPE | ✅ Implemented | Simpler pipeline |

**v2 Result:** 60% accuracy, 29M params

---

### v3: Revolutionary Experiments

| Feature | Description | Status | Impact |
|:---|:---|:---|:---|
| **State-Space Tokenization (SST)** | Hidden state velocity detects word boundaries | ✅ NOVEL | +60% accuracy |
| **Sparse Byte Routing** | Adaptive depth per byte (important=6 layers, common=2) | ✅ NOVEL | 2.3x faster, 5x smaller |
| **Predictive Coding** | Skip compute on predictable bytes | ✅ NOVEL | 3.4x faster |
| **Recursive Self-Compression** | Model predicts own weights, store residuals | ✅ NOVEL | 7.5x smaller files |
| **Holographic Memory** | HDC-based fact storage | ❌ Failed | Memory never queried |

**v3 Result:** 100% accuracy, 5.9M params, 340s training

---

### v4: Unified Model + Memory Fix

| Feature | Description | Status | Impact |
|:---|:---|:---|:---|
| **All v3 Features Combined** | SST + Sparse + Predictive in one model | ✅ Implemented | Robust 100% |
| **Memory Augmentation** | Cross-attention to fact memory with gating | ✅ FIXED! | External knowledge |
| **Checkpointing** | Pause/resume training | ✅ Implemented | Practical training |

**v4 Result:** 100% accuracy, 5.98M params, 479s training

---

## What's Working (v4 Final Architecture)

```
┌──────────────────────────────────────────────────────────────────┐
│                      GHOST MODEL v4                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT: Raw Bytes [0-255]                                        │
│         ↓                                                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ STATE-SPACE TOKENIZER (SST)                                │  │
│  │ • Byte Embedding (256 → dim)                               │  │
│  │ • Conv1d for local context                                 │  │
│  │ • Boundary detection from state velocity                   │  │
│  │ • Output: h weighted by boundary strength                  │  │
│  └────────────────────────────────────────────────────────────┘  │
│         ↓                                                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ DEPTH ROUTER (Sparse Byte Routing)                         │  │
│  │ • depth_predictor: Linear(dim, 1)                          │  │
│  │ • byte_importance: Embedding(256, 1)                       │  │
│  │ • Output: depth ∈ [0, num_layers] per byte                 │  │
│  └────────────────────────────────────────────────────────────┘  │
│         ↓                                                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ SURPRISE DETECTOR (Predictive Coding)                      │  │
│  │ • Quick prediction head                                    │  │
│  │ • Compare with actual next byte                            │  │
│  │ • Output: surprise_mask (1=process, 0=skip)                │  │
│  └────────────────────────────────────────────────────────────┘  │
│         ↓                                                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ SPARSE MAMBA LAYERS (x6)                                   │  │
│  │ For each layer i:                                          │  │
│  │   • depth_mask = sigmoid((depths - i) * 5)                 │  │
│  │   • combined_mask = depth_mask * surprise_mask             │  │
│  │   • h = h + Mamba(Norm(h)) * mask                          │  │
│  │   • h = h + FFN(Norm(h)) * mask                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│         ↓                                                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ MEMORY AUGMENTATION (v4 addition)                          │  │
│  │ • Query = mean(h) projected                                │  │
│  │ • Soft attention over stored facts                         │  │
│  │ • Gate learns when to use memory                           │  │
│  │ • h = h + gate * memory_output                             │  │
│  └────────────────────────────────────────────────────────────┘  │
│         ↓                                                        │
│  OUTPUT: Linear(dim, 256) → next byte logits                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## v5: The Next Evolution

### Selected Features for v5

Based on our analysis, here's what we're implementing:

| Priority | Feature | Reason | Difficulty |
|:---|:---|:---|:---|
| 1 | **Self-Distillation** | Scale 10 Q&A → 1000+ for generalization | Medium |
| 2 | **Sparse Attention + Mamba Hybrid** | Best of both: local (Mamba) + global (attention) | Medium |
| 3 | **Memory at Every Layer** | Different layers use memory differently | Easy |
| 4 | **Fill-in-the-Middle Training** | Bidirectional context without 2x cost | Medium |

### NOT Including (and why)

| Feature | Reason to Skip |
|:---|:---|
| Neural ODE SSM | Academic only, no practical benefit |
| Byte-Level Diffusion | Too experimental, unstable training |
| Full Bidirectional Mamba | 2x compute, fill-in-middle is better |

---

## v5 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      GHOST MODEL v5                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT: Raw Bytes [0-255]                                        │
│         ↓                                                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ STATE-SPACE TOKENIZER (from v4)                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│         ↓                                                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ ROUTING LAYER (from v4)                                    │  │
│  │ • Depth Router + Surprise Detector                         │  │
│  └────────────────────────────────────────────────────────────┘  │
│         ↓                                                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ HYBRID LAYERS (x6) ← NEW!                                  │  │
│  │                                                            │  │
│  │ For each layer i:                                          │  │
│  │   ┌──────────────────────────────────────────────────────┐ │  │
│  │   │ 1. Mamba Block (local processing)                    │ │  │
│  │   │    h = h + Mamba(Norm(h)) * mask                     │ │  │
│  │   └──────────────────────────────────────────────────────┘ │  │
│  │   ┌──────────────────────────────────────────────────────┐ │  │
│  │   │ 2. Memory Query (every layer) ← NEW!                 │ │  │
│  │   │    mem = query_memory(h, layer_idx=i)                │ │  │
│  │   │    h = h + layer_gate[i] * mem                       │ │  │
│  │   └──────────────────────────────────────────────────────┘ │  │
│  │   ┌──────────────────────────────────────────────────────┐ │  │
│  │   │ 3. Sparse Attention (every 2 layers) ← NEW!          │ │  │
│  │   │    if i % 2 == 1:                                    │ │  │
│  │   │      h = h + SparseAttention(h, stride=64)           │ │  │
│  │   └──────────────────────────────────────────────────────┘ │  │
│  │   ┌──────────────────────────────────────────────────────┐ │  │
│  │   │ 4. FFN Block                                         │ │  │
│  │   │    h = h + FFN(Norm(h)) * mask                       │ │  │
│  │   └──────────────────────────────────────────────────────┘ │  │
│  │                                                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│         ↓                                                        │
│  OUTPUT: Linear(dim, 256) → next byte logits                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## v5 New Components

### 1. Sparse Attention (Global Context)

```python
class SparseAttention(nn.Module):
    """
    Efficient attention that only attends to every Nth position.
    O(N * N/stride) instead of O(N²)
    """
    
    def __init__(self, dim, num_heads=4, stride=64):
        self.stride = stride
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def __call__(self, x):
        B, L, D = x.shape
        
        # All positions create queries
        Q = self.q_proj(x)  # [B, L, D]
        
        # Only every stride-th position is a key/value
        sparse_indices = mx.arange(0, L, self.stride)
        x_sparse = x[:, sparse_indices, :]  # [B, L//stride, D]
        
        K = self.k_proj(x_sparse)  # [B, L//stride, D]
        V = self.v_proj(x_sparse)  # [B, L//stride, D]
        
        # Attention: each position attends to sparse keys
        # [B, L, D] @ [B, D, L//stride] -> [B, L, L//stride]
        scores = mx.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(self.head_dim)
        weights = mx.softmax(scores, axis=-1)
        
        # [B, L, L//stride] @ [B, L//stride, D] -> [B, L, D]
        out = mx.matmul(weights, V)
        
        return self.out_proj(out)
```

### 2. Per-Layer Memory Query

```python
class LayerwiseMemory(nn.Module):
    """Memory queried at each layer with layer-specific gates."""
    
    def __init__(self, dim, num_layers):
        self.memory_keys = []
        self.memory_values = []
        
        # Each layer has its own query projection and gate
        self.layer_query_projs = [nn.Linear(dim, dim) for _ in range(num_layers)]
        self.layer_gates = [nn.Linear(dim, 1) for _ in range(num_layers)]
    
    def query_at_layer(self, h, layer_idx):
        """Query memory with layer-specific projection."""
        if len(self.memory_keys) == 0:
            return mx.zeros_like(h)
        
        # Layer-specific query
        Q = self.layer_query_projs[layer_idx](h)
        
        # Attention over memory
        K = mx.stack(self.memory_keys)
        V = mx.stack(self.memory_values)
        
        scores = mx.matmul(Q, K.T) / math.sqrt(h.shape[-1])
        weights = mx.softmax(scores, axis=-1)
        retrieved = mx.matmul(weights, V)
        
        # Layer-specific gate
        gate = mx.sigmoid(self.layer_gates[layer_idx](h))
        
        return gate * retrieved
```

### 3. Self-Distillation Pipeline

```python
class SelfDistillation:
    """Generate synthetic training data from the model itself."""
    
    def __init__(self, model, base_qa_pairs):
        self.model = model
        self.base_qa = base_qa_pairs
    
    def generate_variations(self, num_variations=100):
        """Generate question variations."""
        synthetic_data = []
        
        for question, answer in self.base_qa:
            # Generate paraphrases
            variations = [
                question,  # Original
                self._rephrase(question),
                self._add_noise(question),
                self._synonym_replace(question),
            ]
            
            for var in variations:
                # Only keep if model predicts correctly with high confidence
                pred, confidence = self.model.generate_with_confidence(var)
                if confidence > 0.8 and pred.strip() == answer.strip():
                    synthetic_data.append((var, answer))
        
        return synthetic_data
    
    def _rephrase(self, q):
        # "What is 2+2?" -> "2+2 equals?" -> "Calculate 2+2"
        ...
    
    def _add_noise(self, q):
        # Add typos, case changes
        ...
```

---

## Implementation Roadmap

### Phase 1: Core v5 ✅ DONE
- [x] Copy v4 as base
- [x] Add SparseAttention module
- [x] Add per-layer memory query
- [x] Test on 10-question benchmark → **100%**

### Phase 2: Self-Distillation ✅ DONE
- [x] Implement variation generator (v2 with typos, synonyms)
- [x] Generate 171 Q&A variations (17x expansion)
- [x] Train on expanded dataset
- [x] Generalization: 40% → **60%**

### Phase 3: Fill-in-the-Middle ⏭️ SKIPPED
- [x] Add FIM training objective (implemented)
- [x] Tested with mixed objectives
- ⏭️ **Skipped** - Not beneficial for Q&A tasks (0% FIM accuracy, 100% standard)

### Phase 4: Optimization ✅ DONE
- [x] Profile and optimize sparse attention
- [x] Tune memory query frequency (layers 2,4 only)
- [x] v5 Fast: 6.58M params, 771s training

---

## Success Metrics

| Metric | v4 Baseline | v5 Target |
|:---|:---|:---|
| 10-Q Accuracy | 100% | 100% (maintain) |
| 100-Q Accuracy | ~60%? | 90%+ |
| Training Time | 479s | <600s |
| Parameters | 5.98M | <8M |
| Generalization | Low | High (self-distill) |

---

## Files Structure

```
ghost_model_v5/
├── README.md                    # This file
├── core/
│   ├── ghost_v5.py             # Main model
│   ├── sparse_attention.py      # New: sparse attention module
│   ├── layerwise_memory.py      # New: per-layer memory
│   └── self_distillation.py     # New: synthetic data generation
├── training/
│   ├── train_v5.py             # Training script
│   ├── train_fim.py            # Fill-in-middle training
│   └── distill.py              # Self-distillation pipeline
├── tests/
│   ├── test_10q.py             # 10-question benchmark
│   ├── test_100q.py            # Extended benchmark
│   └── test_generalization.py  # Generalization tests
└── docs/
    ├── CHANGELOG.md            # What changed from v4
    └── BENCHMARKS.md           # All benchmark results
```

---

## Summary: The Complete Ghost Model Story

| Version | Key Innovation | Accuracy | Params | Status |
|:---|:---|:---|:---|:---|
| v1 | Ghost Weights + Mamba | 40% | 29M | ✅ Foundation |
| v2 | Extended Context | 60% | 29M | ✅ Improved |
| v3 | SST + Sparse + Predictive | 100% | 5.9M | ✅ Breakthrough |
| v4 | Unified + Memory Fix | 100% | 5.98M | ✅ Complete |
| **v5** | **Hybrid Attention + Self-Distill** | **100%+** | **<8M** | 🚧 In Progress |

---

## Novel Contributions (Paper-Worthy)

1. **Ghost Weights (2-bit Palette)** - Learnable 4-value codebook with STE
2. **State-Space Tokenization** - Hidden state velocity as boundary detector
3. **Sparse Byte Routing** - Adaptive compute per byte importance
4. **Predictive Coding for LMs** - Skip compute on predictable bytes
5. **Recursive Self-Compression** - Model predicts own weights

These are genuinely novel ideas not seen in published research (as of our knowledge).
