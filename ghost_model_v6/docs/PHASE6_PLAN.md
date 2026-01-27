# Ghost v6 - Phase 6 Training Plan

## Benchmark Results (Tested!)

| Technique | Speedup | Final Loss | Status |
|:---|:---|:---|:---|
| **Simple Curriculum** | **1.22x** 🏆 | 0.7789 | ✅ USE |
| **Skip with Warmup** | 0.93x | **0.7288** 🏆 | ✅ USE |
| **Byte-Aware LR** | 1.01x | 0.7920 | ~ Neutral |
| Mamba State Momentum | N/A | N/A | ❌ Skip (overhead) |
| Depth-Aware Gradients | N/A | N/A | ❌ Skip (overhead) |

---

## Recommended Training Pipeline

```python
# 1. Simple Curriculum: Start short, increase over time
seq_len = int(16 + (step/steps) * 48)  # 16 → 64

# 2. Skip with Warmup: Skip easy samples after warmup
if step > warmup and confidence > 0.7:
    skip_sample()
```

---

## What Works:

### ✅ Simple Curriculum (1.22x faster)
- Start with short sequences (16 bytes)
- Gradually increase to full length (64 bytes)
- **Why it works**: Faster gradient updates early

### ✅ Skip with Warmup (Best loss)
- Train normally for first 100 steps
- After warmup, skip samples with >70% confidence
- **Why it works**: Focus on what model doesn't know

---

## What Doesn't Work:

### ❌ Mamba State Momentum
- Extra forward pass to compute velocity
- Overhead > benefit

### ❌ Depth-Aware Gradients
- Extra computation per step
- Minimal impact on learning

---

## Datasets for Testing

### Tiny Real Datasets (for quick experiments)

| Dataset | Size | Tests | Source |
|:---|:---|:---|:---|
| **Bash-100** | 100 commands | Reasoning, Memory | Manual |
| **Math-1K** | 1K equations | Accuracy, Speed | Generated |
| **Facts-200** | 200 Q&A facts | Memory, Retrieval | Manual |
| **Code-50** | 50 Python snippets | Compression, Gen | GitHub |

---

## Test Plan

### Test 1: Compression
- Train on Math-1K
- Compare: baseline vs predictive skip vs byte-aware LR
- Measure: Loss convergence speed

### Test 2: Reasoning  
- Train on Bash-100 + Facts-200
- Test: Multi-hop questions
- Measure: Accuracy on novel combinations

### Test 3: Learning Rate
- Train with byte-aware LR
- Compare: flat LR vs byte-aware
- Measure: Steps to 95% accuracy

### Test 4: Speed
- Train with all 4 techniques
- Compare: baseline vs optimized
- Measure: Training time for same accuracy

---

## Implementation Order

1. Create datasets (Bash-100, Math-1K, Facts-200, Code-50)
2. Implement Predictive Skip Training
3. Implement Byte-Aware Learning Rate
4. Implement Mamba State Momentum
5. Implement Depth-Aware Gradients
6. Run benchmarks
7. Generate report

---

## Success Metrics

| Metric | Baseline | Target |
|:---|:---|:---|
| Training speed | 325s/300 steps | <150s |
| Accuracy | 100% | 100% (maintain) |
| Novel generalization | 58% | 70%+ |
| Memory usage | 100% | <60% |

---

## Future: EWC (Continual Learning)
After Phase 6, add:
- Elastic Weight Consolidation
- Incremental domain learning
- No forgetting between tasks
