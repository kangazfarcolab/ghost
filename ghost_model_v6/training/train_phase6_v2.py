"""
train_phase6_v2.py - Fixed Phase 6 Training

Fixes:
1. Warmup period before skip training activates
2. Test each technique individually
3. Lighter skip check (less overhead)
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from ghost_v6 import GhostV6
from datasets import get_all_datasets


def train_baseline(model, data, steps=200):
    """Baseline training."""
    optimizer = optim.AdamW(learning_rate=3e-4)
    start = time.time()
    final_loss = 0
    
    for step in range(steps):
        starts = mx.random.randint(0, len(data) - 65, (16,))
        x = mx.stack([data[int(s):int(s)+64] for s in starts.tolist()])
        y = mx.stack([data[int(s)+1:int(s)+65] for s in starts.tolist()])
        
        def loss_fn(m):
            return nn.losses.cross_entropy(m(x), y, reduction='mean')
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        final_loss = float(loss)
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}: Loss={final_loss:.4f}")
    
    return time.time() - start, final_loss


def train_byte_aware_lr(model, data, steps=200):
    """Test: Byte-Aware Learning Rate only."""
    
    # Byte categories
    COMMON = set(ord(c) for c in 'etaoinshrdlu ')
    SPECIAL = set(ord(c) for c in '{}[]().,;:!?+-*/=')
    
    def get_lr_multiplier(x):
        x_flat = x.reshape(-1).tolist()
        common_count = sum(1 for b in x_flat if b in COMMON)
        special_count = sum(1 for b in x_flat if b in SPECIAL)
        total = len(x_flat)
        
        # More special chars = higher LR (they need more learning)
        special_ratio = special_count / max(1, total)
        return 1.0 + special_ratio * 2.0  # Up to 3x LR for special-heavy
    
    optimizer = optim.AdamW(learning_rate=3e-4)
    start = time.time()
    final_loss = 0
    
    for step in range(steps):
        starts = mx.random.randint(0, len(data) - 65, (16,))
        x = mx.stack([data[int(s):int(s)+64] for s in starts.tolist()])
        y = mx.stack([data[int(s)+1:int(s)+65] for s in starts.tolist()])
        
        # Adjust LR based on bytes
        lr_mult = get_lr_multiplier(x)
        optimizer.learning_rate = 3e-4 * lr_mult
        
        def loss_fn(m):
            return nn.losses.cross_entropy(m(x), y, reduction='mean')
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        final_loss = float(loss)
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}: Loss={final_loss:.4f} | LR mult={lr_mult:.2f}")
    
    return time.time() - start, final_loss


def train_skip_with_warmup(model, data, steps=200, warmup=100):
    """Test: Predictive Skip with warmup period."""
    
    optimizer = optim.AdamW(learning_rate=3e-4)
    start = time.time()
    final_loss = 0
    skip_count = 0
    total_after_warmup = 0
    
    for step in range(steps):
        starts = mx.random.randint(0, len(data) - 65, (16,))
        x = mx.stack([data[int(s):int(s)+64] for s in starts.tolist()])
        y = mx.stack([data[int(s)+1:int(s)+65] for s in starts.tolist()])
        
        # Skip check ONLY after warmup
        if step > warmup:
            total_after_warmup += 1
            logits = model(x)
            probs = mx.softmax(logits, axis=-1)
            correct_probs = mx.take_along_axis(probs, y[:, :, None], axis=-1)
            confidence = float(mx.mean(correct_probs).item())
            
            if confidence > 0.7:  # High confidence = skip
                skip_count += 1
                continue
        
        def loss_fn(m):
            return nn.losses.cross_entropy(m(x), y, reduction='mean')
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        final_loss = float(loss)
        
        if (step + 1) % 50 == 0:
            skip_rate = skip_count / max(1, total_after_warmup) * 100
            print(f"  Step {step+1}: Loss={final_loss:.4f} | Skip={skip_rate:.0f}%")
    
    return time.time() - start, final_loss


def train_simple_curriculum(model, data, steps=200):
    """Test: Simple curriculum - start with short sequences."""
    
    optimizer = optim.AdamW(learning_rate=3e-4)
    start = time.time()
    final_loss = 0
    
    for step in range(steps):
        # Curriculum: start with short sequences, increase over time
        progress = step / steps
        seq_len = int(16 + progress * 48)  # 16 -> 64
        
        starts = mx.random.randint(0, len(data) - seq_len - 1, (16,))
        x = mx.stack([data[int(s):int(s)+seq_len] for s in starts.tolist()])
        y = mx.stack([data[int(s)+1:int(s)+seq_len+1] for s in starts.tolist()])
        
        def loss_fn(m):
            return nn.losses.cross_entropy(m(x), y, reduction='mean')
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        final_loss = float(loss)
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}: Loss={final_loss:.4f} | SeqLen={seq_len}")
    
    return time.time() - start, final_loss


def benchmark_all():
    """Compare all techniques."""
    
    print("=" * 60)
    print("PHASE 6 v2 - INDIVIDUAL TECHNIQUE TESTING")
    print("=" * 60)
    
    # Prepare data
    datasets = get_all_datasets()
    qa_data = datasets['math'][:200]
    data_str = "".join([(q + a + "\n") * 50 for q, a in qa_data])
    data = mx.array([ord(c) for c in data_str], dtype=mx.int32)
    print(f"Data: {len(data)} bytes\n")
    
    results = {}
    
    # 1. Baseline
    print("--- 1. BASELINE ---")
    model = GhostV6(dim=256, num_layers=6)
    mx.eval(model.parameters())
    time_taken, loss = train_baseline(model, data)
    results['baseline'] = (time_taken, loss)
    print(f"Time: {time_taken:.1f}s | Final Loss: {loss:.4f}\n")
    
    # 2. Byte-Aware LR
    print("--- 2. BYTE-AWARE LR ---")
    model = GhostV6(dim=256, num_layers=6)
    mx.eval(model.parameters())
    time_taken, loss = train_byte_aware_lr(model, data)
    results['byte_lr'] = (time_taken, loss)
    print(f"Time: {time_taken:.1f}s | Final Loss: {loss:.4f}\n")
    
    # 3. Skip with Warmup
    print("--- 3. SKIP WITH WARMUP ---")
    model = GhostV6(dim=256, num_layers=6)
    mx.eval(model.parameters())
    time_taken, loss = train_skip_with_warmup(model, data)
    results['skip_warmup'] = (time_taken, loss)
    print(f"Time: {time_taken:.1f}s | Final Loss: {loss:.4f}\n")
    
    # 4. Simple Curriculum
    print("--- 4. SIMPLE CURRICULUM ---")
    model = GhostV6(dim=256, num_layers=6)
    mx.eval(model.parameters())
    time_taken, loss = train_simple_curriculum(model, data)
    results['curriculum'] = (time_taken, loss)
    print(f"Time: {time_taken:.1f}s | Final Loss: {loss:.4f}\n")
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    baseline_time = results['baseline'][0]
    baseline_loss = results['baseline'][1]
    
    for name, (t, l) in results.items():
        speedup = baseline_time / t
        loss_diff = baseline_loss - l
        status = "✅" if speedup > 1.0 or l < baseline_loss else "❌"
        print(f"{status} {name:15} | Time: {t:6.1f}s ({speedup:.2f}x) | Loss: {l:.4f} ({loss_diff:+.4f})")
    
    print("=" * 60)


if __name__ == "__main__":
    benchmark_all()
