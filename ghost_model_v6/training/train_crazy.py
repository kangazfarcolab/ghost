"""
train_crazy.py - Crazy Training Combinations

Testing:
1. Curriculum + Skip (no caching overhead)
2. LR Warmup + Cosine Decay
3. Gradient Accumulation
4. ALL: Curriculum + LR Warmup + Accumulation
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import math
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from ghost_v6 import GhostV6
from datasets import get_all_datasets


def train_baseline(model, data, steps=200):
    """Baseline."""
    optimizer = optim.AdamW(learning_rate=3e-4)
    start = time.time()
    
    for step in range(steps):
        starts = mx.random.randint(0, len(data) - 65, (16,))
        x = mx.stack([data[int(s):int(s)+64] for s in starts.tolist()])
        y = mx.stack([data[int(s)+1:int(s)+65] for s in starts.tolist()])
        
        def loss_fn(m):
            return nn.losses.cross_entropy(m(x), y, reduction='mean')
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}: Loss={float(loss):.4f}")
    
    return time.time() - start, float(loss)


def train_curriculum_only(model, data, steps=200):
    """Just curriculum, no other overhead."""
    optimizer = optim.AdamW(learning_rate=3e-4)
    start = time.time()
    
    for step in range(steps):
        # Curriculum: 32 → 64 (not too short!)
        progress = step / steps
        seq_len = int(32 + progress * 32)
        
        starts = mx.random.randint(0, len(data) - seq_len - 1, (16,))
        x = mx.stack([data[int(s):int(s)+seq_len] for s in starts.tolist()])
        y = mx.stack([data[int(s)+1:int(s)+seq_len+1] for s in starts.tolist()])
        
        def loss_fn(m):
            return nn.losses.cross_entropy(m(x), y, reduction='mean')
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}: Loss={float(loss):.4f} | SeqLen={seq_len}")
    
    return time.time() - start, float(loss)


def train_lr_warmup_decay(model, data, steps=200):
    """LR warmup + cosine decay."""
    base_lr = 3e-4
    warmup_steps = 20
    
    optimizer = optim.AdamW(learning_rate=base_lr)
    start = time.time()
    
    for step in range(steps):
        # LR schedule: warmup then cosine decay
        if step < warmup_steps:
            lr = base_lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / (steps - warmup_steps)
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        
        optimizer.learning_rate = lr
        
        starts = mx.random.randint(0, len(data) - 65, (16,))
        x = mx.stack([data[int(s):int(s)+64] for s in starts.tolist()])
        y = mx.stack([data[int(s)+1:int(s)+65] for s in starts.tolist()])
        
        def loss_fn(m):
            return nn.losses.cross_entropy(m(x), y, reduction='mean')
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}: Loss={float(loss):.4f} | LR={lr:.6f}")
    
    return time.time() - start, float(loss)


def train_grad_accumulation(model, data, steps=200, accum_steps=4):
    """Gradient accumulation (larger effective batch)."""
    optimizer = optim.AdamW(learning_rate=3e-4)
    start = time.time()
    
    accumulated_grads = None
    
    for step in range(steps * accum_steps):
        starts = mx.random.randint(0, len(data) - 65, (4,))  # Smaller batch
        x = mx.stack([data[int(s):int(s)+64] for s in starts.tolist()])
        y = mx.stack([data[int(s)+1:int(s)+65] for s in starts.tolist()])
        
        def loss_fn(m):
            return nn.losses.cross_entropy(m(x), y, reduction='mean')
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        
        # Accumulate gradients
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = {k: accumulated_grads[k] + grads[k] for k in grads}
        
        # Update every accum_steps
        if (step + 1) % accum_steps == 0:
            # Average gradients
            avg_grads = {k: v / accum_steps for k, v in accumulated_grads.items()}
            optimizer.update(model, avg_grads)
            mx.eval(model.parameters(), optimizer.state)
            accumulated_grads = None
            
            actual_step = (step + 1) // accum_steps
            if actual_step % 50 == 0:
                print(f"  Step {actual_step}: Loss={float(loss):.4f}")
    
    return time.time() - start, float(loss)


def train_ultimate(model, data, steps=200):
    """Ultimate: Curriculum + LR warmup + larger batch."""
    base_lr = 5e-4  # Slightly higher LR
    warmup_steps = 30
    
    optimizer = optim.AdamW(learning_rate=base_lr)
    start = time.time()
    
    for step in range(steps):
        # LR schedule
        if step < warmup_steps:
            lr = base_lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / (steps - warmup_steps)
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        optimizer.learning_rate = lr
        
        # Curriculum: 32 → 64
        progress = step / steps
        seq_len = int(32 + progress * 32)
        
        # Larger batch (24 instead of 16)
        starts = mx.random.randint(0, len(data) - seq_len - 1, (24,))
        x = mx.stack([data[int(s):int(s)+seq_len] for s in starts.tolist()])
        y = mx.stack([data[int(s)+1:int(s)+seq_len+1] for s in starts.tolist()])
        
        def loss_fn(m):
            return nn.losses.cross_entropy(m(x), y, reduction='mean')
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}: Loss={float(loss):.4f} | SeqLen={seq_len} | LR={lr:.5f}")
    
    return time.time() - start, float(loss)


def benchmark():
    """Test all variations."""
    print("=" * 60)
    print("CRAZY TRAINING COMBINATIONS")
    print("=" * 60)
    
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
    t, l = train_baseline(model, data)
    results['baseline'] = (t, l)
    print(f"Time: {t:.1f}s | Loss: {l:.4f}\n")
    
    # 2. Curriculum only
    print("--- 2. CURRICULUM ONLY ---")
    model = GhostV6(dim=256, num_layers=6)
    mx.eval(model.parameters())
    t, l = train_curriculum_only(model, data)
    results['curriculum'] = (t, l)
    print(f"Time: {t:.1f}s | Loss: {l:.4f}\n")
    
    # 3. LR Warmup + Decay
    print("--- 3. LR WARMUP + DECAY ---")
    model = GhostV6(dim=256, num_layers=6)
    mx.eval(model.parameters())
    t, l = train_lr_warmup_decay(model, data)
    results['lr_schedule'] = (t, l)
    print(f"Time: {t:.1f}s | Loss: {l:.4f}\n")
    
    # 4. Ultimate combination
    print("--- 4. ULTIMATE (Curriculum + LR + Larger Batch) ---")
    model = GhostV6(dim=256, num_layers=6)
    mx.eval(model.parameters())
    t, l = train_ultimate(model, data)
    results['ultimate'] = (t, l)
    print(f"Time: {t:.1f}s | Loss: {l:.4f}\n")
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    baseline_time, baseline_loss = results['baseline']
    
    for name, (t, l) in results.items():
        speedup = baseline_time / t
        loss_diff = baseline_loss - l
        status = "🏆" if speedup > 1.0 and l <= baseline_loss else ("⚡" if speedup > 1.0 else ("📈" if l < baseline_loss else "❌"))
        print(f"{status} {name:15} | Time: {t:6.1f}s ({speedup:.2f}x) | Loss: {l:.4f} ({loss_diff:+.4f})")
    
    print("=" * 60)


if __name__ == "__main__":
    benchmark()
