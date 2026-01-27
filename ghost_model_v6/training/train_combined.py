"""
train_combined.py - All Phase 6 Techniques Combined

Combines all 4 techniques with efficient caching:
1. Curriculum (short → long sequences)
2. Skip with Warmup (skip confident samples)
3. State Momentum (weight by state change) - CACHED
4. Depth Gradients (weight by predicted depth) - CACHED
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


class CombinedTrainer:
    """All Phase 6 techniques combined with caching."""
    
    def __init__(self, model):
        self.model = model
        self.warmup_steps = 100  # More warmup
        self.skip_threshold = 0.85  # Higher threshold = skip less
        
    def forward_with_cache(self, x):
        """Single forward pass that caches all intermediate values."""
        B, L = x.shape
        
        # Tokenization
        h = self.model.byte_embed(x)
        h = nn.silu(self.model.conv1d(h)[:, :L, :])
        
        # Cache: Depth prediction (already computed in forward!)
        depth_score = self.model.depth_predictor(h).squeeze(-1)
        byte_score = self.model.byte_importance(x).squeeze(-1)
        depths = mx.sigmoid(depth_score + byte_score) * self.model.num_layers
        mean_depth = mx.mean(depths)
        
        # Cache: State velocity through layers
        state_velocities = []
        for i, layer in enumerate(self.model.layers):
            h_prev = h
            
            depth_mask = mx.sigmoid((depths - i) * 5)
            combined_mask = depth_mask.reshape(B, L, 1)
            
            h = h + layer['mamba'](layer['norm1'](h)) * combined_mask
            
            # Track state change
            velocity = mx.mean(mx.abs(h - h_prev))
            state_velocities.append(velocity)
            
            # Memory query
            if i in [2, 4]:
                h = h + self.model.query_memory(h)
            
            # Sparse attention
            if 'sparse_attn' in layer:
                h = h + layer['sparse_attn'](h) * combined_mask
            
            h = h + layer['ffn'](layer['norm2'](h)) * combined_mask
        
        # Final output
        logits = self.model.output(self.model.norm(h))
        
        # Average state velocity
        avg_velocity = mx.mean(mx.stack(state_velocities))
        
        return logits, mean_depth, avg_velocity
    
    def train_step(self, x, y, step, total_steps, optimizer):
        """Single training step with all techniques."""
        
        # Forward with caching
        logits, mean_depth, avg_velocity = self.forward_with_cache(x)
        
        # Check for skip (after warmup)
        if step > self.warmup_steps:
            probs = mx.softmax(logits, axis=-1)
            correct_probs = mx.take_along_axis(probs, y[:, :, None], axis=-1)
            confidence = float(mx.mean(correct_probs).item())
            
            if confidence > self.skip_threshold:
                return None, True  # Skip this sample
        
        # Compute loss
        base_loss = nn.losses.cross_entropy(logits, y, reduction='mean')
        
        # Weight by state momentum (higher velocity = train harder)
        momentum_weight = 1.0 + float(avg_velocity.item()) * 0.5
        
        # Weight by depth (deeper = harder = train harder)
        depth_weight = 1.0 + (float(mean_depth.item()) / self.model.num_layers) * 0.5
        
        # Combined weight
        total_weight = momentum_weight * depth_weight
        weighted_loss = base_loss * total_weight
        
        return weighted_loss, False
    
    def train(self, data, steps=200):
        """Full training with all techniques."""
        
        optimizer = optim.AdamW(learning_rate=3e-4)
        
        start = time.time()
        total_loss = 0
        actual_steps = 0
        skip_count = 0
        
        for step in range(steps):
            # Curriculum: Start short, grow longer
            progress = step / steps
            seq_len = int(16 + progress * 48)  # 16 → 64
            
            # Sample batch
            starts = mx.random.randint(0, len(data) - seq_len - 1, (16,))
            x = mx.stack([data[int(s):int(s)+seq_len] for s in starts.tolist()])
            y = mx.stack([data[int(s)+1:int(s)+seq_len+1] for s in starts.tolist()])
            
            # Custom forward to get cached values
            def loss_fn(m):
                logits, depth, velocity = self.forward_with_cache(x)
                base_loss = nn.losses.cross_entropy(logits, y, reduction='mean')
                
                # Apply weights from cached values
                depth_w = 1.0 + (float(depth.item()) / m.num_layers) * 0.3
                vel_w = 1.0 + float(velocity.item()) * 0.3
                
                return base_loss * depth_w * vel_w
            
            # Check for skip (quick check before full backward)
            if step > self.warmup_steps:
                logits, _, _ = self.forward_with_cache(x)
                probs = mx.softmax(logits, axis=-1)
                correct_probs = mx.take_along_axis(probs, y[:, :, None], axis=-1)
                confidence = float(mx.mean(correct_probs).item())
                
                if confidence > self.skip_threshold:
                    skip_count += 1
                    continue
            
            # Backward and update
            loss, grads = mx.value_and_grad(loss_fn)(self.model)
            optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), optimizer.state)
            
            total_loss += float(loss)
            actual_steps += 1
            
            if (step + 1) % 50 == 0:
                avg_loss = total_loss / max(1, actual_steps)
                skip_rate = skip_count / max(1, step - self.warmup_steps) * 100 if step > self.warmup_steps else 0
                print(f"  Step {step+1}: Loss={avg_loss:.4f} | SeqLen={seq_len} | Skip={skip_rate:.0f}%")
        
        return time.time() - start, total_loss / max(1, actual_steps)


def benchmark():
    """Compare baseline vs combined."""
    
    print("=" * 60)
    print("COMBINED TRAINING - ALL TECHNIQUES")
    print("=" * 60)
    
    # Prepare data
    datasets = get_all_datasets()
    qa_data = datasets['math'][:200]
    data_str = "".join([(q + a + "\n") * 50 for q, a in qa_data])
    data = mx.array([ord(c) for c in data_str], dtype=mx.int32)
    print(f"Data: {len(data)} bytes\n")
    
    # Baseline
    print("--- BASELINE ---")
    model = GhostV6(dim=256, num_layers=6)
    mx.eval(model.parameters())
    
    optimizer = optim.AdamW(learning_rate=3e-4)
    start = time.time()
    for step in range(200):
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
    
    baseline_time = time.time() - start
    baseline_loss = float(loss)
    print(f"Time: {baseline_time:.1f}s | Loss: {baseline_loss:.4f}\n")
    
    # Combined
    print("--- COMBINED (ALL 4 TECHNIQUES) ---")
    model = GhostV6(dim=256, num_layers=6)
    mx.eval(model.parameters())
    
    trainer = CombinedTrainer(model)
    combined_time, combined_loss = trainer.train(data, steps=200)
    print(f"Time: {combined_time:.1f}s | Loss: {combined_loss:.4f}\n")
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    speedup = baseline_time / combined_time
    loss_diff = baseline_loss - combined_loss
    print(f"Baseline:  {baseline_time:.1f}s | Loss: {baseline_loss:.4f}")
    print(f"Combined:  {combined_time:.1f}s | Loss: {combined_loss:.4f}")
    print(f"Speedup:   {speedup:.2f}x")
    print(f"Loss diff: {loss_diff:+.4f}")
    print("=" * 60)
    
    if speedup > 1.0 and combined_loss <= baseline_loss:
        print("🏆 Combined is BETTER in both speed and loss!")
    elif speedup > 1.0:
        print("⚡ Combined is FASTER")
    elif combined_loss < baseline_loss:
        print("📈 Combined has BETTER loss")


if __name__ == "__main__":
    benchmark()
