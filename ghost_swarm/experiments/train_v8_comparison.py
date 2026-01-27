"""
Train v8 vs v7 with SwarmMomentum
=================================
Compare Binary Mamba + Adaptive Depth (v8) against standard (v7).
"""

import sys
import os
import time

# Add base path for imports
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)
ex_path = os.path.join(base_path)
sys.path.insert(0, ex_path)

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten


# ============================================================================
# DATA GENERATION (shared)
# ============================================================================

def generate_dream_data(num_samples=32, seq_len=64):
    """Generate synthetic training data."""
    data = []
    patterns = [
        "kubectl get pods -n {}",
        "aws s3 ls s3://{}",
        "docker run -d {}",
        "git commit -m '{}'",
    ]
    namespaces = ["default", "kube-system", "production", "staging"]
    
    for _ in range(num_samples):
        pattern = patterns[_ % len(patterns)]
        ns = namespaces[_ % len(namespaces)]
        text = pattern.format(ns)
        
        # Pad/truncate to seq_len
        bytes_list = [ord(c) for c in text]
        if len(bytes_list) < seq_len:
            bytes_list.extend([0] * (seq_len - len(bytes_list)))
        else:
            bytes_list = bytes_list[:seq_len]
        
        input_bytes = bytes_list[:-1]
        target_bytes = bytes_list[1:]
        data.append((input_bytes, target_bytes))
    
    return data


def pad_batch(batch):
    """Vectorized batch padding."""
    inputs = mx.array([x[0] for x in batch], dtype=mx.int32)
    targets = mx.array([x[1] for x in batch], dtype=mx.int32)
    return inputs, targets


# ============================================================================
# SWARM MOMENTUM TRAINER (adapted for any model)
# ============================================================================

class SwarmMomentumTrainer:
    """SwarmMomentum training for any ghost model."""
    
    def __init__(self, model, num_workers=4, base_lr=1e-3):
        self.model = model
        self.num_workers = num_workers
        self.base_lr = base_lr
        self.optimizer = optim.Adam(learning_rate=base_lr)
    
    def compute_gradient(self, inputs, targets):
        """Compute gradient and loss."""
        def loss_fn(model):
            logits = model(inputs)
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, 256),
                targets.reshape(-1)
            )
            return mx.mean(loss)
        
        loss, grads = mx.value_and_grad(loss_fn)(self.model)
        return grads, float(loss.item())
    
    def compute_consensus(self, gradients):
        """Compute gradient consensus score."""
        if len(gradients) < 2:
            return 1.0
        
        # Flatten all gradients
        flat_grads = []
        for grad_tree in gradients:
            flat = []
            for _, g in tree_flatten(grad_tree):
                flat.append(g.reshape(-1))
            flat_grads.append(mx.concatenate(flat))
        
        # Compute pairwise cosine similarity
        similarities = []
        for i in range(len(flat_grads)):
            for j in range(i + 1, len(flat_grads)):
                g1, g2 = flat_grads[i], flat_grads[j]
                cos_sim = mx.sum(g1 * g2) / (mx.sqrt(mx.sum(g1**2)) * mx.sqrt(mx.sum(g2**2)) + 1e-8)
                similarities.append(float(cos_sim.item()))
        
        return sum(similarities) / len(similarities) if similarities else 1.0
    
    def train_step(self, data):
        """One SwarmMomentum training step."""
        # Compute gradients on full batch
        inputs, targets = pad_batch(data)
        
        def loss_fn(model):
            logits = model(inputs)
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, 256),
                targets.reshape(-1)
            )
            return mx.mean(loss)
        
        loss, grads = mx.value_and_grad(loss_fn)(self.model)
        
        # Compute "pseudo-consensus" from gradient magnitude variance
        flat = []
        for _, g in tree_flatten(grads):
            flat.append(g.reshape(-1))
        flat_grad = mx.concatenate(flat)
        
        # Higher gradient norm variance = lower consensus
        grad_norm = mx.sqrt(mx.sum(flat_grad**2))
        consensus = min(1.0, 1.0 / (1.0 + float(mx.std(flat_grad).item()) * 10))
        
        # Adaptive LR based on consensus
        lr_multiplier = 0.5 + consensus  # Range: 0.5 to 1.5
        
        # Scale gradients by multiplier
        def scale_grad(g):
            return g * lr_multiplier
        
        from mlx.utils import tree_map
        scaled_grads = tree_map(scale_grad, grads)
        
        # Apply update
        self.optimizer.update(self.model, scaled_grads)
        mx.eval(self.model.parameters())
        
        return {
            'loss': float(loss.item()),
            'consensus': consensus,
            'lr_mult': lr_multiplier
        }


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def train_model(model, name, steps=5, samples_per_step=32):
    """Train a model and return stats."""
    print(f"\nTraining {name}...")
    print("-" * 40)
    
    trainer = SwarmMomentumTrainer(model, num_workers=4)
    
    start_time = time.time()
    losses = []
    
    for step in range(steps):
        data = generate_dream_data(num_samples=samples_per_step)
        stats = trainer.train_step(data)
        losses.append(stats['loss'])
        print(f"  Step {step+1}: loss={stats['loss']:.4f}, consensus={stats['consensus']:.2f}")
    
    elapsed = time.time() - start_time
    
    return {
        'name': name,
        'final_loss': losses[-1],
        'avg_loss': sum(losses) / len(losses),
        'time': elapsed,
        'samples_per_sec': (steps * samples_per_step) / elapsed
    }


def main():
    print("=" * 60)
    print("SwarmMomentum Training: v7 vs v8")
    print("=" * 60)
    
    # Import models
    from ghost_model_v7.core.ghost_worker import GhostWorker as GhostWorkerV7
    from ghost_model_v8.core.ghost_worker_v8 import GhostWorkerV8
    
    # Create models
    print("\nInitializing models...")
    v7 = GhostWorkerV7(dim=256, num_layers=6)
    v8 = GhostWorkerV8(dim=256, num_layers=6)
    
    mx.eval(v7.parameters())
    mx.eval(v8.parameters())
    
    v7_params = v7.count_params()
    v8_mem = v8.estimate_memory()
    
    print(f"  v7: {v7_params:,} params ({v7_params*4/1024/1024:.1f} MB)")
    print(f"  v8: {v8_mem['params']:,} params ({v8_mem['binary_mb']:.1f} MB binary)")
    
    # Train both
    steps = 5
    samples = 32
    
    v7_stats = train_model(v7, "Ghost v7", steps, samples)
    v8_stats = train_model(v8, "Ghost v8 (Binary+Adaptive)", steps, samples)
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Metric':<20} {'v7':>15} {'v8':>15} {'Winner':>10}")
    print("-" * 60)
    
    # Time
    print(f"{'Training Time':<20} {v7_stats['time']:>14.2f}s {v8_stats['time']:>14.2f}s", end="")
    print(f" {'v8 ✓' if v8_stats['time'] < v7_stats['time'] else 'v7 ✓':>10}")
    
    # Speed
    print(f"{'Samples/sec':<20} {v7_stats['samples_per_sec']:>15.1f} {v8_stats['samples_per_sec']:>15.1f}", end="")
    print(f" {'v8 ✓' if v8_stats['samples_per_sec'] > v7_stats['samples_per_sec'] else 'v7 ✓':>10}")
    
    # Loss
    print(f"{'Final Loss':<20} {v7_stats['final_loss']:>15.4f} {v8_stats['final_loss']:>15.4f}", end="")
    print(f" {'v8 ✓' if v8_stats['final_loss'] < v7_stats['final_loss'] else 'v7 ✓':>10}")
    
    # Memory
    print(f"{'Memory':<20} {v7_params*4/1024/1024:>14.1f}MB {v8_mem['binary_mb']:>14.1f}MB {'v8 ✓':>10}")
    
    print("\n" + "=" * 60)
    speedup = v8_stats['samples_per_sec'] / v7_stats['samples_per_sec']
    compression = (v7_params * 4) / (v8_mem['params'] * 0.2)
    print(f"v8 is {speedup:.1f}x speed, {compression:.0f}x smaller memory")
    print("=" * 60)


if __name__ == "__main__":
    main()
