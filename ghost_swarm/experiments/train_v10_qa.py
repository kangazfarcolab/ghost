"""
Train Ghost v10 with SwarmMomentum on Q&A Dataset
==================================================
Tests all 10 features with real Q&A training.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ghost_model_v10.core.ghost_worker_v10 import GhostWorkerV10


# ============================================================================
# Q&A DATASET
# ============================================================================

QA_PAIRS = [
    # DevOps
    ("How to list pods?", "kubectl get pods"),
    ("How to list all namespaces?", "kubectl get namespaces"),
    ("How to describe a pod?", "kubectl describe pod <name>"),
    ("How to get logs?", "kubectl logs <pod>"),
    ("How to delete pod?", "kubectl delete pod <name>"),
    # AWS
    ("How to list S3 buckets?", "aws s3 ls"),
    ("How to list EC2 instances?", "aws ec2 describe-instances"),
    ("How to copy to S3?", "aws s3 cp <file> s3://<bucket>/"),
    # Docker
    ("How to list containers?", "docker ps"),
    ("How to run nginx?", "docker run -d nginx"),
    ("How to build image?", "docker build -t <name> ."),
    # Git
    ("How to clone repo?", "git clone <url>"),
    ("How to push changes?", "git push origin main"),
    ("How to create branch?", "git checkout -b <branch>"),
]


def encode(text, max_len=64):
    """Encode text to bytes."""
    b = [ord(c) for c in text]
    b = (b + [0] * max_len)[:max_len]
    return b


def make_batch(qa_pairs, batch_size=8):
    """Create Q&A batch."""
    import random
    pairs = random.sample(qa_pairs, min(batch_size, len(qa_pairs)))
    
    inputs = []
    targets = []
    for q, a in pairs:
        # Format: Q: question\nA: answer
        full = f"Q: {q}\nA: {a}"
        enc = encode(full, max_len=64)
        inputs.append(enc)
        # Target is shifted by 1
        targets.append(enc[1:] + [0])
    
    return mx.array(inputs, dtype=mx.int32), mx.array(targets, dtype=mx.int32)


# ============================================================================
# SWARM MOMENTUM TRAINER
# ============================================================================

class SwarmMomentumTrainer:
    """SwarmMomentum training for v10."""
    
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.base_lr = lr
        self.optimizer = optim.Adam(learning_rate=lr)
        self.history = []
    
    def compute_consensus(self, grads):
        """Compute gradient consensus score."""
        flat = []
        for _, g in tree_flatten(grads):
            flat.append(g.reshape(-1))
        all_grads = mx.concatenate(flat)
        
        # Variance-based consensus
        var = mx.var(all_grads)
        consensus = 1.0 / (1.0 + float(var.item()))
        return consensus
    
    def train_step(self, x, targets):
        """Single training step with consensus-based LR."""
        
        def loss_fn(model):
            logits = model(x, use_memory=False, use_routing=True)
            loss = mx.mean(nn.losses.cross_entropy(
                logits[:, :-1].reshape(-1, 256),
                targets[:, :-1].reshape(-1)
            ))
            return loss
        
        loss, grads = mx.value_and_grad(loss_fn)(self.model)
        
        # Compute consensus
        consensus = self.compute_consensus(grads)
        
        # Scale LR by consensus (high agreement = bigger step)
        lr_mult = 0.5 + consensus
        
        # Scale gradients
        scaled_grads = tree_flatten(grads)
        scaled_grads = [(k, g * lr_mult) for k, g in scaled_grads]
        
        # Reconstruct gradient tree
        def reconstruct(template, flat_items):
            flat_iter = iter(flat_items)
            def _reconstruct(t):
                if isinstance(t, dict):
                    return {k: _reconstruct(v) for k, v in t.items()}
                elif isinstance(t, list):
                    return [_reconstruct(v) for v in t]
                else:
                    _, val = next(flat_iter)
                    return val
            return _reconstruct(template)
        
        scaled = reconstruct(grads, scaled_grads)
        self.optimizer.update(self.model, scaled)
        mx.eval(self.model.parameters())
        
        return float(loss.item()), consensus
    
    def train(self, epochs=10, batch_size=8):
        """Full training loop."""
        print(f"\nTraining v10 with SwarmMomentum")
        print(f"Features: {self.model.FEATURES}")
        print("=" * 50)
        
        start = time.time()
        
        for epoch in range(epochs):
            x, targets = make_batch(QA_PAIRS, batch_size)
            loss, consensus = self.train_step(x, targets)
            
            self.history.append({
                'epoch': epoch + 1,
                'loss': loss,
                'consensus': consensus
            })
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}: loss={loss:.4f}, consensus={consensus:.3f}")
        
        elapsed = time.time() - start
        
        # Stats
        stats = self.model.get_stats()
        
        print(f"\n{'='*50}")
        print(f"Training complete in {elapsed:.2f}s")
        print(f"Final loss: {self.history[-1]['loss']:.4f}")
        print(f"Compute ratio: {stats['compute_ratio']*100:.1f}%")
        print(f"  - Depth skip: {stats.get('depth_skip_ratio', 0)*100:.1f}%")
        print(f"  - MoD skip: {stats.get('mod_skip_ratio', 0)*100:.1f}%")
        
        return self.history


# ============================================================================
# TEST GENERATION
# ============================================================================

def test_generation(model, prompt):
    """Test model generation."""
    enc = encode(f"Q: {prompt}\nA: ", max_len=32)
    x = mx.array([enc], dtype=mx.int32)
    
    generated = []
    for _ in range(30):
        logits = model(x, use_memory=True, use_routing=True)
        next_token = int(mx.argmax(logits[0, -1]).item())
        if next_token == 0 or next_token == ord('\n'):
            break
        generated.append(chr(next_token))
        # Append to input
        enc = enc[1:] + [next_token]
        x = mx.array([enc], dtype=mx.int32)
    
    return ''.join(generated)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Ghost v10 + SwarmMomentum Training")
    print("=" * 50)
    
    # Create model
    model = GhostWorkerV10(dim=256, num_layers=6)
    mx.eval(model.parameters())
    
    mem = model.estimate_memory()
    print(f"Parameters: {mem['params']:,}")
    print(f"Binary size: {mem['binary_mb']:.2f} MB")
    print(f"Features: {model.FEATURES}")
    
    # Train
    trainer = SwarmMomentumTrainer(model, lr=1e-3)
    history = trainer.train(epochs=50, batch_size=8)
    
    # Test generation
    print("\n" + "=" * 50)
    print("Testing generation:")
    
    test_prompts = [
        "How to list pods?",
        "How to list containers?",
        "How to clone repo?"
    ]
    
    for prompt in test_prompts:
        result = test_generation(model, prompt)
        print(f"Q: {prompt}")
        print(f"A: {result}")
        print()
    
    print("✅ Training complete!")
