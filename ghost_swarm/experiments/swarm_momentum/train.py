"""
Swarm Momentum - Novel Training Approach
=========================================
Adaptive learning rate based on gradient consensus across parallel workers.

Key Innovation:
- Multiple workers compute gradients on different data subsets
- Measure consensus (agreement) between gradients
- HIGH consensus = confident update = BIGGER step
- LOW consensus = uncertain = SMALLER step

Benefits:
- Faster convergence (adaptive LR)
- Better for small models (uses variance as signal)
- Self-regulating (no LR tuning needed)
"""

import sys
import os
import time
import json
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
experiments_dir = os.path.dirname(script_dir)
swarm_dir = os.path.dirname(experiments_dir)
base_path = os.path.dirname(swarm_dir)

sys.path.insert(0, base_path)
sys.path.insert(0, experiments_dir)

# Load GhostWorker
import importlib.util
ghost_worker_path = os.path.join(base_path, 'ghost_model_v7', 'core', 'ghost_worker.py')
spec = importlib.util.spec_from_file_location("ghost_worker", ghost_worker_path)
ghost_worker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ghost_worker_module)
GhostWorker = ghost_worker_module.GhostWorker

from speed_demon.train import generate_dream_data


# ============================================================================
# SWARM MOMENTUM TRAINER
# ============================================================================

class SwarmMomentumTrainer:
    """
    Swarm Momentum: Adaptive LR from Gradient Consensus
    
    Novel features:
    1. Parallel gradient computation (speed)
    2. Consensus-based adaptive LR (smarter)
    3. Single model, no population (lighter)
    """
    
    def __init__(self, num_workers: int = 8, base_lr: float = 0.001, boost_factor: float = 2.0):
        self.num_workers = num_workers
        self.base_lr = base_lr
        self.boost_factor = boost_factor
        
        print("=" * 60)
        print("SWARM MOMENTUM TRAINER")
        print("=" * 60)
        print(f"Workers: {num_workers}")
        print(f"Base LR: {base_lr}")
        print(f"Boost Factor: {boost_factor}")
        print("Novel: Adaptive LR from gradient consensus")
        print("=" * 60)
        
        # Single model (light)
        self.model = GhostWorker(dim=256, num_layers=6)
        mx.eval(self.model.parameters())
        
        # Use Adam optimizer (better convergence than manual SGD)
        self.optimizer = optim.Adam(learning_rate=base_lr)
        
        self.history = []
    
    def pad_batch(self, batch: List[str]) -> Tuple[mx.array, mx.array]:
        """Pad batch to equal length."""
        tokenized = [[ord(c) for c in text] for text in batch]
        if not tokenized:
            return mx.array([[0]], dtype=mx.int32), mx.array([[1.0]], dtype=mx.float32)
        max_len = max(len(t) for t in tokenized)
        if max_len == 0:
            max_len = 1
        padded = []
        masks = []
        for t in tokenized:
            pad_len = max_len - len(t)
            padded.append(t + [0] * pad_len if t else [0] * max_len)
            masks.append([1.0] * len(t) + [0.0] * pad_len if t else [0.0] * max_len)
        return mx.array(padded, dtype=mx.int32), mx.array(masks, dtype=mx.float32)
    
    def compute_loss(self, model, batch: List[str]) -> mx.array:
        """Compute loss for batch."""
        x, mask = self.pad_batch(batch)
        B, L = x.shape
        if L < 2:
            return mx.array(0.0)
        
        logits = model(x[:, :-1])
        targets = x[:, 1:]
        target_mask = mask[:, 1:]
        
        logits_flat = logits.reshape(-1, 256)
        targets_flat = targets.reshape(-1)
        
        losses = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='none')
        masked_loss = losses * target_mask.reshape(-1)
        
        return masked_loss.sum() / mx.maximum(target_mask.sum(), 1.0)
    
    def flatten_grads(self, grads) -> mx.array:
        """Flatten gradient dict to single vector."""
        flat = []
        for _, v in tree_flatten(grads):
            flat.append(v.reshape(-1))
        return mx.concatenate(flat)
    
    def compute_consensus(self, gradients: List) -> float:
        """
        Compute consensus score (0 to 1) based on gradient agreement.
        Higher = more agreement = more confident.
        """
        if len(gradients) < 2:
            return 0.5
        
        # Flatten all gradients
        flat_grads = [self.flatten_grads(g) for g in gradients]
        
        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(flat_grads)):
            for j in range(i + 1, len(flat_grads)):
                g1, g2 = flat_grads[i], flat_grads[j]
                
                # Cosine similarity
                dot = mx.sum(g1 * g2)
                norm1 = mx.sqrt(mx.sum(g1 * g1) + 1e-8)
                norm2 = mx.sqrt(mx.sum(g2 * g2) + 1e-8)
                sim = dot / (norm1 * norm2)
                similarities.append(sim)
        
        # Average similarity (will be between -1 and 1)
        avg_sim = sum(similarities) / len(similarities)
        
        # Convert to 0-1 range: (sim + 1) / 2
        consensus = (float(mx.array(avg_sim).item()) + 1) / 2
        
        return consensus
    
    def swarm_step(self, data: List[Tuple[str, str]]) -> Tuple[float, float, float]:
        """
        One Swarm Momentum training step.
        Returns: (loss, consensus, effective_lr)
        """
        step_start = time.time()
        
        # 1. Split data across workers
        worker_chunks = [[] for _ in range(self.num_workers)]
        for i, item in enumerate(data):
            worker_chunks[i % self.num_workers].append(item)
        
        # 2. Compute gradients for each worker
        all_gradients = []
        all_losses = []
        
        for chunk in worker_chunks:
            if not chunk:
                continue
            
            batch = [q + a for q, a in chunk]
            
            loss_fn = lambda m: self.compute_loss(m, batch)
            loss, grads = mx.value_and_grad(loss_fn)(self.model)
            
            all_gradients.append(grads)
            all_losses.append(loss.item())
        
        if not all_gradients:
            return 0.0, 0.5, self.base_lr
        
        # 3. Compute consensus
        consensus = self.compute_consensus(all_gradients)
        
        # 4. Adaptive learning rate
        # High consensus (0.8+) = boost LR up to 3x
        # Low consensus (0.2-) = reduce LR to 0.5x
        lr_multiplier = 0.5 + consensus * self.boost_factor  # 0.5 to 2.5 range
        effective_lr = self.base_lr * lr_multiplier
        
        # 5. Average gradients and SCALE by consensus
        # Higher consensus = trust gradient more = scale up
        avg_grads = tree_map(lambda *gs: sum(gs) / len(gs), *all_gradients)
        scaled_grads = tree_map(lambda g: g * lr_multiplier, avg_grads)
        
        # 6. Update with Adam (better than manual SGD)
        self.optimizer.update(self.model, scaled_grads)
        mx.eval(self.model.parameters())
        
        avg_loss = sum(all_losses) / len(all_losses)
        step_time = time.time() - step_start
        
        return avg_loss, consensus, effective_lr
    
    def train(self, num_steps: int = 50, samples_per_step: int = 100, verbose: bool = False) -> dict:
        """Train with Swarm Momentum."""
        print(f"\nTraining for {num_steps} steps, {samples_per_step} samples/step")
        print("-" * 60)
        
        total_start = time.time()
        
        for step in range(num_steps):
            # Generate data
            data = generate_dream_data(samples_per_step)
            
            # Swarm step
            loss, consensus, eff_lr = self.swarm_step(data)
            
            stats = {
                'step': step + 1,
                'loss': loss,
                'consensus': consensus,
                'effective_lr': eff_lr,
                'lr_multiplier': eff_lr / self.base_lr
            }
            self.history.append(stats)
            
            if verbose or (step + 1) % 10 == 0:
                print(f"Step {step+1:3d}/{num_steps} | "
                      f"Loss: {loss:.4f} | "
                      f"Consensus: {consensus:.2f} | "
                      f"LR: {eff_lr:.5f} ({stats['lr_multiplier']:.1f}x)")
        
        total_time = time.time() - total_start
        total_samples = num_steps * samples_per_step
        
        avg_consensus = sum(s['consensus'] for s in self.history) / len(self.history)
        
        results = {
            'experiment': 'swarm_momentum',
            'num_workers': self.num_workers,
            'base_lr': self.base_lr,
            'boost_factor': self.boost_factor,
            'total_steps': num_steps,
            'total_samples': total_samples,
            'total_time': total_time,
            'avg_samples_per_sec': total_samples / total_time if total_time > 0 else 0,
            'final_loss': self.history[-1]['loss'] if self.history else 0,
            'avg_consensus': avg_consensus,
            'history': self.history
        }
        
        print("\n" + "=" * 60)
        print("SWARM MOMENTUM RESULTS")
        print("=" * 60)
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg speed: {results['avg_samples_per_sec']:.1f} samples/sec")
        print(f"Final loss: {results['final_loss']:.4f}")
        print(f"Avg consensus: {avg_consensus:.2f}")
        
        return results
    
    def save_results(self, path: str):
        """Save results."""
        results = {
            'experiment': 'swarm_momentum',
            'final_loss': self.history[-1]['loss'] if self.history else 0,
            'avg_consensus': sum(s['consensus'] for s in self.history) / len(self.history) if self.history else 0,
            'history': self.history
        }
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Swarm Momentum Training")
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--steps', type=int, default=30, help='Training steps')
    parser.add_argument('--samples', type=int, default=80, help='Samples per step')
    parser.add_argument('--lr', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('--boost', type=float, default=2.0, help='LR boost factor')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose')
    args = parser.parse_args()
    
    trainer = SwarmMomentumTrainer(
        num_workers=args.workers,
        base_lr=args.lr,
        boost_factor=args.boost
    )
    
    results = trainer.train(
        num_steps=args.steps,
        samples_per_step=args.samples,
        verbose=args.verbose
    )
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    trainer.save_results(os.path.join(output_dir, 'swarm_momentum_results.json'))
