"""
Baseline Traditional - Honest Baseline
=======================================
Standard single-model SGD training for comparison.
No workers, no fancy techniques - just vectorized batch training.

Purpose: Establish baseline performance for comparison.
"""

import sys
import os
import time
import json
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

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

# Import dream data generator
from speed_demon.train import generate_dream_data


# ============================================================================
# BASELINE TRAINER
# ============================================================================

class BaselineTrainer:
    """
    Traditional single-model trainer.
    Vectorized batch processing but no parallel workers or fancy techniques.
    """
    
    def __init__(self, learning_rate: float = 0.001):
        print("Initializing Baseline Traditional Trainer...")
        self.model = GhostWorker(dim=256, num_layers=6)
        mx.eval(self.model.parameters())
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.history = []
    
    def pad_batch(self, batch: List[str]) -> Tuple[mx.array, mx.array]:
        """Tokenize and pad a batch of strings."""
        tokenized = [[ord(c) for c in text] for text in batch]
        max_len = max(len(t) for t in tokenized)
        padded = []
        masks = []
        for t in tokenized:
            pad_len = max_len - len(t)
            padded.append(t + [0] * pad_len)
            masks.append([1.0] * len(t) + [0.0] * pad_len)
        return mx.array(padded, dtype=mx.int32), mx.array(masks, dtype=mx.float32)
    
    def compute_loss(self, model, batch: List[str]) -> mx.array:
        """Compute vectorized loss."""
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
    
    def train_step(self, data: List[Tuple[str, str]]) -> float:
        """Single training step."""
        batch = [q + a for q, a in data]
        
        loss_fn = lambda m: self.compute_loss(m, batch)
        loss, grads = mx.value_and_grad(loss_fn)(self.model)
        
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters())
        
        return loss.item()
    
    def train(self, num_steps: int = 100, samples_per_step: int = 100, verbose: bool = False) -> dict:
        """Train using traditional approach."""
        print("=" * 60)
        print("BASELINE TRADITIONAL TRAINING")
        print(f"Steps: {num_steps}")
        print(f"Samples/step: {samples_per_step}")
        print("=" * 60)
        
        total_start = time.time()
        
        for step in range(num_steps):
            step_start = time.time()
            
            # Generate data
            data = generate_dream_data(samples_per_step)
            
            # Train
            loss = self.train_step(data)
            
            step_time = time.time() - step_start
            speed = samples_per_step / step_time if step_time > 0 else 0
            
            stats = {
                'step': step + 1,
                'loss': loss,
                'time': step_time,
                'samples_per_sec': speed
            }
            self.history.append(stats)
            
            if verbose or (step + 1) % 10 == 0:
                print(f"Step {step+1}/{num_steps} | Loss: {loss:.4f} | Speed: {speed:.1f} s/s")
        
        total_time = time.time() - total_start
        total_samples = num_steps * samples_per_step
        
        results = {
            'experiment': 'baseline_traditional',
            'total_steps': num_steps,
            'total_samples': total_samples,
            'total_time': total_time,
            'avg_samples_per_sec': total_samples / total_time if total_time > 0 else 0,
            'final_loss': self.history[-1]['loss'] if self.history else 0,
            'history': self.history
        }
        
        print("\n" + "=" * 60)
        print("BASELINE RESULTS")
        print("=" * 60)
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg speed: {results['avg_samples_per_sec']:.1f} samples/sec")
        print(f"Final loss: {results['final_loss']:.4f}")
        
        return results
    
    def save_results(self, path: str):
        """Save results to JSON."""
        results = {
            'experiment': 'baseline_traditional',
            'final_loss': self.history[-1]['loss'] if self.history else 0,
            'history': self.history
        }
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Traditional Training")
    parser.add_argument('--steps', type=int, default=50, help='Training steps')
    parser.add_argument('--samples', type=int, default=100, help='Samples per step')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    trainer = BaselineTrainer(learning_rate=args.lr)
    results = trainer.train(num_steps=args.steps, samples_per_step=args.samples, verbose=args.verbose)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    trainer.save_results(os.path.join(output_dir, 'baseline_results.json'))
