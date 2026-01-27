"""
Speed Vote - Speed Demon + Light Voting
========================================
Parallel gradient farming with occasional gradient voting.
Voting only every N steps instead of every step to maintain speed.

Goal: ~90% speed of Speed Demon with better stability
"""

import sys
import os
import time
import json
from typing import List, Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

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

# Import from other experiments
from speed_demon.train import generate_dream_data
from precision_master.train import gradient_consensus_filter, weighted_gradient_average, compute_gradient_confidence


# ============================================================================
# SPEED VOTE TRAINER
# ============================================================================

class SpeedVoteTrainer:
    """
    Speed Demon + Light Voting
    
    - Parallel gradient farming (speed)
    - Gradient voting only every N steps (stability without overhead)
    """
    
    def __init__(self, num_workers: int = 10, vote_interval: int = 10):
        self.num_workers = num_workers
        self.vote_interval = vote_interval
        
        print(f"Initializing Speed Vote Trainer with {num_workers} workers...")
        print(f"Voting interval: every {vote_interval} steps")
        
        self.model = GhostWorker(dim=256, num_layers=6)
        mx.eval(self.model.parameters())
        self.optimizer = optim.Adam(learning_rate=0.001)
        
        self.step_count = 0
        self.history = []
    
    def pad_batch(self, batch: List[str]) -> Tuple[mx.array, mx.array]:
        """Tokenize and pad a batch."""
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
    
    def parallel_step(self, data: List[Tuple[str, str]], use_voting: bool = False) -> Dict:
        """Perform parallel gradient step with optional voting."""
        step_start = time.time()
        
        # Distribute data round-robin
        worker_chunks = [[] for _ in range(self.num_workers)]
        for i, item in enumerate(data):
            worker_chunks[i % self.num_workers].append(item)
        
        all_gradients = []
        all_losses = []
        all_confidences = []
        
        for i, chunk in enumerate(worker_chunks):
            if not chunk:
                continue
            
            batch = [q + a for q, a in chunk]
            
            loss_fn = lambda m: self.compute_loss(m, batch)
            loss, grads = mx.value_and_grad(loss_fn)(self.model)
            
            loss_val = loss.item()
            all_gradients.append(grads)
            all_losses.append(loss_val)
            all_confidences.append(compute_gradient_confidence(None, loss_val))
        
        filtered_count = 0
        
        if use_voting and len(all_gradients) > 1:
            # Apply gradient voting
            original_count = len(all_gradients)
            filtered_gradients = gradient_consensus_filter(all_gradients, threshold=1.5)
            filtered_count = original_count - len(filtered_gradients)
            
            if filtered_gradients:
                filtered_conf = all_confidences[:len(filtered_gradients)]
                avg_grads = weighted_gradient_average(filtered_gradients, filtered_conf)
            else:
                avg_grads = tree_map(lambda *gs: sum(gs) / len(gs), *all_gradients)
        else:
            # Simple average (fast path)
            avg_grads = tree_map(lambda *gs: sum(gs) / len(gs), *all_gradients)
        
        # Update model
        self.optimizer.update(self.model, avg_grads)
        mx.eval(self.model.parameters())
        
        step_time = time.time() - step_start
        avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0
        
        return {
            'avg_loss': avg_loss,
            'time': step_time,
            'filtered': filtered_count,
            'used_voting': use_voting
        }
    
    def train(self, num_steps: int = 100, samples_per_step: int = 100, verbose: bool = False) -> dict:
        """Train with speed + occasional voting."""
        print("=" * 60)
        print("SPEED VOTE TRAINING")
        print(f"Workers: {self.num_workers}")
        print(f"Steps: {num_steps}")
        print(f"Samples/step: {samples_per_step}")
        print(f"Vote every: {self.vote_interval} steps")
        print("=" * 60)
        
        total_start = time.time()
        total_filtered = 0
        
        for step in range(num_steps):
            self.step_count += 1
            
            # Generate data
            data = generate_dream_data(samples_per_step)
            
            # Decide if voting this step
            use_voting = (self.step_count % self.vote_interval == 0)
            
            # Train step
            step_stats = self.parallel_step(data, use_voting=use_voting)
            total_filtered += step_stats['filtered']
            
            speed = samples_per_step / step_stats['time'] if step_stats['time'] > 0 else 0
            
            stats = {
                'step': self.step_count,
                'loss': step_stats['avg_loss'],
                'time': step_stats['time'],
                'samples_per_sec': speed,
                'filtered': step_stats['filtered'],
                'used_voting': step_stats['used_voting']
            }
            self.history.append(stats)
            
            if verbose or (step + 1) % 10 == 0:
                vote_str = " [VOTE]" if step_stats['used_voting'] else ""
                print(f"Step {step+1}/{num_steps} | Loss: {step_stats['avg_loss']:.4f} | "
                      f"Speed: {speed:.1f} s/s{vote_str}")
        
        total_time = time.time() - total_start
        total_samples = num_steps * samples_per_step
        
        results = {
            'experiment': 'speed_vote',
            'num_workers': self.num_workers,
            'vote_interval': self.vote_interval,
            'total_steps': num_steps,
            'total_samples': total_samples,
            'total_time': total_time,
            'avg_samples_per_sec': total_samples / total_time if total_time > 0 else 0,
            'total_filtered': total_filtered,
            'final_loss': self.history[-1]['loss'] if self.history else 0,
            'history': self.history
        }
        
        print("\n" + "=" * 60)
        print("SPEED VOTE RESULTS")
        print("=" * 60)
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg speed: {results['avg_samples_per_sec']:.1f} samples/sec")
        print(f"Total filtered: {total_filtered}")
        print(f"Final loss: {results['final_loss']:.4f}")
        
        return results
    
    def save_results(self, path: str):
        """Save results to JSON."""
        results = {
            'experiment': 'speed_vote',
            'final_loss': self.history[-1]['loss'] if self.history else 0,
            'history': self.history
        }
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Speed Vote Training")
    parser.add_argument('--workers', type=int, default=10, help='Number of workers')
    parser.add_argument('--steps', type=int, default=50, help='Training steps')
    parser.add_argument('--samples', type=int, default=100, help='Samples per step')
    parser.add_argument('--vote-interval', type=int, default=10, help='Vote every N steps')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    trainer = SpeedVoteTrainer(num_workers=args.workers, vote_interval=args.vote_interval)
    results = trainer.train(num_steps=args.steps, samples_per_step=args.samples, verbose=args.verbose)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    trainer.save_results(os.path.join(output_dir, 'speed_vote_results.json'))
