"""
Vectorized Swarm - Experiment 8 (OPTIMIZED)
===========================================
Fully vectorized implementation to maximize GPU utilization.
Replaces Python loops with MLX batch operations.

Key Improvements:
1. Pre-tokenization (No encoding inside training loop)
2. Vectorized Batch Processing (No loop over workers)
3. Simulated Parallelism via Batch Dimensions
"""

import sys
import os
import time
import json
import random
from typing import List, Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map

# Add paths to sys.path BEFORE importing mlx to avoid shadowing
script_dir = os.path.dirname(os.path.abspath(__file__))
experiments_dir = os.path.dirname(script_dir)
swarm_dir = os.path.dirname(experiments_dir)
base_path = os.path.dirname(swarm_dir)

sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, 'ghost_model'))
sys.path.insert(0, experiments_dir)

# Load ghost_worker
import importlib.util
ghost_worker_path = os.path.join(base_path, 'ghost_model_v7', 'core', 'ghost_worker.py')
spec = importlib.util.spec_from_file_location("ghost_worker", ghost_worker_path)
ghost_worker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ghost_worker_module)
GhostWorker = ghost_worker_module.GhostWorker

from speed_demon.train import generate_dream_data

# ============================================================================
# EFFICIENT DATA PIPELINE
# ============================================================================

class VectorizedDataset:
    """
    Pre-tokenizes all data into a single efficient MLX array.
    """
    def __init__(self, data: List[Tuple[str, str]], max_len: int = 128):
        self.data = data
        self.max_len = max_len
        
        print(f"  [Data] Pre-tokenizing {len(data)} samples...")
        start = time.time()
        
        # Tokenize all at once (Python loop only once at startup)
        # Pad with 0 (null) or specific pad token
        self.indices = []
        for q, a in data:
            full_text = q + a
            tokens = [ord(c) for c in full_text]
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            self.indices.append(tokens)
            
        # Convert to one massive tensor? 
        # Variable length problematic for MLX? 
        # For max speed on GPU, we want fixed shapes. Padding is best.
        
        # Determine max length actually seen
        actual_max = max(len(t) for t in self.indices) if self.indices else 0
        pad_len = min(actual_max + 8, max_len) # Add buffer
        
        # Pad and stack
        padded_data = []
        for tokens in self.indices:
            pad_size = pad_len - len(tokens)
            padded = tokens + [0] * pad_size # 0 as pad
            padded_data.append(padded)
            
        self.tensor = mx.array(padded_data, dtype=mx.int32)
        print(f"  [Data] Tokenization done in {time.time() - start:.3f}s. Tensor shape: {self.tensor.shape}")
        
    def __len__(self):
        return len(self.data)
        
    def get_batch(self, batch_size: int) -> mx.array:
        """Get random batch of indices."""
        idx = mx.random.randint(0, len(self.data), (batch_size,))
        return self.tensor[idx]

# ============================================================================
# VECTORIZED TRAINER
# ============================================================================

class VectorizedTrainer:
    def __init__(self, num_workers: int = 100):
        # In vectorized mode, "workers" are just batch size multipliers initially.
        # We simulate N workers by processing N samples in parallel.
        self.num_workers = num_workers 
        
        print(f"Initializing Vectorized Swarm (Simulating {num_workers} workers)...")
        self.model = GhostWorker(dim=256, num_layers=6)
        mx.eval(self.model.parameters())
        
        self.optimizer = optim.Adam(learning_rate=0.001)
        self.history = []
        
    def train(self, num_steps: int = 50, samples_per_step: int = 100, verbose: bool = True):
        """
        Train using fully vectorized batch processing.
        samples_per_step effectively becomes batch_size.
        """
        print(f"\n{'='*60}")
        print(f"VECTORIZED SWARM TRAINING")
        print(f"Simulated Workers: {self.num_workers}")
        print(f"Steps: {num_steps}")
        print(f"Batch Size: {samples_per_step}")
        print(f"Verbose: {verbose}")
        print(f"{'='*60}\n")
        
        # Generate initial data
        raw_data = generate_dream_data(num_steps * samples_per_step * 2) # Generate plenty
        dataset = VectorizedDataset(raw_data)
        
        start_time = time.time()
        
        # JIT compile the training step for maximum speed
        # We must use a functional pattern: pass params, update model, compute loss
        
        def loss_fn(params, X):
            self.model.update(params)
            # X shape: (Batch, Seq_Len)
            logits = self.model(X[:, :-1])
            targets = X[:, 1:]
            
            # Mask padding (0)
            mask = (targets != 0)
            
            # Compute Cross Entropy
            ce = nn.losses.cross_entropy(logits, targets)
            
            # Apply mask
            ce = ce * mask
            
            # Average non-padded loss
            return ce.sum() / mask.sum()

        @mx.compile
        def step_fn(params, X):
            loss, grads = mx.value_and_grad(loss_fn)(params, X)
            return loss, grads

        print("  [Train] JIT compiling step function...")
        # Warmup
        dummy_batch = dataset.get_batch(samples_per_step)
        
        # Capture concrete params before warmup compilation
        warmup_concrete = tree_map(lambda x: x, self.model.parameters())
        
        # Pass parameters, not model object
        _, _ = step_fn(warmup_concrete, dummy_batch)
        
        # RESTORE concrete params immediately after warmup
        self.model.update(warmup_concrete)
        
        print("  [Train] Compilation complete. Starting loop.")
        
        total_samples = 0
        
        for i in range(num_steps):
            step_start = time.time()
            
            # Get Batch
            batch = dataset.get_batch(samples_per_step)
            
            # 1. Capture concrete parameters (backup)
            # We map identity to ensure we have references to the concrete arrays
            concrete_params = tree_map(lambda x: x, self.model.parameters())
            
            # 2. Forward + Backward (Model gets corrupted with Tracers here due to internal update)
            loss, grads = step_fn(concrete_params, batch)
            
            # 3. Restore concrete parameters immediately
            self.model.update(concrete_params)
            
            # 4. Update with optimizer (now safely uses concrete arrays + concrete grads)
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state)
            
            step_time = time.time() - step_start
            speed = samples_per_step / max(step_time, 1e-6)
            total_samples += samples_per_step
            
            self.history.append({
                'step': i,
                'loss': loss.item(),
                'speed': speed
            })
            
            if verbose:
                print(f"Step {i+1}/{num_steps} | Loss: {loss.item():.4f} | Speed: {speed:.1f} samples/sec")
                
        total_time = time.time() - start_time
        avg_speed = total_samples / total_time
        
        results = {
            'total_time': total_time,
            'total_samples': total_samples,
            'avg_samples_per_sec': avg_speed,
            'final_loss': self.history[-1]['loss'] if self.history else 0,
            'history': self.history
        }
        
        print("\n" + "="*60)
        print("VECTORIZED RESULTS")
        print("="*60)
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg speed: {avg_speed:.1f} samples/sec")
        print(f"Final loss: {results['final_loss']:.4f}")
        
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--batch', type=int, default=1000)
    args = parser.parse_args()
    
    trainer = VectorizedTrainer(num_workers=args.batch // 10) # Arbitrary worker calculation
    trainer.train(num_steps=args.steps, samples_per_step=args.batch)
