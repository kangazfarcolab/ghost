"""
Speed Demon - Experiment 1
==========================
Parallel Gradient Farming + Dream Synthesis (simulated)

Goal: Maximum training speed through parallelism
Expected: 10-100x faster than single model training

Components:
1. Parallel Gradient Farming: N workers compute gradients on different batches
2. Simple gradient averaging
3. Simulated Dream Synthesis (random Q&A data generation)
"""

import sys
import os
import time
import json
from typing import List, Dict, Tuple

# Add paths - speed_demon/train.py is in ghost_swarm/experiments/speed_demon/
# So we need to go up: speed_demon -> experiments -> ghost_swarm -> ex
script_dir = os.path.dirname(os.path.abspath(__file__))  # speed_demon/
experiments_dir = os.path.dirname(script_dir)  # experiments/
swarm_dir = os.path.dirname(experiments_dir)   # ghost_swarm/
base_path = os.path.dirname(swarm_dir)          # ex/

sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, 'ghost_model'))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map

# Load ghost_worker using importlib (direct file path)
import importlib.util
ghost_worker_path = os.path.join(base_path, 'ghost_model_v7', 'core', 'ghost_worker.py')
spec = importlib.util.spec_from_file_location("ghost_worker", ghost_worker_path)
ghost_worker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ghost_worker_module)
GhostWorker = ghost_worker_module.GhostWorker


# ============================================================================
# DREAM SYNTHESIS (Simulated - no Qwen needed for testing)
# ============================================================================

def generate_dream_data(num_samples: int = 100) -> List[Tuple[str, str]]:
    """Generate synthetic Q&A pairs for training."""
    templates = [
        ("Q: How to list pods?\nA:", "kubectl get pods"),
        ("Q: How to list services?\nA:", "kubectl get services"),
        ("Q: How to check logs?\nA:", "kubectl logs <pod>"),
        ("Q: How to describe pod?\nA:", "kubectl describe pod <name>"),
        ("Q: How to delete pod?\nA:", "kubectl delete pod <name>"),
        ("Q: How to apply yaml?\nA:", "kubectl apply -f <file>.yaml"),
        ("Q: How to get deployments?\nA:", "kubectl get deployments"),
        ("Q: How to scale deployment?\nA:", "kubectl scale deployment <name> --replicas=N"),
        ("Q: How to exec into pod?\nA:", "kubectl exec -it <pod> -- /bin/bash"),
        ("Q: How to port forward?\nA:", "kubectl port-forward <pod> 8080:80"),
        ("Q: How to build docker image?\nA:", "docker build -t <name> ."),
        ("Q: How to run docker container?\nA:", "docker run -d <image>"),
        ("Q: How to list containers?\nA:", "docker ps"),
        ("Q: How to stop container?\nA:", "docker stop <container>"),
        ("Q: How to view docker logs?\nA:", "docker logs <container>"),
    ]
    
    # Expand with variations
    data = []
    for _ in range(num_samples // len(templates) + 1):
        for q, a in templates:
            data.append((q, a))
    
    return data[:num_samples]


# ============================================================================
# PARALLEL GRADIENT FARMING
# ============================================================================

class ParallelGradientFarmer:
    """
    Simulates parallel gradient computation across N workers.
    
    In a real distributed setup, each worker would be on a separate process/GPU.
    Here we simulate by computing gradients sequentially but treating them as parallel.
    """
    
    def __init__(self, num_workers: int = 10, dim: int = 256, num_layers: int = 6):
        self.num_workers = num_workers
        
        # Create base model (shared weights initially)
        self.base_model = GhostWorker(dim=dim, num_layers=num_layers)
        mx.eval(self.base_model.parameters())
        
        # All workers share same weights (in real setup, they'd sync periodically)
        self.workers = [self.base_model for _ in range(num_workers)]
        
        # Optimizer
        self.optimizer = optim.Adam(learning_rate=0.001)
        
        # Stats
        self.total_steps = 0
        self.total_samples = 0
    
    def pad_batch(self, batch: List[str]) -> Tuple[mx.array, mx.array]:
        """
        Tokenize and pad a batch of strings.
        Returns:
            x: [B, L] token indices
            mask: [B, L] 1.0 for valid tokens, 0.0 for padding
        """
        # Tokenize
        tokenized = [[ord(c) for c in text] for text in batch]
        
        # Find max length
        max_len = max(len(t) for t in tokenized)
        
        # Pad
        padded = []
        masks = []
        for t in tokenized:
            # Pad with 0 (null byte)
            pad_len = max_len - len(t)
            padded.append(t + [0] * pad_len)
            masks.append([1.0] * len(t) + [0.0] * pad_len)
            
        return mx.array(padded, dtype=mx.int32), mx.array(masks, dtype=mx.float32)

    def compute_loss(self, model, batch: List[str]) -> mx.array:
        """Compute loss for a batch of text samples (Vectorized)."""
        # 1. Prepare batch
        x, mask = self.pad_batch(batch)
        B, L = x.shape
        
        if L < 2:
            return mx.array(0.0)
            
        # 2. Forward pass (Vectorized)
        # Input: x[:, :-1] -> [B, L-1]
        # Target: x[:, 1:] -> [B, L-1]
        logits = model(x[:, :-1])  # [B, L-1, Vocab]
        targets = x[:, 1:]
        target_mask = mask[:, 1:]
        
        # 3. Compute Loss
        # Flatten for cross_entropy
        logits_flat = logits.reshape(-1, 256)
        targets_flat = targets.reshape(-1)
        
        losses = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='none')
        
        # Apply mask to ignore padding
        mask_flat = target_mask.reshape(-1)
        masked_loss = losses * mask_flat
        
        # Average over valid tokens
        total_loss = masked_loss.sum()
        num_valid = mask_flat.sum()
        
        return total_loss / mx.maximum(num_valid, 1.0)
    
    def farm_gradients(self, data: List[Tuple[str, str]], verbose: bool = False) -> List[Dict]:
        """
        Farm gradients from N workers, each on different data slice.
        Returns list of gradient dictionaries.
        """
        effective_workers = min(self.num_workers, len(data))
        
        if verbose:
            print(f"  [Farm] Data: {len(data)}, Workers: {self.num_workers} → Effective: {effective_workers}")
        
        # Distribute data
        chunks = [[] for _ in range(effective_workers)]
        for i, sample in enumerate(data):
            chunks[i % effective_workers].append(sample)
        
        gradients = []
        losses = []
        
        # Vectorized Gradient Computation
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            
            # Combine Q and A
            batch = [q + a for q, a in chunk]
            
            # Compute gradient for the WHOLE chunk at once
            loss_fn = lambda m: self.compute_loss(m, batch)
            loss, grads = mx.value_and_grad(loss_fn)(self.base_model)
            
            gradients.append(grads)
            losses.append(loss.item())
            
            if verbose and i < 3:
                print(f"  [Worker {i}] Batch Size: {len(chunk)}, Loss: {loss.item():.4f}")
        
        if verbose:
            print(f"  [Farm] Computed {len(gradients)} gradients, Avg loss: {sum(losses)/len(losses):.4f}")
        
        return gradients, losses
    
    def average_gradients(self, gradients: List[Dict]) -> Dict:
        """Average gradients from all workers."""
        if not gradients:
            return {}
        
        # Average each parameter's gradient
        def avg_grad(*gs):
            stacked = mx.stack(gs, axis=0)
            return mx.mean(stacked, axis=0)
        
        averaged = tree_map(avg_grad, *gradients)
        return averaged
    
    def train_step(self, data: List[Tuple[str, str]], verbose: bool = False) -> Dict:
        """
        One training step with parallel gradient farming.
        
        Returns stats about the step.
        """
        start = time.time()
        
        # Farm gradients from all workers
        gradients, losses = self.farm_gradients(data, verbose=verbose)
        
        if verbose:
            print(f"  [Aggregate] Averaging {len(gradients)} gradients...")
        
        # Average gradients
        avg_grads = self.average_gradients(gradients)
        
        # Apply update
        if avg_grads:
            self.optimizer.update(self.base_model, avg_grads)
            mx.eval(self.base_model.parameters())
        elif verbose:
            print("  [Warning] No gradients to apply!")
        
        elapsed = time.time() - start
        
        self.total_steps += 1
        self.total_samples += len(data)
        
        return {
            'step': self.total_steps,
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'num_workers': len(gradients),
            'samples': len(data),
            'time': elapsed,
            'samples_per_sec': len(data) / elapsed if elapsed > 0 else 0
        }


# ============================================================================
# SPEED DEMON TRAINER
# ============================================================================

class SpeedDemonTrainer:
    """
    Speed Demon = Parallel Gradient Farming + Dream Synthesis
    
    Maximizes training speed through:
    1. Parallel gradient computation across N workers
    2. Unlimited synthetic training data generation
    """
    
    def __init__(self, num_workers: int = 10):
        self.farmer = ParallelGradientFarmer(num_workers=num_workers)
        self.history = []
    
    def train(self, num_steps: int = 100, samples_per_step: int = 100, verbose: bool = False) -> Dict:
        """
        Train using Speed Demon strategy.
        
        Args:
            num_steps: Number of training iterations
            samples_per_step: Synthetic samples to generate per step
            verbose: Print detailed logging
        """
        print("="*60)
        print("SPEED DEMON TRAINING")
        print(f"Workers: {self.farmer.num_workers}")
        print(f"Steps: {num_steps}")
        print(f"Samples/step: {samples_per_step}")
        print(f"Verbose: {verbose}")
        print("="*60)
        
        total_start = time.time()
        
        for step in range(num_steps):
            # Generate dream data (simulated)
            data = generate_dream_data(samples_per_step)
            
            # Verbose on first step and every 10th
            step_verbose = verbose and (step == 0 or (step + 1) % 10 == 0)
            if step_verbose:
                print(f"\n--- Step {step+1}/{num_steps} ---")
            
            # Train step with parallel gradient farming
            stats = self.farmer.train_step(data, verbose=step_verbose)
            
            self.history.append(stats)
            
            # Always print progress every step if verbose, else every 10
            if verbose or (step + 1) % 10 == 0:
                print(f"Step {step+1}/{num_steps} | "
                      f"Loss: {stats['avg_loss']:.4f} | "
                      f"Workers: {stats['num_workers']} | "
                      f"Speed: {stats['samples_per_sec']:.1f} samples/sec")
        
        total_time = time.time() - total_start
        total_samples = sum(h['samples'] for h in self.history)
        
        results = {
            'experiment': 'speed_demon',
            'num_workers': self.farmer.num_workers,
            'total_steps': num_steps,
            'total_samples': total_samples,
            'total_time': total_time,
            'avg_samples_per_sec': total_samples / total_time,
            'final_loss': self.history[-1]['avg_loss'] if self.history else 0,
            'history': self.history
        }
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Total time: {total_time:.2f}s")
        print(f"Total samples: {total_samples}")
        print(f"Avg speed: {results['avg_samples_per_sec']:.1f} samples/sec")
        print(f"Final loss: {results['final_loss']:.4f}")
        
        return results
    
    def save_results(self, path: str):
        """Save training results to JSON."""
        results = {
            'experiment': 'speed_demon',
            'num_workers': self.farmer.num_workers,
            'total_steps': len(self.history),
            'final_loss': self.history[-1]['avg_loss'] if self.history else 0,
            'history': self.history
        }
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Speed Demon Training Experiment")
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers')
    parser.add_argument('--steps', type=int, default=50, help='Training steps')
    parser.add_argument('--samples', type=int, default=100, help='Samples per step')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    trainer = SpeedDemonTrainer(num_workers=args.workers)
    results = trainer.train(num_steps=args.steps, samples_per_step=args.samples, verbose=args.verbose)
    
    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    trainer.save_results(os.path.join(output_dir, 'speed_demon_results.json'))

    trainer.save_results(os.path.join(output_dir, 'speed_demon_results.json'))
