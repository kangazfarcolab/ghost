"""
Precision Master - Experiment 2
===============================
Teacher-Student + Gradient Voting + Weight Consensus

Goal: Maximum training precision through quality filtering
Expected: 3-5x better accuracy than baseline

Components:
1. Teacher-Student: Learn from Qwen's soft labels (simulated)
2. Gradient Voting: Filter noisy gradients via consensus
3. Weight Consensus: Confidence-weighted updates
"""

import sys
import os
import time
import json
import math
from typing import List, Dict, Tuple

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
experiments_dir = os.path.dirname(script_dir)
swarm_dir = os.path.dirname(experiments_dir)
base_path = os.path.dirname(swarm_dir)

sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, 'ghost_model'))
sys.path.insert(0, experiments_dir)

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map

# Load ghost_worker using importlib
import importlib.util
ghost_worker_path = os.path.join(base_path, 'ghost_model_v7', 'core', 'ghost_worker.py')
spec = importlib.util.spec_from_file_location("ghost_worker", ghost_worker_path)
ghost_worker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ghost_worker_module)
GhostWorker = ghost_worker_module.GhostWorker


# ============================================================================
# SIMULATED TEACHER (replaces Qwen for testing)
# ============================================================================

class SimulatedTeacher:
    """
    Simulates a teacher model that provides soft labels.
    In production, this would be Qwen 7B or similar.
    """
    
    def __init__(self):
        self.knowledge = {
            "list pods": "kubectl get pods",
            "list services": "kubectl get services",
            "check logs": "kubectl logs",
            "describe pod": "kubectl describe pod",
            "delete pod": "kubectl delete pod",
            "apply yaml": "kubectl apply -f",
            "build image": "docker build -t",
            "run container": "docker run",
        }
    
    def generate_soft_labels(self, question: str) -> mx.array:
        """
        Generate soft probability distribution over bytes.
        Real teacher would run full model inference.
        """
        # Find matching answer
        answer = None
        for key, value in self.knowledge.items():
            if key in question.lower():
                answer = value
                break
        
        if answer is None:
            # Uniform distribution (uncertain)
            return mx.ones(256) / 256
        
        # Create soft labels with high probability on correct bytes
        labels = mx.ones(256) * 0.001  # Small prob for all
        for char in answer:
            byte_val = ord(char)
            # WORKAROUND: .at[].set() is missing in this MLX version, using .add()
            current_val = labels[byte_val]
            diff = 0.1 - current_val
            labels = labels.at[byte_val].add(diff)
        
        # Normalize
        labels = labels / mx.sum(labels)
        return labels


# ============================================================================
# GRADIENT VOTING
# ============================================================================

def gradient_consensus_filter(gradients: List[Dict], threshold: float = 2.0) -> List[Dict]:
    """
    Filter gradients that are too different from consensus.
    
    Args:
        gradients: List of gradient dictionaries from workers
        threshold: Standard deviations to consider as outlier
    
    Returns:
        Filtered list of gradients (outliers removed)
    """
    if len(gradients) <= 2:
        return gradients
    
    # Compute mean and std for each gradient component
    def compute_stats(*gs):
        stacked = mx.stack(gs, axis=0)
        mean = mx.mean(stacked, axis=0)
        std = mx.std(stacked, axis=0) + 1e-8
        return mean, std
    
    # Get stats
    stats = tree_map(compute_stats, *gradients)
    
    # Filter outliers
    filtered = []
    for grad in gradients:
        is_outlier = False
        
        def check_outlier(g, stat):
            nonlocal is_outlier
            mean, std = stat
            diff = mx.abs(g - mean)
            if mx.any(diff > threshold * std):
                is_outlier = True
            return g
        
        tree_map(check_outlier, grad, stats)
        
        if not is_outlier:
            filtered.append(grad)
    
    # Keep at least half
    if len(filtered) < len(gradients) // 2:
        return gradients[:len(gradients)//2]
    
    return filtered


# ============================================================================
# WEIGHT CONSENSUS
# ============================================================================

def compute_gradient_confidence(grad: Dict, loss: float) -> float:
    """
    Compute confidence score for a gradient based on loss.
    Lower loss = higher confidence.
    """
    # Simple inverse relationship
    confidence = 1.0 / (1.0 + loss)
    return confidence


def weighted_gradient_average(gradients: List[Dict], confidences: List[float]) -> Dict:
    """
    Average gradients weighted by confidence.
    """
    if not gradients:
        return {}
    
    total_conf = sum(confidences)
    weights = [c / total_conf for c in confidences]
    
    def weighted_avg(*gs):
        result = mx.zeros_like(gs[0])
        for g, w in zip(gs, weights):
            result = result + g * w
        return result
    
    return tree_map(weighted_avg, *gradients)


# ============================================================================
# PRECISION MASTER TRAINER
# ============================================================================

class PrecisionMasterTrainer:
    """
    Precision Master = Teacher-Student + Gradient Voting + Weight Consensus
    
    Maximizes training precision through:
    1. Learning from teacher's soft labels
    2. Filtering noisy gradients via voting
    3. Confidence-weighted gradient averaging
    """
    
    def __init__(self, num_workers: int = 10):
        self.num_workers = num_workers
        
        # Create model
        self.model = GhostWorker(dim=256, num_layers=6)
        mx.eval(self.model.parameters())
        
        # Teacher
        self.teacher = SimulatedTeacher()
        
        # Optimizer
        self.optimizer = optim.Adam(learning_rate=0.001)
        
        # Stats
        self.history = []
        self.filtered_count = 0
    
    def pad_batch(self, batch: List[str]) -> Tuple[mx.array, mx.array]:
        """
        Tokenize and pad a batch of strings.
        Returns:
            x: [B, L] token indices
            mask: [B, L] 1.0 for valid tokens, 0.0 for padding
        """
        tokenized = [[ord(c) for c in text] for text in batch]
        max_len = max(len(t) for t in tokenized)
        padded = []
        masks = []
        for t in tokenized:
            pad_len = max_len - len(t)
            padded.append(t + [0] * pad_len)
            masks.append([1.0] * len(t) + [0.0] * pad_len)
        return mx.array(padded, dtype=mx.int32), mx.array(masks, dtype=mx.float32)

    def compute_loss(self, model, batch: List[str], questions: List[str]) -> mx.array:
        """Compute loss for a batch of text samples with distillation (Vectorized)."""
        x, mask = self.pad_batch(batch)
        B, L = x.shape
        if L < 2: return mx.array(0.0)
        
        logits = model(x[:, :-1]) # [B, L-1, V]
        targets = x[:, 1:]
        target_mask = mask[:, 1:]
        
        # Cross Entropy
        logits_flat = logits.reshape(-1, 256)
        targets_flat = targets.reshape(-1)
        ce_losses = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='none')
        masked_ce_loss = ce_losses * target_mask.reshape(-1)
        ce_loss_final = masked_ce_loss.sum() / mx.maximum(target_mask.sum(), 1.0)
        
        # Distillation
        # Teacher provides 256-dim soft labels for each sample (independent of L? No, usually per-token or per-sequence)
        # The SimulatedTeacher generates a SINGLE distribution per question (likely bag-of-words style or generic)
        # We will broadcast teacher's single distribution across all timesteps for that sample
        
        # Prepare teacher probs [B, L-1, 256]
        teacher_probs_list = []
        for q in questions:
            tp = self.teacher.generate_soft_labels(q) # [256]
            teacher_probs_list.append(tp)
        
        # Stack [B, 256] -> Broadcast to [B, L-1, 256]
        teacher_probs_batch = mx.stack(teacher_probs_list, axis=0) # [B, 256]
        teacher_probs_seq = mx.broadcast_to(teacher_probs_batch[:, None, :], (B, L-1, 256))
        
        # Compute Distillation Loss (vectorized)
        student_probs = nn.softmax(logits, axis=-1)
        kl_div = mx.sum(teacher_probs_seq * mx.log(teacher_probs_seq / (student_probs + 1e-8) + 1e-8), axis=-1) # [B, L-1]
        
        # Mask padding
        masked_kl = kl_div * target_mask
        distill_loss_final = masked_kl.sum() / mx.maximum(target_mask.sum(), 1.0)
        
        return ce_loss_final + 0.1 * distill_loss_final

    def train_step(self, data: List[Tuple[str, str]]) -> Dict:
        """
        One training step with precision techniques.
        """
        start = time.time()
        
        # Simulate multiple workers
        all_gradients = []
        all_losses = []
        all_confidences = []
        
        # Distribute data round-robin for better balance
        worker_chunks = [[] for _ in range(self.num_workers)]
        for i, item in enumerate(data):
            worker_chunks[i % self.num_workers].append(item)
            
        for i, worker_data in enumerate(worker_chunks):
            if not worker_data:
                continue
            
            # Combine Q and A
            batch = [q + a for q, a in worker_data]
            questions = [q for q, a in worker_data]
            
            # Vectorized Gradient
            loss_fn = lambda m: self.compute_loss(m, batch, questions)
            loss, grads = mx.value_and_grad(loss_fn)(self.model)
            
            loss_val = loss.item()
            all_gradients.append(grads)
            all_losses.append(loss_val)
            all_confidences.append(compute_gradient_confidence(None, loss_val))
        
        # Gradient Voting: Filter outliers
        original_count = len(all_gradients)
        filtered_gradients = gradient_consensus_filter(all_gradients)
        self.filtered_count += original_count - len(filtered_gradients)
        
        # Weight Consensus: Confidence-weighted average
        if filtered_gradients:
            # Match confidences to filtered gradients
            filtered_conf = all_confidences[:len(filtered_gradients)]
            avg_grads = weighted_gradient_average(filtered_gradients, filtered_conf)
            
            # Apply update
            self.optimizer.update(self.model, avg_grads)
            mx.eval(self.model.parameters())
        
        elapsed = time.time() - start
        
        stats = {
            'step': len(self.history) + 1,
            'avg_loss': sum(all_losses) / len(all_losses) if all_losses else 0,
            'gradients_filtered': original_count - len(filtered_gradients),
            'samples': len(data),
            'time': elapsed,
        }
        
        self.history.append(stats)
        return stats
    
    def train(self, num_steps: int = 100, samples_per_step: int = 100, verbose: bool = False) -> Dict:
        """Train using Precision Master strategy."""
        print("="*60)
        print("PRECISION MASTER TRAINING")
        print(f"Workers: {self.num_workers}")
        print(f"Steps: {num_steps}")
        print(f"Samples/step: {samples_per_step}")
        print(f"Verbose: {verbose}")
        print("Features: Teacher-Student + Gradient Voting + Weight Consensus")
        print("="*60)
        
        total_start = time.time()
        
        # Generate training data
        from speed_demon.train import generate_dream_data
        
        for step in range(num_steps):
            data = generate_dream_data(samples_per_step)
            
            # Verbose on first step and every 10th
            step_verbose = verbose and (step == 0 or (step + 1) % 10 == 0)
            if step_verbose:
                print(f"\n--- Step {step+1}/{num_steps} ---")
                print(f"  [Data] Generated {len(data)} samples")
            
            stats = self.train_step(data)
            
            if step_verbose:
                print(f"  [Gradients] Total: {stats['samples']//max(1, self.num_workers)}")
                print(f"  [Voting] Filtered: {stats['gradients_filtered']} outliers")
                print(f"  [Teacher] Distillation applied")
            
            # Always print progress every step if verbose, else every 10
            if verbose or (step + 1) % 10 == 0:
                speed = stats['samples'] / stats['time'] if stats['time'] > 0 else 0
                print(f"Step {step+1}/{num_steps} | "
                      f"Loss: {stats['avg_loss']:.4f} | "
                      f"Filtered: {stats['gradients_filtered']} | "
                      f"Speed: {speed:.1f} samples/sec")
        
        total_time = time.time() - total_start
        total_samples = num_steps * samples_per_step
        
        results = {
            'experiment': 'precision_master',
            'num_workers': self.num_workers,
            'total_steps': num_steps,
            'total_time': total_time,
            'total_samples': total_samples,
            'avg_samples_per_sec': total_samples / total_time if total_time > 0 else 0,
            'total_filtered': self.filtered_count,
            'final_loss': self.history[-1]['avg_loss'] if self.history else 0,
            'history': self.history
        }
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Total time: {total_time:.2f}s")
        print(f"Total samples: {total_samples}")
        print(f"Avg speed: {results['avg_samples_per_sec']:.1f} samples/sec")
        print(f"Gradients filtered: {self.filtered_count}")
        print(f"Final loss: {results['final_loss']:.4f}")
        
        return results
    
    def save_results(self, path: str):
        """Save training results to JSON."""
        results = {
            'experiment': 'precision_master',
            'final_loss': self.history[-1]['avg_loss'] if self.history else 0,
            'total_filtered': self.filtered_count,
            'history': self.history
        }
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Precision Master Training Experiment")
    parser.add_argument('--workers', type=int, default=10, help='Number of workers')
    parser.add_argument('--steps', type=int, default=20, help='Training steps')
    parser.add_argument('--samples', type=int, default=50, help='Samples per step')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    trainer = PrecisionMasterTrainer(num_workers=args.workers)
    results = trainer.train(num_steps=args.steps, samples_per_step=args.samples, verbose=args.verbose)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    trainer.save_results(os.path.join(output_dir, 'precision_master_results.json'))

