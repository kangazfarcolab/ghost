"""
Swarm Look-Ahead - Ultimate Training Approach
==============================================
Combines:
1. Parallel Gradient Farming (speed)
2. Consensus-based Adaptive LR (smart)
3. Look-Ahead Validation (ONLY apply gradients that actually work)

The key innovation: We TEST gradients before applying them.
Only gradients that ACTUALLY reduce loss are kept.
"""

import sys
import os
import time
import json
from typing import List, Tuple
from copy import deepcopy

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


class SwarmLookAheadTrainer:
    """
    The Ultimate Swarm Trainer:
    1. Parallel gradients (speed)
    2. Consensus adaptive LR (smart)
    3. Look-ahead validation (GUARANTEED improvement)
    """
    
    def __init__(self, num_workers: int = 8, base_lr: float = 0.001, boost_factor: float = 2.0):
        self.num_workers = num_workers
        self.base_lr = base_lr
        self.boost_factor = boost_factor
        
        print("=" * 60)
        print("SWARM LOOK-AHEAD TRAINER")
        print("=" * 60)
        print(f"Workers: {num_workers}")
        print(f"Base LR: {base_lr}")
        print("Novel Features:")
        print("  1. Parallel Gradient Farming")
        print("  2. Consensus-based Adaptive LR")
        print("  3. Look-Ahead Validation (only apply if loss drops)")
        print("=" * 60)
        
        self.model = GhostWorker(dim=256, num_layers=6)
        mx.eval(self.model.parameters())
        self.optimizer = optim.Adam(learning_rate=base_lr)
        
        self.history = []
        self.rejected_updates = 0
        self.accepted_updates = 0
    
    def pad_batch(self, batch: List[str]) -> Tuple[mx.array, mx.array]:
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
    
    def compute_consensus(self, gradients: List) -> float:
        if len(gradients) < 2:
            return 0.5
        
        def flatten_grads(grads):
            flat = []
            for _, v in tree_flatten(grads):
                flat.append(v.reshape(-1))
            return mx.concatenate(flat)
        
        flat_grads = [flatten_grads(g) for g in gradients]
        
        similarities = []
        for i in range(len(flat_grads)):
            for j in range(i + 1, len(flat_grads)):
                g1, g2 = flat_grads[i], flat_grads[j]
                dot = mx.sum(g1 * g2)
                norm1 = mx.sqrt(mx.sum(g1 * g1) + 1e-8)
                norm2 = mx.sqrt(mx.sum(g2 * g2) + 1e-8)
                sim = dot / (norm1 * norm2)
                similarities.append(sim)
        
        avg_sim = sum(similarities) / len(similarities)
        return (float(mx.array(avg_sim).item()) + 1) / 2
    
    def save_model_state(self):
        """Save current model state for rollback."""
        return dict(tree_flatten(self.model.parameters()))
    
    def restore_model_state(self, state):
        """Restore model state."""
        self.model.load_weights(list(state.items()))
        mx.eval(self.model.parameters())
    
    def swarm_step_with_lookahead(self, train_data: List[Tuple[str, str]], 
                                   val_data: List[Tuple[str, str]]) -> dict:
        """
        The key innovation: Look-ahead validation.
        
        1. Compute gradients from swarm
        2. SAVE current model state
        3. Apply update
        4. CHECK: did validation loss go down?
        5. If YES: keep update
        6. If NO: rollback and try smaller step
        """
        
        # 1. Compute validation loss BEFORE update
        val_batch = [q + a for q, a in val_data]
        loss_before = self.compute_loss(self.model, val_batch).item()
        
        # 2. Compute gradients from swarm
        worker_chunks = [[] for _ in range(self.num_workers)]
        for i, item in enumerate(train_data):
            worker_chunks[i % self.num_workers].append(item)
        
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
            return {'accepted': False, 'reason': 'no_grads'}
        
        # 3. Compute consensus
        consensus = self.compute_consensus(all_gradients)
        lr_multiplier = 0.5 + consensus * self.boost_factor
        
        # 4. Average and scale gradients
        avg_grads = tree_map(lambda *gs: sum(gs) / len(gs), *all_gradients)
        scaled_grads = tree_map(lambda g: g * lr_multiplier, avg_grads)
        
        # 5. SAVE state before update (for rollback)
        saved_state = self.save_model_state()
        
        # 6. Apply update
        self.optimizer.update(self.model, scaled_grads)
        mx.eval(self.model.parameters())
        
        # 7. CHECK: did validation loss improve?
        loss_after = self.compute_loss(self.model, val_batch).item()
        
        if loss_after < loss_before:
            # SUCCESS! Keep the update
            self.accepted_updates += 1
            return {
                'accepted': True,
                'train_loss': sum(all_losses) / len(all_losses),
                'val_before': loss_before,
                'val_after': loss_after,
                'improvement': loss_before - loss_after,
                'consensus': consensus,
                'lr_mult': lr_multiplier
            }
        else:
            # ROLLBACK! Update made things worse
            self.restore_model_state(saved_state)
            self.rejected_updates += 1
            
            # Try with smaller step (half the LR)
            smaller_grads = tree_map(lambda g: g * 0.3, avg_grads)
            self.optimizer.update(self.model, smaller_grads)
            mx.eval(self.model.parameters())
            
            loss_retry = self.compute_loss(self.model, val_batch).item()
            
            if loss_retry < loss_before:
                # Smaller step worked
                return {
                    'accepted': True,
                    'train_loss': sum(all_losses) / len(all_losses),
                    'val_before': loss_before,
                    'val_after': loss_retry,
                    'improvement': loss_before - loss_retry,
                    'consensus': consensus,
                    'lr_mult': 0.3,
                    'was_retry': True
                }
            else:
                # Even smaller step didn't help, rollback completely
                self.restore_model_state(saved_state)
                return {
                    'accepted': False,
                    'train_loss': sum(all_losses) / len(all_losses),
                    'val_before': loss_before,
                    'val_after': loss_after,
                    'consensus': consensus,
                    'reason': 'no_improvement'
                }
    
    def train(self, num_steps: int = 30, samples_per_step: int = 80, verbose: bool = False) -> dict:
        print(f"\nTraining for {num_steps} steps, {samples_per_step} samples/step")
        print("-" * 60)
        
        total_start = time.time()
        
        for step in range(num_steps):
            # Generate training data
            train_data = generate_dream_data(samples_per_step)
            
            # Generate validation data (fresh, unseen)
            val_data = generate_dream_data(20)
            
            # Swarm step with look-ahead
            result = self.swarm_step_with_lookahead(train_data, val_data)
            
            stats = {
                'step': step + 1,
                'accepted': result.get('accepted', False),
                'train_loss': result.get('train_loss', 0),
                'val_before': result.get('val_before', 0),
                'val_after': result.get('val_after', 0),
                'improvement': result.get('improvement', 0),
                'consensus': result.get('consensus', 0)
            }
            self.history.append(stats)
            
            if verbose or (step + 1) % 5 == 0:
                status = "✓" if result.get('accepted', False) else "✗"
                retry = " (retry)" if result.get('was_retry', False) else ""
                print(f"Step {step+1:3d}/{num_steps} {status}{retry} | "
                      f"ValLoss: {result.get('val_after', 0):.4f} | "
                      f"Δ: {result.get('improvement', 0):+.4f} | "
                      f"Consensus: {result.get('consensus', 0):.2f}")
        
        total_time = time.time() - total_start
        total_samples = num_steps * samples_per_step
        
        accept_rate = self.accepted_updates / max(self.accepted_updates + self.rejected_updates, 1)
        
        # Final validation
        final_val = generate_dream_data(50)
        final_batch = [q + a for q, a in final_val]
        final_loss = self.compute_loss(self.model, final_batch).item()
        
        results = {
            'experiment': 'swarm_lookahead',
            'num_workers': self.num_workers,
            'total_steps': num_steps,
            'total_samples': total_samples,
            'total_time': total_time,
            'avg_samples_per_sec': total_samples / total_time if total_time > 0 else 0,
            'final_loss': final_loss,
            'accepted_updates': self.accepted_updates,
            'rejected_updates': self.rejected_updates,
            'accept_rate': accept_rate,
            'history': self.history
        }
        
        print("\n" + "=" * 60)
        print("SWARM LOOK-AHEAD RESULTS")
        print("=" * 60)
        print(f"Total time: {total_time:.2f}s")
        print(f"Speed: {results['avg_samples_per_sec']:.1f} samples/sec")
        print(f"Final loss: {final_loss:.4f}")
        print(f"Accept rate: {accept_rate:.1%} ({self.accepted_updates}/{self.accepted_updates + self.rejected_updates})")
        
        return results
    
    def save_results(self, path: str):
        final_val = generate_dream_data(50)
        final_batch = [q + a for q, a in final_val]
        final_loss = self.compute_loss(self.model, final_batch).item()
        
        results = {
            'experiment': 'swarm_lookahead',
            'final_loss': final_loss,
            'accept_rate': self.accepted_updates / max(self.accepted_updates + self.rejected_updates, 1),
            'history': self.history
        }
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Swarm Look-Ahead Training")
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--steps', type=int, default=20, help='Training steps')
    parser.add_argument('--samples', type=int, default=60, help='Samples per step')
    parser.add_argument('--lr', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('--boost', type=float, default=2.0, help='LR boost factor')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    trainer = SwarmLookAheadTrainer(
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
    trainer.save_results(os.path.join(output_dir, 'swarm_lookahead_results.json'))
