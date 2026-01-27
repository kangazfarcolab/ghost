"""
Swarm Forge - Experiment 4 (ULTIMATE COMBO)
============================================
Parallel Gradient + Gradient Voting + Competitive Evolution + Dream Synthesis

Goal: Best of all worlds - speed, precision, and self-improvement
Expected: 50-100x faster, 3x more precise, autonomous

Components:
1. Parallel Gradient Farming: 100x speed
2. Gradient Voting: Filter noise
3. Competitive Evolution: Self-improvement every N steps
4. Dream Synthesis: Unlimited data
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

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
experiments_dir = os.path.dirname(script_dir)
swarm_dir = os.path.dirname(experiments_dir)
base_path = os.path.dirname(swarm_dir)

sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, 'ghost_model'))
sys.path.insert(0, experiments_dir)

# Load ghost_worker using importlib
import importlib.util
ghost_worker_path = os.path.join(base_path, 'ghost_model_v7', 'core', 'ghost_worker.py')
spec = importlib.util.spec_from_file_location("ghost_worker", ghost_worker_path)
ghost_worker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ghost_worker_module)
GhostWorker = ghost_worker_module.GhostWorker

# Import components from other experiments
from speed_demon.train import generate_dream_data, ParallelGradientFarmer
from precision_master.train import gradient_consensus_filter, weighted_gradient_average, compute_gradient_confidence
from self_improving.train import evaluate_fitness, clone_model, mutate_weights


# ============================================================================
# SWARM FORGE TRAINER
# ============================================================================

class SwarmForgeTrainer:
    """
    Swarm Forge = ALL TECHNIQUES COMBINED
    
    The ultimate swarm training system:
    1. Parallel Gradient Farming (speed)
    2. Gradient Voting (precision)
    3. Competitive Evolution (self-improvement)
    4. Dream Synthesis (unlimited data)
    """
    
    def __init__(self, num_workers: int = 20, elite_ratio: float = 0.2, 
                 evolution_interval: int = 50):
        self.num_workers = num_workers
        self.elite_ratio = elite_ratio
        self.elite_count = max(1, int(num_workers * elite_ratio))
        self.evolution_interval = evolution_interval
        
        # Initialize population of workers
        print(f"Initializing Swarm Forge with {num_workers} workers...")
        self.workers = []
        for i in range(num_workers):
            model = GhostWorker(dim=256, num_layers=6)
            mx.eval(model.parameters())
            self.workers.append({
                'id': i,
                'model': model,
                'fitness': 0.0,
                'optimizer': optim.Adam(learning_rate=0.001)
            })
        
        self.step_count = 0
        self.generation = 0
        self.history = []
        self.evolution_history = []
    
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
            pad_len = max_len - len(t)
            padded.append(t + [0] * pad_len)
            masks.append([1.0] * len(t) + [0.0] * pad_len)
        return mx.array(padded, dtype=mx.int32), mx.array(masks, dtype=mx.float32)

    def compute_loss(self, model, batch: List[str]) -> mx.array:
        """Compute loss for a batch of text samples (Vectorized)."""
        x, mask = self.pad_batch(batch)
        B, L = x.shape
        if L < 2: return mx.array(0.0)
        
        logits = model(x[:, :-1])
        targets = x[:, 1:]
        target_mask = mask[:, 1:]
        
        logits_flat = logits.reshape(-1, 256)
        targets_flat = targets.reshape(-1)
        
        losses = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='none')
        masked_loss = losses * target_mask.reshape(-1)
        
        return masked_loss.sum() / mx.maximum(target_mask.sum(), 1.0)

    def parallel_gradient_step(self, data: List[Tuple[str, str]]) -> Dict:
        """
        Perform one step with parallel gradient farming + voting.
        """
        step_start = time.time()
        
        # Split data among workers
        chunk_size = max(1, len(data) // self.num_workers)
        
        all_gradients = []
        all_losses = []
        all_confidences = []
        
        for i, worker in enumerate(self.workers):
            # Get worker's data chunk
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(data))
            worker_data = data[start_idx:end_idx]
            
            if not worker_data:
                continue
            
            model = worker['model']
            
            # Combine Q and A
            batch = [q + a for q, a in worker_data]
            
            # Vectorized Gradient Computation
            loss_fn = lambda m: self.compute_loss(m, batch)
            loss, grads = mx.value_and_grad(loss_fn)(model)
            
            loss_val = loss.item()
            all_gradients.append(grads)
            all_losses.append(loss_val)
            all_confidences.append(compute_gradient_confidence(None, loss_val))

        
        # GRADIENT VOTING: Filter outliers
        original_count = len(all_gradients)
        filtered_gradients = gradient_consensus_filter(all_gradients, threshold=2.0)
        filtered_count = original_count - len(filtered_gradients)
        
        # WEIGHT CONSENSUS: Confidence-weighted average
        if filtered_gradients:
            filtered_conf = all_confidences[:len(filtered_gradients)]
            avg_grads = weighted_gradient_average(filtered_gradients, filtered_conf)
            
            # Apply to ALL workers (synchronized update)
            for worker in self.workers:
                worker['optimizer'].update(worker['model'], avg_grads)
                mx.eval(worker['model'].parameters())
        
        avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0
        step_time = time.time() - step_start
        
        return {
            'avg_loss': avg_loss,
            'gradients_computed': original_count,
            'gradients_filtered': filtered_count,
            'time': step_time
        }
    
    def evolve_population(self, test_data: List[Tuple[str, str]]):
        """
        Perform competitive evolution step.
        """
        self.generation += 1
        
        # Evaluate all workers
        for worker in self.workers:
            worker['fitness'] = evaluate_fitness(worker['model'], test_data)
        
        # Sort by fitness
        self.workers.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Get stats
        fitnesses = [w['fitness'] for w in self.workers]
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        
        # Keep elite, replace rest with mutated clones
        survivors = self.workers[:self.elite_count]
        
        new_workers = []
        
        # Keep elites
        for elite in survivors:
            new_workers.append(elite)
        
        # Fill rest with mutated clones
        while len(new_workers) < self.num_workers:
            parent = random.choice(survivors)
            child_model = clone_model(parent['model'], mutate=True, mutation_rate=0.01)
            
            new_workers.append({
                'id': len(new_workers),
                'model': child_model,
                'fitness': 0.0,
                'optimizer': optim.Adam(learning_rate=0.001)
            })
        
        self.workers = new_workers
        
        self.evolution_history.append({
            'generation': self.generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness
        })
        
        return best_fitness, avg_fitness
    
    def train(self, num_steps: int = 200, samples_per_step: int = 100, verbose: bool = False) -> Dict:
        """Train using Swarm Forge - the ultimate combination."""
        print("="*60)
        print("SWARM FORGE TRAINING (Ultimate Combo)")
        print(f"Workers: {self.num_workers}")
        print(f"Steps: {num_steps}")
        print(f"Samples/step: {samples_per_step}")
        print(f"Evolution interval: every {self.evolution_interval} steps")
        print(f"Verbose: {verbose}")
        print("Features: Parallel Gradient + Voting + Evolution + Dream")
        print("="*60)
        
        total_start = time.time()
        
        for step in range(num_steps):
            self.step_count += 1
            
            # Verbose on first step and every 10th
            step_verbose = verbose and (step == 0 or (step + 1) % 10 == 0)
            
            if step_verbose:
                print(f"\n--- Step {self.step_count}/{num_steps} ---")
            
            # DREAM SYNTHESIS: Generate training data
            data = generate_dream_data(samples_per_step)
            
            if step_verbose:
                print(f"  [Dream] Generated {len(data)} samples")
            
            # PARALLEL GRADIENT + VOTING: Train step
            step_stats = self.parallel_gradient_step(data)
            
            if step_verbose:
                print(f"  [Parallel] {step_stats['gradients_computed']} gradients computed")
                print(f"  [Voting] {step_stats['gradients_filtered']} outliers filtered")
            
            stats = {
                'step': self.step_count,
                'loss': step_stats['avg_loss'],
                'filtered': step_stats['gradients_filtered'],
                'generation': self.generation
            }
            self.history.append(stats)
            
            # COMPETITIVE EVOLUTION: Every N steps
            if self.step_count % self.evolution_interval == 0:
                if verbose:
                    print(f"  [Evolve] Evaluating fitness and evolving...")
                test_data = generate_dream_data(30)
                best_fit, avg_fit = self.evolve_population(test_data)
                print(f"  → Evolution Gen {self.generation}: "
                      f"Best={best_fit:.4f}, Avg={avg_fit:.4f}")
            
            # Progress printing
            if verbose or (step + 1) % 10 == 0:
                speed = samples_per_step / step_stats['time'] if step_stats.get('time', 0) > 0 else 0
                print(f"Step {step+1}/{num_steps} | "
                      f"Loss: {stats['loss']:.4f} | "
                      f"Filtered: {stats['filtered']} | "
                      f"Gen: {self.generation} | "
                      f"Speed: {speed:.1f} s/s")
        
        total_time = time.time() - total_start
        total_samples = num_steps * samples_per_step
        
        results = {
            'experiment': 'swarm_forge',
            'num_workers': self.num_workers,
            'total_steps': num_steps,
            'total_samples': total_samples,
            'total_time': total_time,
            'avg_samples_per_sec': total_samples / total_time if total_time > 0 else 0,
            'final_generation': self.generation,
            'final_loss': self.history[-1]['loss'] if self.history else 0,
            'final_best_fitness': self.evolution_history[-1]['best_fitness'] if self.evolution_history else 0,
            'history': self.history,
            'evolution_history': self.evolution_history
        }
        
        print("\n" + "="*60)
        print("SWARM FORGE RESULTS")
        print("="*60)
        print(f"Total time: {total_time:.2f}s")
        print(f"Total samples: {total_samples}")
        print(f"Avg speed: {results['avg_samples_per_sec']:.1f} samples/sec")
        print(f"Final loss: {results['final_loss']:.4f}")
        print(f"Final generation: {results['final_generation']}")
        print(f"Final best fitness: {results['final_best_fitness']:.4f}")
        
        return results
    
    def save_results(self, path: str):
        """Save training results to JSON."""
        results = {
            'experiment': 'swarm_forge',
            'num_workers': self.num_workers,
            'final_loss': self.history[-1]['loss'] if self.history else 0,
            'final_best_fitness': self.evolution_history[-1]['best_fitness'] if self.evolution_history else 0,
            'history': self.history,
            'evolution_history': self.evolution_history
        }
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Swarm Forge Training Experiment")
    parser.add_argument('--workers', type=int, default=5, help='Number of workers')
    parser.add_argument('--steps', type=int, default=20, help='Training steps')
    parser.add_argument('--samples', type=int, default=30, help='Samples per step')
    parser.add_argument('--evolution', type=int, default=10, help='Evolution interval')
    parser.add_argument('--elite', type=float, default=0.2, help='Elite ratio')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    trainer = SwarmForgeTrainer(
        num_workers=args.workers, 
        elite_ratio=args.elite, 
        evolution_interval=args.evolution
    )
    results = trainer.train(num_steps=args.steps, samples_per_step=args.samples, verbose=args.verbose)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    trainer.save_results(os.path.join(output_dir, 'swarm_forge_results.json'))

