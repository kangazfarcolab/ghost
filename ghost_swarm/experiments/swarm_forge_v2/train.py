"""
Swarm Forge V2 - Redesigned Phased Approach
============================================
Sequential phases instead of everything mixed.
Each phase focuses on one technique for maximum benefit.

Phase 1: Speed training (parallel gradients)
Phase 2: Precision refinement (gradient voting)
Phase 3: Evolution (population optimization)

Goal: Clear benefits per phase, reduced overhead
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

# Import components
from speed_demon.train import generate_dream_data
from precision_master.train import gradient_consensus_filter, weighted_gradient_average, compute_gradient_confidence
from self_improving.train import evaluate_fitness, clone_model


# ============================================================================
# SWARM FORGE V2 TRAINER
# ============================================================================

class SwarmForgeV2Trainer:
    """
    Swarm Forge V2 - Phased approach
    
    Phase 1: Speed (parallel gradient farming)
    Phase 2: Precision (gradient voting for stability)
    Phase 3: Evolution (create population, select best)
    """
    
    def __init__(self, num_workers: int = 10, population_size: int = 5):
        self.num_workers = num_workers
        self.population_size = population_size
        
        print(f"Initializing Swarm Forge V2 Trainer...")
        print(f"Workers: {num_workers}, Population: {population_size}")
        
        self.model = GhostWorker(dim=256, num_layers=6)
        mx.eval(self.model.parameters())
        self.optimizer = optim.Adam(learning_rate=0.001)
        
        self.population = []
        self.history = {'phase1': [], 'phase2': [], 'phase3': []}
    
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
    
    def speed_step(self, model, optimizer, data: List[Tuple[str, str]]) -> Tuple[float, float]:
        """Fast parallel step without voting."""
        step_start = time.time()
        
        worker_chunks = [[] for _ in range(self.num_workers)]
        for i, item in enumerate(data):
            worker_chunks[i % self.num_workers].append(item)
        
        all_gradients = []
        all_losses = []
        
        for chunk in worker_chunks:
            if not chunk:
                continue
            
            batch = [q + a for q, a in chunk]
            loss_fn = lambda m: self.compute_loss(m, batch)
            loss, grads = mx.value_and_grad(loss_fn)(model)
            
            all_gradients.append(grads)
            all_losses.append(loss.item())
        
        if all_gradients:
            avg_grads = tree_map(lambda *gs: sum(gs) / len(gs), *all_gradients)
            optimizer.update(model, avg_grads)
            mx.eval(model.parameters())
        
        return sum(all_losses) / len(all_losses) if all_losses else 0, time.time() - step_start
    
    def precision_step(self, model, optimizer, data: List[Tuple[str, str]]) -> Tuple[float, int, float]:
        """Precision step with gradient voting."""
        step_start = time.time()
        
        worker_chunks = [[] for _ in range(self.num_workers)]
        for i, item in enumerate(data):
            worker_chunks[i % self.num_workers].append(item)
        
        all_gradients = []
        all_losses = []
        all_confidences = []
        
        for chunk in worker_chunks:
            if not chunk:
                continue
            
            batch = [q + a for q, a in chunk]
            loss_fn = lambda m: self.compute_loss(m, batch)
            loss, grads = mx.value_and_grad(loss_fn)(model)
            
            loss_val = loss.item()
            all_gradients.append(grads)
            all_losses.append(loss_val)
            all_confidences.append(compute_gradient_confidence(None, loss_val))
        
        # Apply voting
        original_count = len(all_gradients)
        filtered_gradients = gradient_consensus_filter(all_gradients, threshold=1.5)
        filtered_count = original_count - len(filtered_gradients)
        
        if filtered_gradients:
            filtered_conf = all_confidences[:len(filtered_gradients)]
            avg_grads = weighted_gradient_average(filtered_gradients, filtered_conf)
        elif all_gradients:
            avg_grads = tree_map(lambda *gs: sum(gs) / len(gs), *all_gradients)
        else:
            return 0, 0, time.time() - step_start
        
        optimizer.update(model, avg_grads)
        mx.eval(model.parameters())
        
        return sum(all_losses) / len(all_losses), filtered_count, time.time() - step_start
    
    def phase1_speed(self, num_steps: int, samples_per_step: int, verbose: bool = False) -> float:
        """Phase 1: Pure speed training."""
        print("\n" + "=" * 60)
        print("PHASE 1: SPEED TRAINING")
        print("=" * 60)
        
        phase_start = time.time()
        
        for step in range(num_steps):
            data = generate_dream_data(samples_per_step)
            loss, step_time = self.speed_step(self.model, self.optimizer, data)
            speed = samples_per_step / step_time if step_time > 0 else 0
            
            self.history['phase1'].append({'step': step + 1, 'loss': loss, 'speed': speed})
            
            if verbose or (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{num_steps} | Loss: {loss:.4f} | Speed: {speed:.1f} s/s")
        
        phase_time = time.time() - phase_start
        print(f"Phase 1 complete: {phase_time:.2f}s")
        return phase_time
    
    def phase2_precision(self, num_steps: int, samples_per_step: int, verbose: bool = False) -> float:
        """Phase 2: Precision training with voting."""
        print("\n" + "=" * 60)
        print("PHASE 2: PRECISION TRAINING")
        print("=" * 60)
        
        # Lower learning rate for precision
        self.optimizer = optim.Adam(learning_rate=0.0005)
        
        phase_start = time.time()
        total_filtered = 0
        
        for step in range(num_steps):
            data = generate_dream_data(samples_per_step)
            loss, filtered, step_time = self.precision_step(self.model, self.optimizer, data)
            total_filtered += filtered
            
            self.history['phase2'].append({'step': step + 1, 'loss': loss, 'filtered': filtered})
            
            if verbose or (step + 1) % 5 == 0:
                print(f"  Step {step+1}/{num_steps} | Loss: {loss:.4f} | Filtered: {filtered}")
        
        phase_time = time.time() - phase_start
        print(f"Phase 2 complete: {phase_time:.2f}s, Total filtered: {total_filtered}")
        return phase_time
    
    def phase3_evolve(self, num_generations: int, train_steps_per_gen: int, 
                      samples_per_step: int, verbose: bool = False) -> float:
        """Phase 3: Evolution - create population, train, select best."""
        print("\n" + "=" * 60)
        print("PHASE 3: EVOLUTION")
        print("=" * 60)
        
        phase_start = time.time()
        
        # Create population from trained model
        print(f"  Creating population of {self.population_size}...")
        self.population = []
        for i in range(self.population_size):
            if i == 0:
                model = self.model
            else:
                model = clone_model(self.model, mutate=True, mutation_rate=0.02)
            
            self.population.append({
                'id': i,
                'model': model,
                'fitness': 0.0,
                'optimizer': optim.Adam(learning_rate=0.0003)
            })
        
        for gen in range(num_generations):
            # Train each member
            for member in self.population:
                for _ in range(train_steps_per_gen):
                    data = generate_dream_data(samples_per_step // self.population_size)
                    self.speed_step(member['model'], member['optimizer'], data)
            
            # Evaluate fitness
            test_data = generate_dream_data(30)
            for member in self.population:
                member['fitness'] = evaluate_fitness(member['model'], test_data)
            
            self.population.sort(key=lambda x: x['fitness'], reverse=True)
            
            best_fit = self.population[0]['fitness']
            avg_fit = sum(m['fitness'] for m in self.population) / len(self.population)
            
            self.history['phase3'].append({
                'generation': gen + 1,
                'best_fitness': best_fit,
                'avg_fitness': avg_fit
            })
            
            print(f"  Gen {gen+1}/{num_generations} | Best: {best_fit:.4f} | Avg: {avg_fit:.4f}")
            
            # Reproduce
            if gen < num_generations - 1:
                survivors = self.population[:2]
                new_pop = list(survivors)
                while len(new_pop) < self.population_size:
                    parent = random.choice(survivors)
                    child = clone_model(parent['model'], mutate=True, mutation_rate=0.01)
                    new_pop.append({
                        'id': len(new_pop),
                        'model': child,
                        'fitness': 0.0,
                        'optimizer': optim.Adam(learning_rate=0.0003)
                    })
                self.population = new_pop
        
        # Keep best
        self.model = self.population[0]['model']
        
        phase_time = time.time() - phase_start
        print(f"Phase 3 complete: {phase_time:.2f}s")
        return phase_time
    
    def train(self, p1_steps: int = 25, p2_steps: int = 15, p3_gens: int = 3, 
              p3_train_steps: int = 5, samples_per_step: int = 100, verbose: bool = False) -> dict:
        """Full phased training."""
        print("=" * 60)
        print("SWARM FORGE V2 TRAINING (Phased)")
        print(f"Phase 1: {p1_steps} speed steps")
        print(f"Phase 2: {p2_steps} precision steps")
        print(f"Phase 3: {p3_gens} evolution generations")
        print("=" * 60)
        
        total_start = time.time()
        
        t1 = self.phase1_speed(p1_steps, samples_per_step, verbose)
        t2 = self.phase2_precision(p2_steps, samples_per_step, verbose)
        t3 = self.phase3_evolve(p3_gens, p3_train_steps, samples_per_step, verbose)
        
        total_time = time.time() - total_start
        
        # Final evaluation
        test_data = generate_dream_data(50)
        final_fitness = evaluate_fitness(self.model, test_data)
        
        results = {
            'experiment': 'swarm_forge_v2',
            'num_workers': self.num_workers,
            'population_size': self.population_size,
            'p1_steps': p1_steps,
            'p2_steps': p2_steps,
            'p3_gens': p3_gens,
            'total_time': total_time,
            'phase1_time': t1,
            'phase2_time': t2,
            'phase3_time': t3,
            'final_loss': self.history['phase2'][-1]['loss'] if self.history['phase2'] else 0,
            'final_fitness': final_fitness,
            'history': self.history
        }
        
        print("\n" + "=" * 60)
        print("SWARM FORGE V2 RESULTS")
        print("=" * 60)
        print(f"Total time: {total_time:.2f}s")
        print(f"  Phase 1: {t1:.2f}s (Speed)")
        print(f"  Phase 2: {t2:.2f}s (Precision)")
        print(f"  Phase 3: {t3:.2f}s (Evolution)")
        print(f"Final fitness: {final_fitness:.4f}")
        
        return results
    
    def save_results(self, path: str):
        """Save results to JSON."""
        test_data = generate_dream_data(50)
        final_fitness = evaluate_fitness(self.model, test_data)
        
        results = {
            'experiment': 'swarm_forge_v2',
            'final_loss': self.history['phase2'][-1]['loss'] if self.history['phase2'] else 0,
            'final_fitness': final_fitness,
            'history': self.history
        }
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Swarm Forge V2 Training")
    parser.add_argument('--workers', type=int, default=10, help='Number of workers')
    parser.add_argument('--population', type=int, default=5, help='Population size')
    parser.add_argument('--p1-steps', type=int, default=25, help='Phase 1 speed steps')
    parser.add_argument('--p2-steps', type=int, default=15, help='Phase 2 precision steps')
    parser.add_argument('--p3-gens', type=int, default=3, help='Phase 3 generations')
    parser.add_argument('--samples', type=int, default=100, help='Samples per step')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    trainer = SwarmForgeV2Trainer(num_workers=args.workers, population_size=args.population)
    results = trainer.train(
        p1_steps=args.p1_steps,
        p2_steps=args.p2_steps,
        p3_gens=args.p3_gens,
        samples_per_step=args.samples,
        verbose=args.verbose
    )
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    trainer.save_results(os.path.join(output_dir, 'swarm_forge_v2_results.json'))
