"""
Speed Evolve - Phased Speed + Evolution
========================================
Phase 1: Speed train (fast convergence)
Phase 2: Evolution (diversity + selection)
Phase 3: Fine-tune winner (polish)

Goal: Best of both worlds without mixing overheads
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
from self_improving.train import evaluate_fitness, clone_model, mutate_weights


# ============================================================================
# SPEED EVOLVE TRAINER
# ============================================================================

class SpeedEvolveTrainer:
    """
    Phased training: Speed → Evolution → Fine-tune
    
    Phase 1: Fast parallel training for initial convergence
    Phase 2: Create population, evolve best
    Phase 3: Fine-tune the winner
    """
    
    def __init__(self, num_workers: int = 10, population_size: int = 5):
        self.num_workers = num_workers
        self.population_size = population_size
        
        print(f"Initializing Speed Evolve Trainer...")
        print(f"Workers: {num_workers}, Population: {population_size}")
        
        # Start with single model
        self.model = GhostWorker(dim=256, num_layers=6)
        mx.eval(self.model.parameters())
        self.optimizer = optim.Adam(learning_rate=0.001)
        
        self.population = []  # Created in Phase 2
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
    
    def speed_train_step(self, model, optimizer, data: List[Tuple[str, str]]) -> float:
        """Fast parallel training step."""
        # Distribute data
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
        
        # Average gradients
        if all_gradients:
            avg_grads = tree_map(lambda *gs: sum(gs) / len(gs), *all_gradients)
            optimizer.update(model, avg_grads)
            mx.eval(model.parameters())
        
        return sum(all_losses) / len(all_losses) if all_losses else 0
    
    def phase1_speed_train(self, num_steps: int, samples_per_step: int, verbose: bool = False):
        """Phase 1: Fast parallel training."""
        print("\n" + "=" * 60)
        print("PHASE 1: SPEED TRAINING")
        print("=" * 60)
        
        phase_start = time.time()
        
        for step in range(num_steps):
            data = generate_dream_data(samples_per_step)
            loss = self.speed_train_step(self.model, self.optimizer, data)
            
            stats = {'step': step + 1, 'loss': loss}
            self.history['phase1'].append(stats)
            
            if verbose or (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{num_steps} | Loss: {loss:.4f}")
        
        phase_time = time.time() - phase_start
        print(f"Phase 1 complete: {phase_time:.2f}s, Final loss: {loss:.4f}")
        return phase_time
    
    def phase2_evolve(self, num_generations: int, samples_per_gen: int, verbose: bool = False):
        """Phase 2: Create population and evolve."""
        print("\n" + "=" * 60)
        print("PHASE 2: EVOLUTION")
        print("=" * 60)
        
        phase_start = time.time()
        
        # Create population from trained model
        print(f"  Creating population of {self.population_size} from trained model...")
        self.population = []
        for i in range(self.population_size):
            if i == 0:
                # Keep original
                model = self.model
            else:
                # Mutated clone
                model = clone_model(self.model, mutate=True, mutation_rate=0.02)
            
            self.population.append({
                'id': i,
                'model': model,
                'fitness': 0.0,
                'optimizer': optim.Adam(learning_rate=0.001)
            })
        
        # Evolve
        for gen in range(num_generations):
            # Train each member
            data = generate_dream_data(samples_per_gen)
            subset_size = len(data) // self.population_size
            
            for i, member in enumerate(self.population):
                subset = data[i * subset_size:(i + 1) * subset_size]
                if subset:
                    self.speed_train_step(member['model'], member['optimizer'], subset)
            
            # Evaluate fitness
            test_data = generate_dream_data(30)
            for member in self.population:
                member['fitness'] = evaluate_fitness(member['model'], test_data)
            
            # Sort by fitness
            self.population.sort(key=lambda x: x['fitness'], reverse=True)
            
            best_fit = self.population[0]['fitness']
            avg_fit = sum(m['fitness'] for m in self.population) / len(self.population)
            
            stats = {
                'generation': gen + 1,
                'best_fitness': best_fit,
                'avg_fitness': avg_fit
            }
            self.history['phase2'].append(stats)
            
            if verbose or True:
                print(f"  Gen {gen+1}/{num_generations} | Best: {best_fit:.4f} | Avg: {avg_fit:.4f}")
            
            # Reproduce (keep top half, replace bottom half)
            if gen < num_generations - 1:
                elite_count = max(1, self.population_size // 2)
                survivors = self.population[:elite_count]
                
                new_pop = list(survivors)
                while len(new_pop) < self.population_size:
                    parent = random.choice(survivors)
                    child = clone_model(parent['model'], mutate=True, mutation_rate=0.01)
                    new_pop.append({
                        'id': len(new_pop),
                        'model': child,
                        'fitness': 0.0,
                        'optimizer': optim.Adam(learning_rate=0.001)
                    })
                self.population = new_pop
        
        # Set best as main model
        self.model = self.population[0]['model']
        self.optimizer = optim.Adam(learning_rate=0.0005)  # Lower LR for fine-tuning
        
        phase_time = time.time() - phase_start
        print(f"Phase 2 complete: {phase_time:.2f}s, Best fitness: {self.population[0]['fitness']:.4f}")
        return phase_time
    
    def phase3_finetune(self, num_steps: int, samples_per_step: int, verbose: bool = False):
        """Phase 3: Fine-tune the winner."""
        print("\n" + "=" * 60)
        print("PHASE 3: FINE-TUNE WINNER")
        print("=" * 60)
        
        phase_start = time.time()
        
        for step in range(num_steps):
            data = generate_dream_data(samples_per_step)
            loss = self.speed_train_step(self.model, self.optimizer, data)
            
            stats = {'step': step + 1, 'loss': loss}
            self.history['phase3'].append(stats)
            
            if verbose or (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{num_steps} | Loss: {loss:.4f}")
        
        phase_time = time.time() - phase_start
        print(f"Phase 3 complete: {phase_time:.2f}s, Final loss: {loss:.4f}")
        return phase_time
    
    def train(self, phase1_steps: int = 30, phase2_gens: int = 3, phase3_steps: int = 20,
              samples_per_step: int = 100, verbose: bool = False) -> dict:
        """Full phased training."""
        print("=" * 60)
        print("SPEED EVOLVE TRAINING (Phased)")
        print(f"Phase 1: {phase1_steps} speed steps")
        print(f"Phase 2: {phase2_gens} evolution generations")
        print(f"Phase 3: {phase3_steps} fine-tune steps")
        print("=" * 60)
        
        total_start = time.time()
        
        p1_time = self.phase1_speed_train(phase1_steps, samples_per_step, verbose)
        p2_time = self.phase2_evolve(phase2_gens, samples_per_step * 2, verbose)
        p3_time = self.phase3_finetune(phase3_steps, samples_per_step, verbose)
        
        total_time = time.time() - total_start
        total_steps = phase1_steps + phase3_steps
        total_samples = total_steps * samples_per_step + phase2_gens * samples_per_step * 2
        
        # Final fitness
        test_data = generate_dream_data(50)
        final_fitness = evaluate_fitness(self.model, test_data)
        
        results = {
            'experiment': 'speed_evolve',
            'num_workers': self.num_workers,
            'population_size': self.population_size,
            'phase1_steps': phase1_steps,
            'phase2_gens': phase2_gens,
            'phase3_steps': phase3_steps,
            'total_time': total_time,
            'phase1_time': p1_time,
            'phase2_time': p2_time,
            'phase3_time': p3_time,
            'total_samples': total_samples,
            'avg_samples_per_sec': total_samples / total_time if total_time > 0 else 0,
            'final_loss': self.history['phase3'][-1]['loss'] if self.history['phase3'] else 0,
            'final_fitness': final_fitness,
            'history': self.history
        }
        
        print("\n" + "=" * 60)
        print("SPEED EVOLVE RESULTS")
        print("=" * 60)
        print(f"Total time: {total_time:.2f}s")
        print(f"  Phase 1: {p1_time:.2f}s")
        print(f"  Phase 2: {p2_time:.2f}s")
        print(f"  Phase 3: {p3_time:.2f}s")
        print(f"Final loss: {results['final_loss']:.4f}")
        print(f"Final fitness: {final_fitness:.4f}")
        
        return results
    
    def save_results(self, path: str):
        """Save results to JSON."""
        test_data = generate_dream_data(50)
        final_fitness = evaluate_fitness(self.model, test_data)
        
        results = {
            'experiment': 'speed_evolve',
            'final_loss': self.history['phase3'][-1]['loss'] if self.history['phase3'] else 0,
            'final_fitness': final_fitness,
            'history': self.history
        }
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Speed Evolve Training")
    parser.add_argument('--workers', type=int, default=10, help='Number of workers')
    parser.add_argument('--population', type=int, default=5, help='Population size')
    parser.add_argument('--p1-steps', type=int, default=30, help='Phase 1 steps')
    parser.add_argument('--p2-gens', type=int, default=3, help='Phase 2 generations')
    parser.add_argument('--p3-steps', type=int, default=20, help='Phase 3 steps')
    parser.add_argument('--samples', type=int, default=100, help='Samples per step')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    trainer = SpeedEvolveTrainer(num_workers=args.workers, population_size=args.population)
    results = trainer.train(
        phase1_steps=args.p1_steps,
        phase2_gens=args.p2_gens,
        phase3_steps=args.p3_steps,
        samples_per_step=args.samples,
        verbose=args.verbose
    )
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    trainer.save_results(os.path.join(output_dir, 'speed_evolve_results.json'))
