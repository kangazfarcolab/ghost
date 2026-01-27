"""
Run All Experiments - Compare Results
======================================
Runs all 4 swarm training experiments and compares their results.
"""

import sys
import os
import time
import json

# Add paths
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, 'experiments'))

def run_experiment(name: str, module_path: str, trainer_class: str, **kwargs):
    """Run a single experiment and return results."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {name}")
    print(f"{'='*60}\n")
    
    # Dynamic import
    import importlib.util
    spec = importlib.util.spec_from_file_location("experiment", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get trainer class
    TrainerClass = getattr(module, trainer_class)
    
    # Run training
    start = time.time()
    trainer = TrainerClass(**kwargs.get('init', {}))
    results = trainer.train(**kwargs.get('train', {}))
    elapsed = time.time() - start
    
    results['total_wall_time'] = elapsed
    
    return results


def compare_results(all_results: dict):
    """Print comparison of all experiments."""
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)
    
    # Table header
    print(f"\n{'Experiment':<20} {'Time (s)':<12} {'Final Loss':<12} {'Speed (s/s)':<12} {'Fitness':<12}")
    print("-" * 68)
    
    for name, results in all_results.items():
        time_s = results.get('total_wall_time', results.get('total_time', 0))
        loss = results.get('final_loss', 0)
        speed = results.get('avg_samples_per_sec', 0)
        fitness = results.get('final_best_fitness', results.get('final_avg_fitness', 0))
        
        print(f"{name:<20} {time_s:<12.2f} {loss:<12.4f} {speed:<12.1f} {fitness:<12.4f}")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Find best in each category
    best_speed = max(all_results.items(), key=lambda x: x[1].get('avg_samples_per_sec', 0))
    best_loss = min(all_results.items(), key=lambda x: x[1].get('final_loss', float('inf')))
    best_fitness = max(all_results.items(), key=lambda x: x[1].get('final_best_fitness', x[1].get('final_avg_fitness', 0)))
    
    print(f"\n🏃 FASTEST: {best_speed[0]} ({best_speed[1].get('avg_samples_per_sec', 0):.1f} samples/sec)")
    print(f"🎯 BEST LOSS: {best_loss[0]} ({best_loss[1].get('final_loss', 0):.4f})")
    print(f"💪 BEST FITNESS: {best_fitness[0]} ({best_fitness[1].get('final_best_fitness', best_fitness[1].get('final_avg_fitness', 0)):.4f})")


def main():
    """Run all experiments and compare."""
    print("="*80)
    print("GHOST SWARM TRAINING EXPERIMENTS")
    print("="*80)
    
    experiments_dir = os.path.dirname(os.path.abspath(__file__))
    
    all_results = {}
    
    # 1. Speed Demon
    try:
        results = run_experiment(
            "Speed Demon",
            os.path.join(experiments_dir, 'speed_demon', 'train.py'),
            "SpeedDemonTrainer",
            init={'num_workers': 10},
            train={'num_steps': 50, 'samples_per_step': 100}
        )
        all_results['Speed Demon'] = results
    except Exception as e:
        print(f"Speed Demon failed: {e}")
        all_results['Speed Demon'] = {'error': str(e)}
    
    # 2. Precision Master
    try:
        results = run_experiment(
            "Precision Master",
            os.path.join(experiments_dir, 'precision_master', 'train.py'),
            "PrecisionMasterTrainer",
            init={'num_workers': 10},
            train={'num_steps': 50, 'samples_per_step': 100}
        )
        all_results['Precision Master'] = results
    except Exception as e:
        print(f"Precision Master failed: {e}")
        all_results['Precision Master'] = {'error': str(e)}
    
    # 3. Self-Improving
    try:
        results = run_experiment(
            "Self-Improving",
            os.path.join(experiments_dir, 'self_improving', 'train.py'),
            "SelfImprovingTrainer",
            init={'population_size': 10, 'elite_ratio': 0.2},
            train={'num_generations': 5, 'steps_per_generation': 10, 'samples_per_step': 50}
        )
        all_results['Self-Improving'] = results
    except Exception as e:
        print(f"Self-Improving failed: {e}")
        all_results['Self-Improving'] = {'error': str(e)}
    
    # 4. Swarm Forge
    try:
        results = run_experiment(
            "Swarm Forge",
            os.path.join(experiments_dir, 'swarm_forge', 'train.py'),
            "SwarmForgeTrainer",
            init={'num_workers': 10, 'elite_ratio': 0.2, 'evolution_interval': 25},
            train={'num_steps': 50, 'samples_per_step': 100}
        )
        all_results['Swarm Forge'] = results
    except Exception as e:
        print(f"Swarm Forge failed: {e}")
        all_results['Swarm Forge'] = {'error': str(e)}
    
    # Compare results
    compare_results(all_results)
    
    # Save combined results
    with open(os.path.join(experiments_dir, 'comparison_results.json'), 'w') as f:
        # Filter out non-serializable items
        serializable = {}
        for name, res in all_results.items():
            serializable[name] = {k: v for k, v in res.items() if k != 'history' and k != 'evolution_history'}
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to {os.path.join(experiments_dir, 'comparison_results.json')}")


if __name__ == "__main__":
    main()
