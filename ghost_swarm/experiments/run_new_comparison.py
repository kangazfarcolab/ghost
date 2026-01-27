#!/usr/bin/env python3
"""
Run all 4 new experiments and compare results.
"""

import subprocess
import json
import time
import os

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))

experiments = [
    {
        'name': 'Baseline Traditional',
        'dir': 'baseline_traditional',
        'cmd': ['python3', 'train.py', '--steps', '50', '--samples', '100']
    },
    {
        'name': 'Speed Vote',
        'dir': 'speed_vote',
        'cmd': ['python3', 'train.py', '--workers', '10', '--steps', '50', '--samples', '100', '--vote-interval', '10']
    },
    {
        'name': 'Speed Evolve',
        'dir': 'speed_evolve', 
        'cmd': ['python3', 'train.py', '--workers', '10', '--p1-steps', '30', '--p2-gens', '3', '--p3-steps', '20', '--samples', '100']
    },
    {
        'name': 'Swarm Forge V2',
        'dir': 'swarm_forge_v2',
        'cmd': ['python3', 'train.py', '--workers', '10', '--p1-steps', '25', '--p2-steps', '15', '--p3-gens', '3', '--samples', '100']
    }
]

results = {}

print("=" * 70)
print("NEW EXPERIMENTS COMPARISON BENCHMARK")
print("=" * 70)

for exp in experiments:
    print(f"\n{'='*70}")
    print(f"Running: {exp['name']}")
    print("=" * 70)
    
    exp_dir = os.path.join(EXPERIMENTS_DIR, exp['dir'])
    
    start = time.time()
    result = subprocess.run(exp['cmd'], cwd=exp_dir, capture_output=True, text=True)
    wall_time = time.time() - start
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Load results file
    result_files = {
        'baseline_traditional': 'baseline_results.json',
        'speed_vote': 'speed_vote_results.json',
        'speed_evolve': 'speed_evolve_results.json',
        'swarm_forge_v2': 'swarm_forge_v2_results.json'
    }
    
    result_file = os.path.join(exp_dir, result_files[exp['dir']])
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
            data['total_wall_time'] = wall_time
            results[exp['name']] = data

# Save comparison
output_path = os.path.join(EXPERIMENTS_DIR, 'new_comparison_results.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

for name, data in results.items():
    print(f"\n{name}:")
    print(f"  Final Loss: {data.get('final_loss', 'N/A'):.4f}" if data.get('final_loss') else "  Final Loss: N/A")
    print(f"  Final Fitness: {data.get('final_fitness', 'N/A'):.4f}" if data.get('final_fitness') else "  Final Fitness: N/A")
    print(f"  Wall Time: {data.get('total_wall_time', 0):.2f}s")

print(f"\nResults saved to: {output_path}")
