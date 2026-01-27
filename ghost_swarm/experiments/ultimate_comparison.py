#!/usr/bin/env python3
"""
Compare all swarm approaches:
1. Baseline Traditional
2. SwarmMomentum (consensus LR)
3. SwarmLookAhead (consensus + validation check)
"""

import subprocess
import json
import time
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

STEPS = 15
SAMPLES = 50

experiments = [
    ('Baseline', 'baseline_traditional', 
     ['python3', 'train.py', '--steps', str(STEPS), '--samples', str(SAMPLES)]),
    ('SwarmMomentum', 'swarm_momentum', 
     ['python3', 'train.py', '--workers', '8', '--steps', str(STEPS), '--samples', str(SAMPLES)]),
    ('SwarmLookAhead', 'swarm_lookahead', 
     ['python3', 'train.py', '--workers', '8', '--steps', str(STEPS), '--samples', str(SAMPLES)]),
]

results = {}

print("=" * 70)
print("ULTIMATE SWARM COMPARISON")
print(f"Config: {STEPS} steps, {SAMPLES} samples/step")
print("=" * 70)

for name, dir_name, cmd in experiments:
    print(f"\n{'='*50}")
    print(f"Running: {name}")
    print("=" * 50)
    
    start = time.time()
    try:
        result = subprocess.run(cmd, cwd=dir_name, capture_output=True, text=True, timeout=600)
        lines = result.stdout.strip().split('\n')
        for line in lines[-10:]:
            print(line)
        if result.returncode != 0 and result.stderr:
            print("STDERR:", result.stderr[-300:])
    except Exception as e:
        print(f"Error: {e}")
        continue
    
    wall_time = time.time() - start
    
    result_files = {
        'baseline_traditional': 'baseline_results.json',
        'swarm_momentum': 'swarm_momentum_results.json',
        'swarm_lookahead': 'swarm_lookahead_results.json'
    }
    
    result_file = os.path.join(dir_name, result_files[dir_name])
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
            data['wall_time'] = wall_time
            results[name] = data

# Summary
print("\n" + "=" * 70)
print("ULTIMATE COMPARISON")
print("=" * 70)
print(f"\n{'Method':<20} {'Time':>10} {'Loss':>12} {'Accept%':>10}")
print("-" * 55)

for name in ['Baseline', 'SwarmMomentum', 'SwarmLookAhead']:
    if name not in results:
        continue
    data = results[name]
    t = data.get('wall_time', 0)
    loss = data.get('final_loss', 999)
    accept = data.get('accept_rate', 1.0)
    print(f"{name:<20} {t:>10.1f}s {loss:>12.4f} {accept:>10.1%}")

print("-" * 55)

# Find winner
if 'Baseline' in results:
    b_loss = results['Baseline'].get('final_loss', 999)
    for name in ['SwarmMomentum', 'SwarmLookAhead']:
        if name in results:
            loss = results[name].get('final_loss', 999)
            if loss < b_loss:
                improvement = (b_loss - loss) / b_loss * 100
                print(f"🏆 {name} beats Baseline by {improvement:.1f}%")

with open('ultimate_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to ultimate_comparison.json")
