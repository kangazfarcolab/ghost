#!/usr/bin/env python3
"""
Compare Swarm Momentum vs Baseline Traditional
Metrics: Speed, Final Loss, Memory
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
]

results = {}

print("=" * 70)
print("SWARM MOMENTUM vs BASELINE COMPARISON")
print(f"Config: {STEPS} steps, {SAMPLES} samples/step")
print("=" * 70)

for name, dir_name, cmd in experiments:
    print(f"\n{'='*50}")
    print(f"Running: {name}")
    print("=" * 50)
    
    start = time.time()
    try:
        result = subprocess.run(cmd, cwd=dir_name, capture_output=True, text=True, timeout=600)
        # Print last part of output
        lines = result.stdout.strip().split('\n')
        for line in lines[-15:]:
            print(line)
        if result.stderr:
            print("STDERR:", result.stderr[-200:])
    except subprocess.TimeoutExpired:
        print("TIMEOUT!")
        continue
    except Exception as e:
        print(f"Error: {e}")
        continue
    
    wall_time = time.time() - start
    
    # Load results
    result_files = {
        'baseline_traditional': 'baseline_results.json',
        'swarm_momentum': 'swarm_momentum_results.json'
    }
    
    result_file = os.path.join(dir_name, result_files[dir_name])
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
            data['wall_time'] = wall_time
            results[name] = data

# Summary
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print(f"{'Metric':<20} {'Baseline':>15} {'SwarmMomentum':>15} {'Winner':>12}")
print("-" * 70)

if 'Baseline' in results and 'SwarmMomentum' in results:
    b = results['Baseline']
    s = results['SwarmMomentum']
    
    # Speed
    b_speed = b.get('avg_samples_per_sec', b.get('total_samples', 0) / b.get('wall_time', 1))
    s_speed = s.get('avg_samples_per_sec', s.get('total_samples', 0) / s.get('wall_time', 1))
    speed_winner = 'SwarmMomentum' if s_speed > b_speed else 'Baseline'
    print(f"{'Speed (samples/s)':<20} {b_speed:>15.1f} {s_speed:>15.1f} {speed_winner:>12}")
    
    # Loss
    b_loss = b.get('final_loss', 999)
    s_loss = s.get('final_loss', 999)
    loss_winner = 'SwarmMomentum' if s_loss < b_loss else 'Baseline'
    print(f"{'Final Loss':<20} {b_loss:>15.4f} {s_loss:>15.4f} {loss_winner:>12}")
    
    # Time
    b_time = b.get('wall_time', 0)
    s_time = s.get('wall_time', 0)
    time_winner = 'SwarmMomentum' if s_time < b_time else 'Baseline'
    print(f"{'Wall Time (s)':<20} {b_time:>15.2f} {s_time:>15.2f} {time_winner:>12}")
    
    # Speedup
    speedup = s_speed / b_speed if b_speed > 0 else 0
    print("-" * 70)
    print(f"Speedup: {speedup:.2f}x")
    print(f"Avg Consensus: {s.get('avg_consensus', 0):.2f}")

# Save comparison
with open('swarm_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to swarm_comparison.json")
