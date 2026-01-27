#!/usr/bin/env python3
"""
FAIR COMPARISON: Same Wall Time Budget
=======================================
Give both methods ~2 minutes of training time.
SwarmMomentum gets more steps → should have lower loss.
"""

import subprocess
import json
import time
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

TIME_BUDGET = 120  # 2 minutes each

print("=" * 70)
print("FAIR TEST: Same Wall Time Budget")
print(f"Each method gets ~{TIME_BUDGET}s of training")
print("=" * 70)

# Based on previous runs:
# Baseline: ~25s per step → ~5 steps in 120s
# SwarmMomentum: ~9s per step → ~13 steps in 120s

experiments = [
    ('Baseline', 'baseline_traditional', 
     ['python3', 'train.py', '--steps', '5', '--samples', '50']),
    ('SwarmMomentum', 'swarm_momentum', 
     ['python3', 'train.py', '--workers', '8', '--steps', '15', '--samples', '50', '--boost', '2.5']),
]

results = {}

for name, dir_name, cmd in experiments:
    print(f"\n{'='*50}")
    print(f"Running: {name}")
    print("=" * 50)
    
    start = time.time()
    try:
        result = subprocess.run(cmd, cwd=dir_name, capture_output=True, text=True, timeout=600)
        lines = result.stdout.strip().split('\n')
        for line in lines[-12:]:
            print(line)
    except Exception as e:
        print(f"Error: {e}")
        continue
    
    wall_time = time.time() - start
    
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
print("FAIR COMPARISON: SAME TIME BUDGET")
print("=" * 70)

if 'Baseline' in results and 'SwarmMomentum' in results:
    b = results['Baseline']
    s = results['SwarmMomentum']
    
    print(f"\n{'Method':<20} {'Steps':>10} {'Time':>10} {'Loss':>12}")
    print("-" * 55)
    
    b_steps = len(b.get('history', []))
    s_steps = len(s.get('history', []))
    b_time = b.get('wall_time', 0)
    s_time = s.get('wall_time', 0)
    b_loss = b.get('final_loss', 999)
    s_loss = s.get('final_loss', 999)
    
    print(f"{'Baseline':<20} {b_steps:>10} {b_time:>10.1f}s {b_loss:>12.4f}")
    print(f"{'SwarmMomentum':<20} {s_steps:>10} {s_time:>10.1f}s {s_loss:>12.4f}")
    
    print("-" * 55)
    
    if s_loss < b_loss:
        improvement = (b_loss - s_loss) / b_loss * 100
        print(f"🏆 SwarmMomentum WINS! {improvement:.1f}% better loss")
        print(f"   With {s_steps/b_steps:.1f}x more steps in similar time")
    else:
        gap = (s_loss - b_loss) / b_loss * 100
        print(f"⚠️ Baseline wins by {gap:.1f}% loss")
    
    print(f"\nConsensus: {s.get('avg_consensus', 0):.2f}")

with open('fair_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)
