#!/usr/bin/env python3
"""
Quick benchmark of all new experiments with minimal config.
"""

import subprocess
import json
import time
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

experiments = [
    ('Baseline', 'baseline_traditional', ['python3', 'train.py', '--steps', '10', '--samples', '30']),
    ('SpeedVote', 'speed_vote', ['python3', 'train.py', '--workers', '5', '--steps', '10', '--samples', '30']),
    ('SpeedEvolve', 'speed_evolve', ['python3', 'train.py', '--workers', '5', '--p1-steps', '5', '--p2-gens', '2', '--p3-steps', '5', '--samples', '30']),
    ('SwarmForgeV2', 'swarm_forge_v2', ['python3', 'train.py', '--workers', '5', '--p1-steps', '5', '--p2-steps', '5', '--p3-gens', '2', '--samples', '30']),
]

results = {}

for name, dir_name, cmd in experiments:
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print("=" * 60)
    
    start = time.time()
    try:
        result = subprocess.run(cmd, cwd=dir_name, capture_output=True, text=True, timeout=300)
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[-500:])
    except subprocess.TimeoutExpired:
        print("TIMEOUT!")
        continue
    except Exception as e:
        print(f"Error: {e}")
        continue
    
    wall_time = time.time() - start
    results[name] = {'wall_time': wall_time, 'exit_code': result.returncode}
    print(f"Wall time: {wall_time:.2f}s")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name, data in results.items():
    print(f"{name}: {data['wall_time']:.2f}s (exit: {data['exit_code']})")
