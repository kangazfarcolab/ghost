"""
spawn_test.py - Verify Ghost Swarm Capacity

Goal: Initialize 100 Ghost v6 agents and measure memory usage.
"""

import sys
import os
import time
import psutil
import mlx.core as mx

# Add core path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from ghost_v6 import GhostV6

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    print("="*60)
    print("GHOST SWARM - CAPACITY TEST")
    print("="*60)
    
    initial_mem = get_memory_usage_mb()
    print(f"Initial Memory: {initial_mem:.2f} MB")
    
    agents = []
    target_agents = 100
    
    print(f"\n🚀 Spawning {target_agents} Agents...")
    
    start_time = time.time()
    for i in range(target_agents):
        # Create new independent model instance
        # In a real swarm, these might share some weights, but let's test worst case (independent)
        agent_model = GhostV6(dim=256, num_layers=6)
        
        # We perform one eval to ensure memory is allocated
        mx.eval(agent_model.parameters())
        
        agents.append(agent_model)
        
        if (i + 1) % 10 == 0:
            current_mem = get_memory_usage_mb()
            print(f"  Spawned {i+1} agents... Mem: {current_mem:.2f} MB (+{current_mem - initial_mem:.2f} MB)")
    
    end_time = time.time()
    final_mem = get_memory_usage_mb()
    total_model_mem = final_mem - initial_mem
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Agents Spawned: {len(agents)}")
    print(f"Total Time:     {end_time - start_time:.2f}s")
    print(f"Total Memory:   {final_mem:.2f} MB")
    print(f"Memory Check:   {total_model_mem:.2f} MB used for models")
    print(f"Per Agent:      {total_model_mem / len(agents):.2f} MB")
    print("="*60)
    
    if total_model_mem < 4096: # 4GB
        print("✅ SUCCESS: 100 Agents run easily on this machine!")
    else:
        print("⚠️ WARNING: High memory usage.")

if __name__ == "__main__":
    main()
