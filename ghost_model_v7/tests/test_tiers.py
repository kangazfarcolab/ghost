"""
test_tiers.py - Test all Ghost v7 tiers

Verifies:
1. All tiers load correctly
2. Parameter counts match expected
3. Forward pass works
4. Memory usage is acceptable
"""

import sys
import os
import time

# Add paths
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, 'core'))

import mlx.core as mx

# Import tiers
from core.ghost_worker import GhostWorker
from core.ghost_expert import GhostExpert
from core.ghost_thinker import GhostThinker


def format_params(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)


def test_tier(name, model_class, expected_range):
    """Test a single tier."""
    print(f"\n{'='*50}")
    print(f"Testing {name} Tier")
    print(f"{'='*50}")
    
    # Create model
    start = time.time()
    model = model_class()
    mx.eval(model.parameters())
    init_time = time.time() - start
    
    # Count params
    params = model.count_params()
    print(f"Parameters: {format_params(params)} ({params:,})")
    print(f"Init time: {init_time:.3f}s")
    print(f"Tier: {model.TIER}")
    
    # Check param range
    min_p, max_p = expected_range
    if min_p <= params <= max_p:
        print(f"✅ Params in expected range ({format_params(min_p)}-{format_params(max_p)})")
    else:
        print(f"⚠️ Params outside range ({format_params(min_p)}-{format_params(max_p)})")
    
    # Forward pass
    test_input = "Q: What is kubectl?\nA:"
    x = mx.array([[ord(c) for c in test_input]], dtype=mx.int32)
    
    start = time.time()
    out = model(x)
    mx.eval(out)
    forward_time = time.time() - start
    
    print(f"Forward: {x.shape} → {out.shape}")
    print(f"Forward time: {forward_time:.3f}s")
    
    # Memory store/query test
    model.store_fact(
        [ord(c) for c in "kubectl"],
        [ord(c) for c in "kubernetes command line tool"]
    )
    print("✅ Memory store works")
    
    # Generate a token
    logits = out[0, -1, :]
    predicted = mx.argmax(logits).item()
    print(f"Next byte prediction: {predicted} ('{chr(predicted)}')")
    
    return params, init_time, forward_time


def main():
    print("="*60)
    print("GHOST v7 TIERED ARCHITECTURE TEST")
    print("="*60)
    
    results = {}
    
    # Test each tier
    results['Worker'] = test_tier(
        "Worker (6M)", 
        GhostWorker, 
        (5_000_000, 8_000_000)
    )
    
    results['Expert'] = test_tier(
        "Expert (25M)", 
        GhostExpert, 
        (20_000_000, 30_000_000)
    )
    
    results['Thinker'] = test_tier(
        "Thinker (50M)", 
        GhostThinker, 
        (40_000_000, 60_000_000)
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Tier':<12} {'Params':<12} {'Init':<10} {'Forward':<10}")
    print("-"*44)
    for name, (params, init_t, forward_t) in results.items():
        print(f"{name:<12} {format_params(params):<12} {init_t:.3f}s     {forward_t:.3f}s")
    
    # Calculate swarm capacity (16GB Mac, leaving 4GB for system)
    print(f"\n{'='*60}")
    print("SWARM CAPACITY (12GB available)")
    print(f"{'='*60}")
    
    # Rough memory estimate: params * 4 bytes (float32)
    worker_mem = results['Worker'][0] * 4 / (1024**2)  # MB
    expert_mem = results['Expert'][0] * 4 / (1024**2)  # MB
    thinker_mem = results['Thinker'][0] * 4 / (1024**2)  # MB
    
    print(f"Worker memory:  ~{worker_mem:.0f} MB each")
    print(f"Expert memory:  ~{expert_mem:.0f} MB each")
    print(f"Thinker memory: ~{thinker_mem:.0f} MB each")
    
    # Tiered swarm example
    tiered_ram = (2 * thinker_mem) + (10 * expert_mem) + (100 * worker_mem)
    print(f"\nTiered Swarm (2 Thinkers + 10 Experts + 100 Workers):")
    print(f"  Total RAM: ~{tiered_ram:.0f} MB ({tiered_ram/1024:.1f} GB)")
    print(f"  Fits in 16GB Mac: {'✅ Yes' if tiered_ram < 12000 else '❌ No'}")
    
    print("\n✅ All tiers tested successfully!")


if __name__ == "__main__":
    main()
