
"""
Ghost v11 - Comprehensive Benchmark Suite
=========================================
Measures Performance, Speed, and Cognitive Capabilities.
"""

import os
import sys
import time
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# Add paths
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

from ghost_model_v11.core.ghost_worker_v11 import GhostWorkerV11
from ghost_model_v11.training.train_perceptual import SwarmTrainerOptimized, Config as TrainConfig

def print_header(title):
    print("\n" + "="*50)
    print(f" {title}")
    print("="*50)

def benchmark_system():
    print_header("1. System & Model Stats")
    print(f"Device: {mx.default_device()}")
    
    model = GhostWorkerV11(dim=256, num_layers=6, memory_size=512)
    mx.eval(model.parameters())
    
    params = model.count_params()
    print(f"Model: Ghost v11 (Ternary Mamba)")
    print(f"Params: {params:,}")
    print(f"Memory: 512 slots (Cognitive)")
    return model

def benchmark_training_speed():
    print_header("2. Training Throughput (Swarm Optimized)")
    
    cfg = TrainConfig()
    cfg.use_synthetic = True # Use synthetic for fair benchmarking vs hardware
    cfg.steps = 50
    
    trainer = SwarmTrainerOptimized(cfg)
    
    
    # Warmup
    print("Warmup...")
    bx_dummy = mx.random.randint(0, 256, (cfg.num_workers, cfg.batch_size_per_worker, cfg.seq_len))
    by_dummy = mx.random.randint(0, 256, (cfg.num_workers, cfg.batch_size_per_worker, cfg.seq_len))
    trainer.step(bx_dummy, by_dummy)
    mx.eval(trainer.model.parameters())
    
    # We'll use the trainer's internal loop via a modified Config or direct call
    # Since we can't easily modify the class instance method on the fly if it's not designed for it,
    # let's just instantiate and run a custom loop using trainer parts.
    
    print(f"Running {cfg.steps} steps...")
    start = time.time()
    
    # Synthetic batch
    shape = (cfg.num_workers, cfg.batch_size_per_worker, cfg.seq_len)
    bx = mx.random.randint(0, 256, shape)
    by = mx.random.randint(0, 256, shape)
    
    for _ in range(cfg.steps):
        loss, cons, lr = trainer.step(bx, by)
        mx.eval(trainer.model.parameters(), trainer.optimizer.state)
        
    end = time.time()
    elapsed = end - start
    
    total_tokens = cfg.steps * cfg.num_workers * cfg.batch_size_per_worker * cfg.seq_len
    tps = total_tokens / elapsed
    
    print(f"Training Speed: {tps:,.0f} tokens/sec")
    print(f"Swarm Consensus: Enabled (4 workers)")
    
    return tps

def benchmark_inference_speed(model):
    print_header("3. Inference Latency")
    
    prompt = mx.array([1, 2, 3]).reshape(1, 3) # Batch 1, Len 3
    
    # Fast Generate uses cache usually, but current implementation might not have KV cache for Mamba fully exposed?
    # GhostWorkerV11.generate uses simple autoregressive loop.
    
    print("Generating 100 tokens...")
    start = time.time()
    
    # Simple generation loop
    # Fast Generation (KV Cache)
    print(f"Generating 100 tokens using {model.generate_step.__name__}...")
    start = time.time()
    
    # 1. Prefill (Simple linear prefill for benchmark)
    cache = None
    logits = None
    for i in range(prompt.shape[1]):
        token = prompt[:, i:i+1]
        logits, cache = model.generate_step(token, cache)
        
    # 2. Generate
    curr_token = mx.argmax(logits[:, -1, :], axis=-1).reshape(1, 1)
    
    for _ in range(99): # 1 token already generated above effectively or just starting loop
        logits, cache = model.generate_step(curr_token, cache)
        curr_token = mx.argmax(logits[:, -1, :], axis=-1).reshape(1, 1)
        mx.eval(curr_token)
        
    end = time.time()
    elapsed = end - start
    tps = 100 / elapsed
    lat = (elapsed / 100) * 1000
    
    print(f"Inference Speed: {tps:.1f} tokens/sec")
    print(f"Latency: {lat:.1f} ms/token")
    
    return tps

def benchmark_cognitive_capabilities(model):
    print_header("4. Cognitive Capabilities (The 'Ghost' Check)")
    
    # 1. One-Shot Learning
    print(">> Testing One-Shot Memory...")
    model.clear_memory()
    
    # Create a "Memory" - a random vector that acts as a key thought
    thought = mx.random.normal((1, 1, 256))
    model._store_memory(thought, curiosity_score=mx.array([[1.0]])) # Force store
    
    if model.memory.size > 0:
        print("   [PASS] Memory Storage (One-Shot)")
    else:
        print("   [FAIL] Memory Storage")
        
    # 2. Retrieval
    query_concept = thought + mx.random.normal((1, 1, 256)) * 0.1 # Noisy version
    
    # We need to flatten or index properly because recall_associative expects [D]
    # Our thought is [1, 1, 256].
    q_c = query_concept.squeeze()
    
    retrieved, conf = model.memory.recall_associative(q_c, query_context=None)
    
    if conf > 0.5: # Arbitrary high confidence threshold for near match
         print(f"   [PASS] Memory Retrieval (Conf: {conf:.2f})")
    else:
         print(f"   [FAIL] Memory Retrieval (Conf: {conf:.2f})")
         
    return True

if __name__ == "__main__":
    print_header("👻 Ghost v11 BENCHMARK 👻")
    
    model = benchmark_system()
    benchmark_training_speed()
    benchmark_inference_speed(model)
    benchmark_cognitive_capabilities(model)
    
    print("\n" + "="*50)
    print("✅ Benchmark Complete")
    print("="*50)
