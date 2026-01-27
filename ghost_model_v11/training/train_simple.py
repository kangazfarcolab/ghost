
"""
Ghost v11 - Simple Training Benchmark
=====================================
Standard AdamW training to verify model speed without Swarm overhead.
"""

import sys
import os
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Add paths
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

from ghost_model_v11.core.ghost_worker_v11 import GhostWorkerV11

class Config:
    dim = 256
    num_layers = 6
    memory_size = 512
    batch_size = 32
    seq_len = 256
    steps = 50
    lr = 1e-3

def run_benchmark():
    print("🚀 Running Simple Benchmark (No Swarm)...")
    
    model = GhostWorkerV11(dim=Config.dim, num_layers=Config.num_layers, memory_size=Config.memory_size)
    mx.eval(model.parameters())
    print(f"Params: {model.count_params():,}")
    
    optimizer = optim.AdamW(learning_rate=Config.lr)
    
    # Synthetic data
    x = mx.random.randint(0, 255, (Config.batch_size, Config.seq_len + 1))
    
    def loss_fn(m, x, y):
        logits = m(x, use_memory=False, use_routing=True)
        return nn.losses.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction='mean')
    
    state = [model.state, optimizer.state]
    
    # @mx.compile
    def step(x):
        input_seq = x[:, :-1]
        target_seq = x[:, 1:]
        loss, grads = mx.value_and_grad(loss_fn)(model, input_seq, target_seq)
        optimizer.update(model, grads)
        return loss
    
    start = time.time()
    for i in range(Config.steps):
        loss = step(x)
        mx.eval(state)
        
        if i % 10 == 0:
            print(f"Step {i} | Loss: {loss.item():.4f}")
            
    end = time.time()
    elapsed = end - start
    toks = Config.steps * Config.batch_size * Config.seq_len
    print(f"Done! {toks/elapsed:.0f} tok/s")

if __name__ == "__main__":
    run_benchmark()
