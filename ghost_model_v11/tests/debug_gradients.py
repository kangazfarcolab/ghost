
import sys
import os
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# Add project root to path
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

from ghost_model_v11.core.ghost_worker_v11 import GhostWorkerV11
# Removed GhostConfig import

class GhostConfig:
    dim = 256
    num_layers = 6
    vocab_size = 256 # Bytes
    memory_size = 512

def debug_gradients():
    print("🕵️ Debugging Gradients...")
    
    # 1. Initialize Model
    config = GhostConfig()
    model = GhostWorkerV11(dim=config.dim, num_layers=config.num_layers, memory_size=config.memory_size)
    
    # 2. Create Dummy Batch
    B, L = 2, 128
    x = mx.random.randint(0, 256, (B, L))
    y = mx.random.randint(0, 256, (B, L))
    
    # 3. Define Loss Function
    def loss_fn(params, x, y):
        model.update(params)
        logits = model(x, use_memory=False, use_routing=True)
        return nn.losses.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1), reduction='mean')
        
    # 4. Compute Gradients
    loss_and_grad = mx.value_and_grad(loss_fn)
    (loss), grads = loss_and_grad(model.trainable_parameters(), x, y)
    
    print(f"📉 Loss: {loss.item()}")
    
    # 5. Inspect Gradients
    print("\n📊 Gradient Norms by Layer:")
    flat_grads = tree_flatten(grads)
    
    max_norm = 0.0
    problem_layer = None
    
    for path, g in flat_grads:
        norm = mx.linalg.norm(g).item()
        if norm > max_norm:
            max_norm = norm
            problem_layer = path
            
        status = "✅"
        if norm > 10.0: status = "⚠️ High"
        if norm > 100.0: status = "🚨 EXPLODING"
        if mx.isnan(mx.array(norm)): status = "💀 NaN"
        
        # Only print significant or problem layers
        if norm > 1.0 or "NaN" in status:
            print(f"{status} {path}: {norm:.4f}")
            
    print(f"\n🏆 Max Gradient: {max_norm:.4f} in {problem_layer}")

if __name__ == "__main__":
    debug_gradients()
