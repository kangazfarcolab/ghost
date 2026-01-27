
import sys
import os
import mlx.core as mx

# Add project root to path
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

from ghost_model_v11.core.ghost_worker_v11 import GhostWorkerV11

def test_sleep_learning():
    print("🧪 Testing Sleep Learning (Contrastive Consolidation)")
    print("==================================================")
    
    # Initialize model
    model = GhostWorkerV11(dim=256, num_layers=2, memory_size=100)
    mx.eval(model.parameters())
    
    # 1. Fill Memory with Random "Experiences"
    print("\n[Step 1] Filling Memory...")
    
    for i in range(20):
        c = mx.random.normal((256,))
        ctx = mx.random.normal((256,))
        o = mx.random.normal((256,))
        model.memory.store_one_shot(c, ctx, o, surprise_score=0.8)
        
    print(f"Memory Size: {model.memory.size}")
    
    # 2. Trigger Sleep
    print("\n[Step 2] Sleeping...")
    
    loss = model.sleep(cycles=50)
    
    print(f"Sleep Loss: {loss:.4f}")
    
    if loss >= 0.0:
        print("✅ SUCCESS: Sleep learning executed successfully!")
    else:
        print("❌ FAILURE: Sleep learning failed.")

if __name__ == "__main__":
    test_sleep_learning()
