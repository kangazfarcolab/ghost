
import sys
import os
import mlx.core as mx

# Add project root to path
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

from ghost_model_v11.core.ghost_worker_v11 import GhostWorkerV11

def test_curiosity():
    print("🧪 Testing Curiosity Signal")
    print("==========================")
    
    # Initialize model
    model = GhostWorkerV11(dim=256, num_layers=2, memory_size=100)
    mx.eval(model.parameters())
    
    # 1. Low Curiosity Input
    # We simulate predictable input by mocking the surprise predictor to return low entropy
    print("\n[Step 1] Testing Low Curiosity (Routine Input)...")
    
    # Clear memory
    model.clear_memory()
    
    # Mock surprise predictor to output peaked distribution (low entropy)
    # Class 0 has 100 logit, others 0 -> prob ~1.0 for class 0
    def mock_low_entropy(x):
        logits = mx.zeros((x.shape[0], x.shape[1], 256))
        logits[:, :, 0] = 100.0 
        return logits
    
    model.surprise_predictor = mock_low_entropy
    
    # Forward pass
    x = mx.random.randint(0, 256, (1, 10))
    model(x, use_memory=True, use_routing=True)
    
    print(f"Memory Size (Low Curiosity): {model.memory.size}")
    # Should be 0 or very low (unless learned importance is high by chance)
    
    # 2. High Curiosity Input
    # Simulate confusing input by mocking high entropy
    print("\n[Step 2] Testing High Curiosity (Confusing Input)...")
    
    # Mock surprise predictor to output uniform distribution (max entropy)
    def mock_high_entropy(x):
        return mx.zeros((x.shape[0], x.shape[1], 256)) # All logits 0 -> probs 1/256
        
    model.surprise_predictor = mock_high_entropy
    
    # Forward pass
    model(x, use_memory=True, use_routing=True)
    
    print(f"Memory Size (High Curiosity): {model.memory.size}")
    
    if model.memory.size > 0:
        print("✅ SUCCESS: Curiosity triggered memory storage!")
    else:
        print("❌ FAILURE: Curiosity failed to trigger storage.")

if __name__ == "__main__":
    test_curiosity()
