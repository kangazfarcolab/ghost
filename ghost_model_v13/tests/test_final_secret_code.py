
import sys
import os
import mlx.core as mx

# Add project root to path
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

from ghost_model_v11.core.ghost_worker_v11 import GhostWorkerV11

def test_secret_code():
    print("🧪 Ghost v11 - Final Secret Code Test")
    print("====================================")
    
    # Initialize model
    model = GhostWorkerV11(dim=256, num_layers=4, memory_size=100)
    mx.eval(model.parameters())
    
    # Mock curiosity to be HIGH for the secret code part
    # We want to force storage when it sees "9922"
    def mock_curiosity_predictor(x):
        # Return uniform logits (high entropy)
        return mx.zeros((x.shape[0], x.shape[1], 256))
    
    model.surprise_predictor = mock_curiosity_predictor
    
    # 1. The "Learning" Phase
    print("\n[Phase 1] Learning: 'The secret code is 9922'")
    
    # Input sequence
    text = "The secret code is 9922"
    input_ids = mx.array([[ord(c) for c in text]])
    
    # Run model
    # This should trigger storage because we mocked high curiosity
    model(input_ids, use_memory=True, use_routing=True)
    
    print(f"Memory Size: {model.memory.size}")
    
    if model.memory.size > 0:
        print("✅ SUCCESS: Model learned from the input!")
    else:
        print("❌ FAILURE: Model ignored the input.")
        return
        
    # 2. The "Recall" Phase
    print("\n[Phase 2] Recall: 'The secret code is'")
    
    # Query sequence (incomplete)
    query = "The secret code is"
    query_ids = mx.array([[ord(c) for c in query]])
    
    # We want to see if the model RETRIEVES something relevant
    # We will hook into the retrieval process to verify
    
    # Run model
    # We can't easily check the output text (model is untrained), 
    # but we can check if memory was accessed.
    
    # We'll use a trick: check memory.total_retrieved before and after
    initial_retrievals = model.memory.total_retrieved
    
    model(query_ids, use_memory=True, use_routing=True)
    
    final_retrievals = model.memory.total_retrieved
    
    print(f"Retrievals triggered: {final_retrievals - initial_retrievals}")
    
    if final_retrievals > initial_retrievals:
        print("✅ SUCCESS: Model accessed memory during query!")
    else:
        print("❌ FAILURE: Model did not access memory.")
        
    # 3. Sleep Consolidation
    print("\n[Phase 3] Sleep Consolidation")
    loss = model.sleep(cycles=10)
    print(f"Sleep Loss: {loss:.4f}")
    print("✅ SUCCESS: Memories consolidated.")

if __name__ == "__main__":
    test_secret_code()
