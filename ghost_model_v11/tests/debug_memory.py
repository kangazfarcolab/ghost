
import sys
import os
import mlx.core as mx

# Add project root to path
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

from ghost_model_v11.core.cognitive_memory import CognitiveMemory

def debug_memory():
    print("🧠 Debugging Cognitive Memory")
    
    dim = 256
    mem = CognitiveMemory(dim, max_entries=10)
    
    # 1. Create Vectors
    # Normalize them to ensure dot product is cosine similarity directly
    concept = mx.random.normal((dim,))
    concept = concept / mx.linalg.norm(concept)
    
    context = mx.random.normal((dim,))
    context = context / mx.linalg.norm(context)
    
    outcome = mx.random.normal((dim,))
    
    print(f"Concept Norm: {mx.linalg.norm(concept)}")
    
    # 2. Store
    mem.store_one_shot(concept, context, outcome)
    print("Stored memory.")
    
    # 3. Retrieve with exact match
    print("\n--- Exact Match Test ---")
    retrieved, conf = mem.recall_associative(concept, query_context=None)
    print(f"Confidence (Exact): {conf}")
    
    # 4. Debug Internals
    # Replicate calculation manually
    stored_concept = mem.concepts[0]
    dot = mx.sum(concept * stored_concept).item()
    print(f"Manual Dot Product: {dot}")
    
    # Check scaling in recall_associative
    # The code does: concept_scores = concept_scores / math.sqrt(self.dim)
    # If vectors are normalized, dot is 1.0. 
    # score = 1.0 / 16 = 0.0625
    import math
    expected_score = dot / math.sqrt(dim)
    print(f"Expected Score (scaled): {expected_score}")
    
    if abs(conf - expected_score) < 0.001:
        print(">> Logic Confirmed: The raw confidence score is scaled by sqrt(dim).")
        print("   This makes it non-intuitive (0.06 instead of 1.0).")
    
    # 5. Retrieve with Noisy match
    print("\n--- Noisy Match Test ---")
    noise = mx.random.normal((dim,)) * 0.1
    query = concept + noise
    query = query / mx.linalg.norm(query)
    
    retrieved, conf = mem.recall_associative(query)
    print(f"Confidence (Noisy): {conf}")

if __name__ == "__main__":
    debug_memory()
