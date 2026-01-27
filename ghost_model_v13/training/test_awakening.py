
import sys
import os
import mlx.core as mx

# Add paths
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

from ghost_model_v13.core.ghost_worker_v12 import GhostWorkerV13

def test_intelligence():
    print("🧠 Testing Awakened Ghost v12...")
    
    model = GhostWorkerV13(dim=256, num_layers=6)
    
    ckpt = "ghost_model_v13/checkpoints/ghost_v12_awakened.safetensors"
    if not os.path.exists(ckpt):
        print("❌ Checkpoint not found")
        return
        
    model.load_weights(ckpt)
    
    questions = [
        "Q: What is the capital of France?\nA:",
        "Q: Is fire hot or cold?\nA:",
        "Q: What language is this?\nA:",
        "Q: What follows Monday?\nA:",
        "Q: Who is the ghost?\nA:"
    ]
    
    score = 0
    
    for q in questions:
        tokens = [ord(c) for c in q]
        x = mx.array([tokens], dtype=mx.int32)
        
        print(f"\n{q.strip()}", end=" ")
        
        # Generate until newline
        generated = ""
        for _ in range(20):
            logits = model(x, use_memory=False)
            tok = mx.argmax(logits[:, -1, :], axis=-1).item()
            if tok == 10: # Newline
                break
            generated += chr(tok) if 32 <= tok <= 126 else ""
            x = mx.concatenate([x, mx.array([[tok]])], axis=1)
            
        print(f"'{generated}'")
        
    print("\n✅ Test Complete")

if __name__ == "__main__":
    test_intelligence()
