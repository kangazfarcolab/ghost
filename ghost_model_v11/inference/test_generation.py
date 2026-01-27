import mlx.core as mx
import mlx.nn as nn
import os
import sys

# Add path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ghost_model_v11.core.ghost_worker_v11 import GhostWorkerV11

def generate():
    print("👻 Loading Ghost v11 for Generation...")
    
    # Initialize Model (Same config as training)
    # dim=256, num_layers=6
    model = GhostWorkerV11(dim=256, num_layers=6)
    
    # Load Weights
    checkpoint_path = "ghost_model_v11/checkpoints/ghost_v11_perceptual_final.safetensors"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        # Try finding the step 900 checkpoint if final is missing
        return
        
    print(f"📂 Loading weights from {checkpoint_path}...")
    model.load_weights(checkpoint_path)
    model.eval()
    
    from mlx.utils import tree_flatten
    
    # 0. Check Weight Stats
    print("\n📊 Weight Diagnostics:")
    params = model.parameters() # Nested dict
    flat_params = tree_flatten(params) # List of (key, value)
    
    for name, p in flat_params:
        if "depth_predictor" in name or "conv1d" in name:
            print(f"   {name}: Mean={mx.mean(p).item():.4f}, Std={mx.std(p).item():.4f}, Max={mx.max(p).item():.4f}")
            break # Just check one
            
    # Prompt
    prompt = "Once upon a time"
    print(f"\n📝 Prompt: '{prompt}'")
    
    tokens = [ord(c) for c in prompt]
    x = mx.array([tokens], dtype=mx.int32) # [1, L]
    
    # Generate
    print("🤖 Generating...", end="\n", flush=True)
    
    for i in range(20): # Short run for debug
        logits = model(x, use_memory=False)
        next_token_logits = logits[:, -1, :] # [1, 256]
        
        # Softmax to see probs
        probs = mx.softmax(next_token_logits, axis=-1)
        
        # Softmax to see probs
        probs = mx.softmax(next_token_logits, axis=-1)
        
        # Get Top 3 manually using argsort
        # argsort sorts ascending, so we take LAST 3 and reverse
        all_indices = mx.argsort(probs, axis=-1)
        top_indices = all_indices[:, -3:][:, ::-1] # [1, 3]
        
        # Gather probabilities
        # mx.take_along_axis requires indices to have same dim as input except axis
        # We assume batch size 1
        top_probs = mx.take_along_axis(probs, top_indices, axis=-1)
        
        top_probs = top_probs[0]
        top_indices = top_indices[0]
        
        print(f"Step {i}: ", end="")
        for k in range(3):
            tok = top_indices[k].item()
            prob = top_probs[k].item()
            char = chr(tok) if 32 <= tok <= 126 else f"\\x{tok:02x}"
            print(f"'{char}'({prob:.2f}) ", end="")
        print("")
        
        # Greedy
        next_token = top_indices[0].item()
        tokens.append(next_token)
        x = mx.array([tokens], dtype=mx.int32)

        
    print("\n\n✅ Generation Complete")
    
    decoded = "".join([chr(t) for t in tokens])
    print(f"Full Output:\n{decoded}")

if __name__ == "__main__":
    generate()
