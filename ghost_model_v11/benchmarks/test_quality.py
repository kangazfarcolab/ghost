
"""
Ghost v11 - Quality Test
========================
Generates samples for 20 prompts to evaluate model quality.
"""

import os
import sys
import time
import mlx.core as mx

# Add paths
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

from ghost_model_v11.core.ghost_worker_v11 import GhostWorkerV11

def load_prompts():
    """20 Standard TinyStories prompts"""
    return [
        "Once upon a time,",
        "Lily was a little girl who",
        "Tom liked to play with",
        "One day, a big bear",
        "The sun was shining and",
        "Sara found a new",
        "In the middle of the forest,",
        "The dog barked at",
        "Mom said, 'Please do not",
        "Timmy wanted to buy",
        "There was a cat named",
        "The bird flew high in",
        "Grandma gave Ann a",
        "The box was very",
        "Under the bed, there was",
        "The sky turned dark and",
        "Alice had a red",
        "The old man walked to",
        "A little mouse ran",
        "Everyone was happy because"
    ]

def decode(tokens):
    """Simple ASCII decoding for v11 byte-level tokenizer"""
    # tokens is [1, L]
    t = tokens.flatten().tolist()
    return "".join([chr(c) if 32 <= c <= 126 else "" for c in t])

def test_quality():
    print("👻 Ghost v11 - Quality Evaluation")
    print("=================================")
    
    # 1. Load Model (Untrained unless checkpoint loaded)
    print("Initializing Model...")
    model = GhostWorkerV11(dim=256, num_layers=6)
    mx.eval(model.parameters())
    
    # Check for weights
    checkpoint_path = os.path.join(os.path.dirname(__file__), "../checkpoints/ghost_v11_latest.safetensors")
    if os.path.exists(checkpoint_path):
        print(f"✅ Loading Checkpoint: {checkpoint_path}")
        model.load_weights(checkpoint_path) # Assuming load_weights exists or use mx.load
    else:
        print("⚠️  No checkpoint found! Testing with RANDOM WEIGHTS.")
    
    prompts = load_prompts()
    
    print(f"\nGeneratin stories for {len(prompts)} prompts...")
    print("-" * 60)
    
    for i, p_text in enumerate(prompts):
        # Encode
        input_ids = mx.array([[ord(c) for c in p_text]])
        
        # Generate with cache
        cache = None
        # Prefill
        logits, cache = model.generate_step(input_ids)
        
        # Generate new tokens
        generated = []
        curr_token = mx.argmax(logits[:, -1, :], axis=-1).reshape(1, 1)
        
        for _ in range(50): # Generate 50 tokens
            generated.append(curr_token.item())
            logits, cache = model.generate_step(curr_token, cache)
            curr_token = mx.argmax(logits[:, -1, :], axis=-1).reshape(1, 1)
            
        gen_text = decode(mx.array(generated))
        
        print(f"[{i+1:02d}] Prompt: {p_text}")
        print(f"     Result: {gen_text}")
        print("-" * 60)

if __name__ == "__main__":
    test_quality()
