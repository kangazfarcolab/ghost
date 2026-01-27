"""
Ghost v12 - Minimal Q&A Debug Training
======================================
No swarm, no complex dataset. Just pure debugging.
"""

import sys
import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

from ghost_model_v12.core.ghost_worker_v12 import GhostWorkerV12
from ghost_model_v12.training.dataset_qna import QnADataset


def train_qna_debug():
    print("=" * 60)
    print("Ghost v12 - Minimal Q&A Debug")
    print("=" * 60)
    
    # 1. Model
    model = GhostWorkerV12(dim=256, num_layers=6)
    mx.eval(model.parameters())
    print(f"Params: {model.count_params():,}")
    
    # 2. Dataset (small)
    dataset = QnADataset(batch_size=4, seq_len=64)
    print(f"Q&A pairs: {len(dataset.raw_pairs)}")
    
    # 3. Optimizer - AdamW with reasonable LR
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)
    
    # 4. Loss function
    def loss_fn(params, x, y):
        model.update(params)
        # Disable all extra features for debugging
        logits = model(x, use_memory=False, use_routing=False)
        return nn.losses.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction='mean')
    
    loss_and_grad = mx.value_and_grad(loss_fn)
    
    # 5. Training loop
    print("\nTraining...")
    print("-" * 60)
    
    nan_count = 0
    for step in range(2000):  # More steps to overfit
        batch = dataset.next_batch()
        bx = batch[:, :-1]
        by = batch[:, 1:]
        
        loss, grads = loss_and_grad(model.trainable_parameters(), bx, by)
        
        # Check for NaN
        loss_val = loss.item()
        if loss_val != loss_val:  # NaN check
            nan_count += 1
            print(f"Step {step:4d} | ⚠️ NaN detected! (count: {nan_count})")
            if nan_count > 10:
                print("Too many NaNs. Stopping.")
                break
            continue
        
        # Check gradient norms
        grad_norm = 0.0
        for k, v in grads.items():
            if hasattr(v, 'flatten'):
                grad_norm += float(mx.sum(v * v).item())
        grad_norm = grad_norm ** 0.5
        
        if grad_norm != grad_norm:  # NaN grad
            nan_count += 1
            print(f"Step {step:4d} | ⚠️ NaN gradient! (count: {nan_count})")
            if nan_count > 10:
                print("Too many NaNs. Stopping.")
                break
            continue
        
        # Update
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        if step % 50 == 0:
            print(f"Step {step:4d} | Loss: {loss_val:.4f} | GradNorm: {grad_norm:.2f}")
    
    print("-" * 60)
    print(f"Training complete. NaN count: {nan_count}")
    
    # 6. Test generation
    print("\n" + "=" * 60)
    print("Testing Q&A Generation")
    print("=" * 60)
    
    test_prompts = [
        "Q: What is 2 + 2?\nA:",
        "Q: What is the capital of France?\nA:",
        "Q: Is fire hot or cold?\nA:",
    ]
    
    for prompt in test_prompts:
        tokens = [ord(c) for c in prompt]
        x = mx.array([tokens], dtype=mx.int32)
        
        generated = ""
        for _ in range(30):  # Max tokens
            logits = model(x, use_memory=False, use_routing=False)
            probs = mx.softmax(logits[:, -1, :], axis=-1)
            next_tok = int(mx.argmax(probs, axis=-1).item())
            
            # Stop at newline or null
            if next_tok == ord('\n') or next_tok == 0:
                break
                
            char = chr(next_tok) if 32 <= next_tok <= 126 else ''
            generated += char
            x = mx.concatenate([x, mx.array([[next_tok]])], axis=1)
        
        print(f"Prompt: {prompt.strip()}")
        print(f"Answer: {generated}")
        print()


if __name__ == "__main__":
    train_qna_debug()
