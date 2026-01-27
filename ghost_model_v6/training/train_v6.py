"""
train_v6.py - Training script for Ghost v6

Usage:
    source /Users/azfar.naufal/Documents/myprodjet/ex/learnable_tok/venv/bin/activate
    python train_v6.py
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from ghost_v6 import GhostV6

# Default Q&A training data
DEFAULT_QA = [
    ("Q: What is 2+2? A:", " 4"),
    ("Q: Capital of France? A:", " Paris"),
    ("Q: Color of sky? A:", " Blue"),
    ("Q: Largest planet? A:", " Jupiter"),
    ("Q: H2O is? A:", " Water"),
    ("Q: Python creator? A:", " Guido"),
    ("Q: 10*10? A:", " 100"),
    ("Q: Opposite of hot? A:", " Cold"),
    ("Q: Earth's star? A:", " Sun"),
    ("Q: Binary of 2? A:", " 10"),
]


def create_training_data(qa_pairs, repeat=200):
    """Convert Q&A pairs to training bytes."""
    data_str = "".join([(q + a + "\n") * repeat for q, a in qa_pairs])
    return mx.array([ord(c) for c in data_str], dtype=mx.int32)


def train(model, data, steps=300, batch_size=16, seq_len=64, lr=3e-4,
          checkpoint_dir=None, checkpoint_every=100):
    """Main training loop with optional checkpointing."""
    
    optimizer = optim.AdamW(learning_rate=lr)
    
    start = time.time()
    for step in range(steps):
        # Sample batch
        starts = mx.random.randint(0, len(data) - seq_len - 1, (batch_size,))
        x = mx.stack([data[int(s):int(s)+seq_len] for s in starts.tolist()])
        y = mx.stack([data[int(s)+1:int(s)+seq_len+1] for s in starts.tolist()])
        
        # Forward + backward
        def loss_fn(m):
            return nn.losses.cross_entropy(m(x), y, reduction='mean')
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        # Logging
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}: Loss = {float(loss):.4f} | Time: {time.time()-start:.1f}s")
        
        # Checkpoint
        if checkpoint_dir and (step + 1) % checkpoint_every == 0:
            save_checkpoint(model, checkpoint_dir, step + 1)
    
    return model


def save_checkpoint(model, checkpoint_dir, step):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_{step}.npz")
    
    params = dict(model.parameters())
    flat_params = {}
    for k, v in params.items():
        flat_params[k] = np.array(v)
    
    np.savez(path, **flat_params)
    print(f"  💾 Saved checkpoint: {path}")


def evaluate(model, qa_pairs):
    """Evaluate on Q&A pairs."""
    correct = 0
    for q, a in qa_pairs:
        x = mx.array([[ord(c) for c in q]], dtype=mx.int32)
        generated = []
        
        for _ in range(len(a) + 3):
            logits = model(x)
            probs = nn.softmax(logits[0, -1, :] / 0.3)
            val = int(mx.argmax(probs).item())
            mx.eval(val)
            if val == 10: break
            generated.append(chr(val) if 32 <= val < 127 else '?')
            x = mx.concatenate([x, mx.array([[val]], dtype=mx.int32)], axis=1)
            mx.eval(x)
        
        if ''.join(generated).strip() == a.strip():
            correct += 1
    
    return correct, len(qa_pairs)


def main():
    print("=" * 60)
    print("GHOST v6 - Training Script")
    print("=" * 60)
    
    # Initialize
    model = GhostV6(dim=256, num_layers=6)
    mx.eval(model.parameters())
    print(f"Params: {model.count_params():,}")
    
    # Store facts
    print("\n📦 Storing facts...")
    for q, a in DEFAULT_QA:
        key = q.replace("Q: ", "").replace("? A:", "").strip()
        model.store_fact([ord(c) for c in key], [ord(c) for c in a.strip()])
        mx.eval(model.memory._keys[-1], model.memory._values[-1])
    
    # Create training data
    print("\n📊 Creating training data...")
    data = create_training_data(DEFAULT_QA)
    print(f"Data size: {len(data)} bytes")
    
    # Train
    print("\n🎓 Training...")
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    model = train(model, data, steps=300, checkpoint_dir=checkpoint_dir)
    
    # Evaluate
    print("\n📈 Evaluating...")
    correct, total = evaluate(model, DEFAULT_QA)
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
