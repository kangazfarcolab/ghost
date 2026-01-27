"""
train_fim.py - Fill-in-the-Middle Training

FIM (Fill-in-the-Middle) training:
- Randomly mask a span in the middle
- Model learns to predict the masked span given left+right context

Format:
  Original: "The capital of France is Paris"
  FIM:      "<PRE>The capital of <SUF> is Paris<MID>France"

Benefits:
- Bidirectional context understanding
- Better for code completion
- No architectural changes needed
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import random
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from ghost_v5_fast import GhostV5Fast

# Special tokens (using rare bytes)
PRE_TOKEN = 1   # SOH - Start of Heading
SUF_TOKEN = 2   # STX - Start of Text
MID_TOKEN = 3   # ETX - End of Text

# Base Q&A pairs
BASE_QA = [
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


def create_fim_sample(text, mask_ratio=0.15):
    """
    Create a FIM training sample.
    
    Returns: (fim_sequence, original_sequence)
    """
    if len(text) < 10:
        return None, None
    
    # Determine mask span
    mask_len = max(1, int(len(text) * mask_ratio))
    max_start = len(text) - mask_len - 1
    if max_start <= 1:
        return None, None
    
    start = random.randint(1, max_start)
    end = start + mask_len
    
    # Split into prefix, middle, suffix
    prefix = text[:start]
    middle = text[start:end]
    suffix = text[end:]
    
    # FIM format: <PRE>prefix<SUF>suffix<MID>middle
    fim_bytes = [PRE_TOKEN] + [ord(c) for c in prefix] + \
                [SUF_TOKEN] + [ord(c) for c in suffix] + \
                [MID_TOKEN] + [ord(c) for c in middle]
    
    return fim_bytes, text


def create_training_data(qa_pairs, fim_ratio=0.5):
    """Create mixed training data: regular + FIM."""
    data_bytes = []
    
    for q, a in qa_pairs:
        full_text = q + a + "\n"
        
        # Regular next-byte prediction
        for _ in range(50):
            data_bytes.extend([ord(c) for c in full_text])
        
        # FIM samples
        for _ in range(int(50 * fim_ratio)):
            fim, _ = create_fim_sample(full_text)
            if fim:
                data_bytes.extend(fim)
                data_bytes.append(10)  # newline
    
    return mx.array(data_bytes, dtype=mx.int32)


def train_step(model, data, optimizer, seq_len=64, batch_size=16):
    """Single training step."""
    starts = mx.random.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = mx.stack([data[int(s):int(s)+seq_len] for s in starts.tolist()])
    y = mx.stack([data[int(s)+1:int(s)+seq_len+1] for s in starts.tolist()])
    
    def loss_fn(m):
        logits = m(x)
        return nn.losses.cross_entropy(logits, y, reduction='mean')
    
    loss, grads = mx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    
    return float(loss)


def test_fim_understanding(model, qa_pairs):
    """Test if model understands FIM format."""
    correct = 0
    total = 0
    
    for q, a in qa_pairs:
        full_text = q + a
        
        # Test: given prefix and suffix, predict middle
        # We'll mask a key part of the answer
        if len(a.strip()) < 2:
            continue
        
        middle = a.strip()[:2]  # First 2 chars of answer
        prefix = q
        suffix = a.strip()[2:] if len(a.strip()) > 2 else ""
        
        # FIM prompt: <PRE>prefix<SUF>suffix<MID>
        fim_prompt = [PRE_TOKEN] + [ord(c) for c in prefix] + \
                     [SUF_TOKEN] + [ord(c) for c in suffix] + \
                     [MID_TOKEN]
        
        x = mx.array([fim_prompt], dtype=mx.int32)
        generated = []
        
        for _ in range(len(middle) + 2):
            logits = model(x)
            probs = nn.softmax(logits[0, -1, :] / 0.3)
            val = int(mx.argmax(probs).item())
            mx.eval(val)
            if val in [10, PRE_TOKEN, SUF_TOKEN, MID_TOKEN]: break
            generated.append(chr(val) if 32 <= val < 127 else '?')
            x = mx.concatenate([x, mx.array([[val]], dtype=mx.int32)], axis=1)
            mx.eval(x)
        
        result = ''.join(generated)
        if result.strip().startswith(middle.strip()):
            correct += 1
        total += 1
    
    return correct, total


def test_standard(model, qa_pairs):
    """Standard next-byte prediction test."""
    correct = 0
    for q, a in qa_pairs:
        x = mx.array([[ord(c) for c in q]], dtype=mx.int32)
        generated = []
        
        for _ in range(len(a) + 2):
            logits = model(x)
            probs = nn.softmax(logits[0, -1, :] / 0.3)
            val = int(mx.argmax(probs).item())
            mx.eval(val)
            if val == 10: break
            generated.append(chr(val) if 32 <= val < 127 else '?')
            x = mx.concatenate([x, mx.array([[val]], dtype=mx.int32)], axis=1)
            mx.eval(x)
        
        result = ''.join(generated)
        if result.strip() == a.strip():
            correct += 1
    
    return correct, len(qa_pairs)


def main():
    print("=" * 60)
    print("FILL-IN-THE-MIDDLE (FIM) Training")
    print("=" * 60)
    
    # Initialize model
    model = GhostV5Fast(dim=256, num_layers=6)
    mx.eval(model.parameters())
    print(f"Params: {model.count_params():,}")
    
    # Store facts
    print("\n📦 Storing facts...")
    for q, a in BASE_QA:
        key = q.replace("Q: ", "").replace("? A:", "").strip()
        model.store_fact([ord(c) for c in key], [ord(c) for c in a.strip()])
        mx.eval(model.memory._keys[-1], model.memory._values[-1])
    
    # Create mixed training data
    print("\n📊 Creating FIM training data...")
    data = create_training_data(BASE_QA, fim_ratio=0.3)
    print(f"Data size: {len(data)} bytes")
    
    # Train
    print("\n🎓 Training (300 steps)...")
    optimizer = optim.AdamW(learning_rate=3e-4)
    
    start = time.time()
    for step in range(300):
        loss = train_step(model, data, optimizer)
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}: Loss = {loss:.4f}")
    
    train_time = time.time() - start
    print(f"\nTraining done in {train_time:.1f}s")
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # Standard next-byte prediction
    std_correct, std_total = test_standard(model, BASE_QA)
    print(f"Standard (next-byte): {std_correct}/{std_total} ({100*std_correct/std_total:.0f}%)")
    
    # FIM understanding (harder)
    fim_correct, fim_total = test_fim_understanding(model, BASE_QA)
    print(f"FIM (fill-middle): {fim_correct}/{fim_total} ({100*fim_correct/fim_total:.0f}%)")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Standard accuracy: {100*std_correct/std_total:.0f}%")
    print(f"FIM accuracy: {100*fim_correct/fim_total:.0f}%")
    print(f"Training time: {train_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
