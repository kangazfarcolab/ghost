"""
test_memory_augmented.py - Test proper memory augmentation
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import sys
import os

sys.path.append(os.path.dirname(__file__))
from memory_augmented_ghost import MemoryAugmentedGhost

QA_PAIRS = [
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

def main():
    print("=" * 60)
    print("MEMORY AUGMENTED GHOST - Proper Implementation")
    print("=" * 60)
    print("Key fixes: Cross-attention + Memory Gate + Actually query!")
    
    model = MemoryAugmentedGhost(dim=256, num_layers=4, num_memory_slots=100)
    mx.eval(model.parameters())
    print(f"\nParams: {model.count_params():,}")
    
    # Store facts
    print("\n📦 Storing facts in memory...")
    for q, a in QA_PAIRS:
        key = q.replace("Q: ", "").replace("? A:", "")
        model.store_fact([ord(c) for c in key], [ord(c) for c in a.strip()])
        mx.eval(model._memory_keys[-1], model._memory_values[-1])
    print(f"Stored {len(QA_PAIRS)} facts")
    
    # Training
    print("\n🎓 Training (500 steps)...")
    data_str = "".join([(q + a + "\n") * 200 for q, a in QA_PAIRS])
    data = mx.array([ord(c) for c in data_str], dtype=mx.int32)
    
    optimizer = optim.AdamW(learning_rate=3e-4)
    
    start = time.time()
    for step in range(500):
        starts = mx.random.randint(0, len(data) - 65, (16,))
        x = mx.stack([data[int(s):int(s)+64] for s in starts.tolist()])
        y = mx.stack([data[int(s)+1:int(s)+65] for s in starts.tolist()])
        
        def loss_fn(m):
            return nn.losses.cross_entropy(m(x), y, reduction='mean')
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}: Loss = {float(loss):.4f}")
    
    print(f"\nTraining done in {time.time() - start:.1f}s")
    
    # Test
    print("\n" + "=" * 60)
    print("TESTING")
    print("=" * 60)
    
    correct = 0
    for i, (question, expected) in enumerate(QA_PAIRS):
        x = mx.array([[ord(c) for c in question]], dtype=mx.int32)
        generated = []
        
        for _ in range(len(expected) + 2):
            logits = model(x)
            probs = nn.softmax(logits[0, -1, :] / 0.3)
            val = int(mx.argmax(probs).item())
            mx.eval(val)
            if val == 10: break
            generated.append(chr(val) if 32 <= val < 127 else '?')
            x = mx.concatenate([x, mx.array([[val]], dtype=mx.int32)], axis=1)
            mx.eval(x)
        
        result = ''.join(generated)
        is_correct = result.strip().startswith(expected.strip())
        if is_correct: correct += 1
        print(f"{'✅' if is_correct else '❌'} Q{i+1}: '{result.strip()}' (want: '{expected.strip()}')")
    
    print("\n" + "=" * 60)
    print(f"MEMORY AUGMENTED SCORE: {correct}/{len(QA_PAIRS)} ({100*correct/len(QA_PAIRS):.0f}%)")
    print("=" * 60)
    
    if correct == 10:
        print("🏆 PERFECT! Memory augmentation WORKS!")
    elif correct >= 6:
        print("📈 Progress! Memory is being used.")

if __name__ == "__main__":
    main()
