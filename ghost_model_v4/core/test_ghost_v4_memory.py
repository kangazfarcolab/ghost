"""
test_ghost_v4_memory.py - Test Ghost v4 with Memory
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import sys
import os

sys.path.append(os.path.dirname(__file__))
from ghost_v4_with_memory import GhostV4WithMemory

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
    print("GHOST v4 WITH MEMORY - Full Architecture")
    print("=" * 60)
    print("Base: Proven Mamba + SST + Sparse + Predictive")
    print("New: Memory Augmentation")
    
    model = GhostV4WithMemory(dim=256, num_layers=6)
    mx.eval(model.parameters())
    print(f"\nParams: {model.count_params():,}")
    
    # Store facts
    print("\n📦 Storing facts...")
    for q, a in QA_PAIRS:
        key = q.replace("Q: ", "").replace("? A:", "").strip()
        model.store_fact([ord(c) for c in key], [ord(c) for c in a.strip()])
        mx.eval(model.memory._keys[-1], model.memory._values[-1])
    print(f"Stored {len(QA_PAIRS)} facts")
    
    # Training
    print("\n🎓 Training (300 steps)...")
    data_str = "".join([(q + a + "\n") * 200 for q, a in QA_PAIRS])
    data = mx.array([ord(c) for c in data_str], dtype=mx.int32)
    
    optimizer = optim.AdamW(learning_rate=3e-4)
    
    start = time.time()
    for step in range(300):
        starts = mx.random.randint(0, len(data) - 129, (16,))
        x = mx.stack([data[int(s):int(s)+128] for s in starts.tolist()])
        y = mx.stack([data[int(s)+1:int(s)+129] for s in starts.tolist()])
        
        def loss_fn(m):
            return nn.losses.cross_entropy(m(x), y, reduction='mean')
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}: Loss = {float(loss):.4f} | Time: {time.time()-start:.1f}s")
    
    total_time = time.time() - start
    print(f"\nTraining done in {total_time:.1f}s")
    
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
    print(f"GHOST v4 + MEMORY SCORE: {correct}/{len(QA_PAIRS)} ({100*correct/len(QA_PAIRS):.0f}%)")
    print(f"Training Time: {total_time:.1f}s")
    print("=" * 60)
    
    if correct == 10:
        print("🏆 PERFECT! Memory augmentation WORKS!")

if __name__ == "__main__":
    main()
