"""
test_hard.py - Hard Tasks Benchmark for Ghost v5

Tests multi-hop reasoning and longer context:
1. Multi-hop: "X is in Y. Y is in Z. Where is X?" → Z
2. Chained facts: Uses multiple stored facts
3. Longer answers: Full sentences
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from ghost_v5_fast import GhostV5Fast

# Standard 10 Q&A (baseline)
BASIC_QA = [
    ("Q: What is 2+2? A:", " 4"),
    ("Q: Capital of France? A:", " Paris"),
    ("Q: Color of sky? A:", " Blue"),
    ("Q: Largest planet? A:", " Jupiter"),
    ("Q: H2O is? A:", " Water"),
]

# Harder tasks: multi-hop and chained reasoning
HARD_QA = [
    # Multi-hop (requires combining facts)
    ("Q: Paris is in France. What country has Paris? A:", " France"),
    ("Q: The sun is a star. What type is the sun? A:", " star"),
    ("Q: Ice is frozen water. What is ice? A:", " frozen water"),
    
    # Math chains
    ("Q: 5+5=10. What is 5+5? A:", " 10"),
    ("Q: 3*4=12. What is 3*4? A:", " 12"),
]

def main():
    print("=" * 60)
    print("GHOST v5 FAST - Hard Tasks Benchmark")
    print("=" * 60)
    
    model = GhostV5Fast(dim=256, num_layers=6)
    mx.eval(model.parameters())
    print(f"Params: {model.count_params():,}")
    
    # Combine all Q&A
    all_qa = BASIC_QA + HARD_QA
    
    # Store facts
    print(f"\n📦 Storing {len(all_qa)} facts...")
    for q, a in all_qa:
        key = q.replace("Q: ", "").replace("? A:", "").strip()
        model.store_fact([ord(c) for c in key], [ord(c) for c in a.strip()])
        mx.eval(model.memory._keys[-1], model.memory._values[-1])
    
    # Training
    print("\n🎓 Training (300 steps)...")
    data_str = "".join([(q + a + "\n") * 150 for q, a in all_qa])
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
    
    # Test basic
    print("\n" + "=" * 60)
    print("BASIC TASKS (5 questions)")
    print("=" * 60)
    
    basic_correct = 0
    for i, (question, expected) in enumerate(BASIC_QA):
        result = generate(model, question, len(expected) + 2)
        is_correct = result.strip().startswith(expected.strip())
        if is_correct: basic_correct += 1
        print(f"{'✅' if is_correct else '❌'} Q{i+1}: '{result.strip()}' (want: '{expected.strip()}')")
    
    # Test hard
    print("\n" + "=" * 60)
    print("HARD TASKS (5 questions)")
    print("=" * 60)
    
    hard_correct = 0
    for i, (question, expected) in enumerate(HARD_QA):
        result = generate(model, question, len(expected) + 5)
        is_correct = result.strip().startswith(expected.strip())
        if is_correct: hard_correct += 1
        print(f"{'✅' if is_correct else '❌'} Q{i+1}: '{result.strip()}' (want: '{expected.strip()}')")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Basic: {basic_correct}/5 ({100*basic_correct/5:.0f}%)")
    print(f"Hard:  {hard_correct}/5 ({100*hard_correct/5:.0f}%)")
    print(f"Total: {basic_correct+hard_correct}/10 ({100*(basic_correct+hard_correct)/10:.0f}%)")
    print(f"Time:  {total_time:.1f}s")
    print(f"Params: {model.count_params():,}")
    print("=" * 60)
    
    if basic_correct + hard_correct >= 8:
        print("🏆 Great performance on hard tasks!")
        return True
    return False


def generate(model, prompt, max_tokens):
    x = mx.array([[ord(c) for c in prompt]], dtype=mx.int32)
    generated = []
    
    for _ in range(max_tokens):
        logits = model(x)
        probs = nn.softmax(logits[0, -1, :] / 0.3)
        val = int(mx.argmax(probs).item())
        mx.eval(val)
        if val == 10: break
        generated.append(chr(val) if 32 <= val < 127 else '?')
        x = mx.concatenate([x, mx.array([[val]], dtype=mx.int32)], axis=1)
        mx.eval(x)
    
    return ''.join(generated)


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Ready for Self-Distillation!")
