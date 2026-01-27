"""
self_distillation.py - Generate Synthetic Training Data

Self-Distillation Loop:
1. Train model on base Q&A (10 pairs)
2. Generate variations of questions
3. Filter by model confidence
4. Add to training set
5. Repeat

Goal: Expand 10 Q&A → 100+ for better generalization.
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


class SelfDistillation:
    """Generate synthetic training data from model predictions."""
    
    def __init__(self, base_qa):
        self.base_qa = base_qa
        self.synthetic_qa = []
    
    def generate_question_variations(self, question, answer):
        """Generate variations of a question."""
        variations = []
        
        # Original
        variations.append((question, answer))
        
        # Case variations
        variations.append((question.lower(), answer))
        variations.append((question.upper(), answer))
        
        # Word order variations
        if "What is" in question:
            alt = question.replace("What is", "Tell me")
            variations.append((alt, answer))
        
        if "Capital of" in question:
            alt = question.replace("Capital of", "Main city of")
            variations.append((alt, answer))
        
        # Add "please"
        alt = question.replace("Q:", "Q: Please tell me,")
        variations.append((alt, answer))
        
        # Short form
        q_short = question.replace("Q: ", "").replace("? A:", "=")
        variations.append((q_short, answer))
        
        return variations
    
    def generate_math_variations(self, question, answer, num=5):
        """Generate math variations if it's a math question."""
        variations = []
        
        # Detect math patterns
        if "+" in question:
            for _ in range(num):
                a, b = random.randint(1, 20), random.randint(1, 20)
                q = f"Q: What is {a}+{b}? A:"
                ans = f" {a+b}"
                variations.append((q, ans))
        
        if "*" in question or "×" in question:
            for _ in range(num):
                a, b = random.randint(2, 12), random.randint(2, 12)
                q = f"Q: What is {a}*{b}? A:"
                ans = f" {a*b}"
                variations.append((q, ans))
        
        return variations
    
    def expand_dataset(self, num_per_question=5):
        """Expand base Q&A with variations."""
        expanded = []
        
        for q, a in self.base_qa:
            # Add original
            expanded.append((q, a))
            
            # Add text variations
            text_vars = self.generate_question_variations(q, a)
            expanded.extend(text_vars[:num_per_question])
            
            # Add math variations if applicable
            math_vars = self.generate_math_variations(q, a, num_per_question)
            expanded.extend(math_vars)
        
        self.synthetic_qa = expanded
        return expanded
    
    def filter_by_confidence(self, model, threshold=0.7):
        """Keep only Q&A pairs where model predicts correctly with confidence."""
        filtered = []
        
        for q, a in self.synthetic_qa:
            pred, confidence = self.generate_with_confidence(model, q, len(a) + 2)
            if pred.strip() == a.strip() and confidence > threshold:
                filtered.append((q, a))
        
        return filtered
    
    def generate_with_confidence(self, model, prompt, max_tokens):
        """Generate and return confidence score."""
        x = mx.array([[ord(c) for c in prompt]], dtype=mx.int32)
        generated = []
        total_confidence = 0
        
        for i in range(max_tokens):
            logits = model(x)
            probs = nn.softmax(logits[0, -1, :] / 0.3)
            val = int(mx.argmax(probs).item())
            conf = float(probs[val].item())
            mx.eval(val)
            
            if val == 10: break
            
            generated.append(chr(val) if 32 <= val < 127 else '?')
            total_confidence += conf
            x = mx.concatenate([x, mx.array([[val]], dtype=mx.int32)], axis=1)
            mx.eval(x)
        
        avg_confidence = total_confidence / max(1, len(generated))
        return ''.join(generated), avg_confidence


def train_model(model, qa_pairs, steps=200):
    """Train model on Q&A pairs."""
    data_str = "".join([(q + a + "\n") * 100 for q, a in qa_pairs])
    data = mx.array([ord(c) for c in data_str], dtype=mx.int32)
    
    optimizer = optim.AdamW(learning_rate=3e-4)
    
    for step in range(steps):
        starts = mx.random.randint(0, len(data) - 65, (16,))
        x = mx.stack([data[int(s):int(s)+64] for s in starts.tolist()])
        y = mx.stack([data[int(s)+1:int(s)+65] for s in starts.tolist()])
        
        def loss_fn(m):
            return nn.losses.cross_entropy(m(x), y, reduction='mean')
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}: Loss = {float(loss):.4f}")
    
    return model


def evaluate(model, qa_pairs):
    """Evaluate model on Q&A pairs."""
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
    print("SELF-DISTILLATION - Expand 10 Q&A → 100+")
    print("=" * 60)
    
    # Initialize distillation
    distiller = SelfDistillation(BASE_QA)
    
    # Expand dataset
    print("\n📊 Expanding dataset...")
    expanded = distiller.expand_dataset(num_per_question=5)
    print(f"Original: {len(BASE_QA)} pairs")
    print(f"Expanded: {len(expanded)} pairs")
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = GhostV5Fast(dim=256, num_layers=6)
    mx.eval(model.parameters())
    print(f"Params: {model.count_params():,}")
    
    # Store facts
    print("\n📦 Storing facts...")
    for q, a in BASE_QA:
        key = q.replace("Q: ", "").replace("? A:", "").strip()
        model.store_fact([ord(c) for c in key], [ord(c) for c in a.strip()])
        mx.eval(model.memory._keys[-1], model.memory._values[-1])
    
    # === ROUND 1: Train on base ===
    print("\n" + "=" * 60)
    print("ROUND 1: Train on base 10 Q&A")
    print("=" * 60)
    start = time.time()
    model = train_model(model, BASE_QA, steps=150)
    
    correct, total = evaluate(model, BASE_QA)
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")
    
    # === ROUND 2: Train on expanded ===
    print("\n" + "=" * 60)
    print(f"ROUND 2: Train on expanded ({len(expanded)} Q&A)")
    print("=" * 60)
    model = train_model(model, expanded, steps=150)
    
    correct, total = evaluate(model, BASE_QA)
    print(f"\nBase accuracy: {correct}/{total} ({100*correct/total:.0f}%)")
    
    # Test on NEW variations not in training
    print("\n" + "=" * 60)
    print("TEST: Novel variations (generalization)")
    print("=" * 60)
    
    novel_qa = [
        ("Q: What is 7+8? A:", " 15"),
        ("Q: What is 6*7? A:", " 42"),
        ("Q: CAPITAL OF FRANCE? A:", " Paris"),
        ("Q: color of sky? A:", " Blue"),
        ("What is 2+2=", " 4"),
    ]
    
    correct, total = evaluate(model, novel_qa)
    print(f"Generalization: {correct}/{total} ({100*correct/total:.0f}%)")
    
    total_time = time.time() - start
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original data: 10 Q&A")
    print(f"Expanded data: {len(expanded)} Q&A")
    print(f"Training time: {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
