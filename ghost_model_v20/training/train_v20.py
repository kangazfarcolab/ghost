"""
Ghost v20 - Full Training Pipeline
===================================
Train the complete unified swarm:
1. Math tables (digit addition/subtraction)
2. Text workers (facts, geography, etc.)
3. Self-learning through debate
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from collections import Counter
import random
import time
import sys
import os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(base_path))

from ghost_model_v20.core.unified_swarm import (
    SharedBase, TextWorker, MathWorker, SharedBrain, 
    UnifiedSwarm, create_v20_swarm
)


# ============================================================
# TRAINING DATA
# ============================================================

# Math (addition table)
MATH_ADD = [(a, b, a + b) for a in range(10) for b in range(10)]

# Text/Facts
TEXT_PAIRS = [
    ("France?", "Paris"), ("Japan?", "Tokyo"), ("USA?", "Washington"),
    ("UK?", "London"), ("Germany?", "Berlin"), ("Italy?", "Rome"),
    ("Spain?", "Madrid"), ("China?", "Beijing"), ("Brazil?", "Brasilia"),
    ("Fire?", "Hot"), ("Ice?", "Cold"), ("Sky?", "Blue"), ("Grass?", "Green"),
    ("Sun?", "Yellow"), ("Blood?", "Red"), ("Water?", "Wet"), ("Snow?", "White"),
    ("Dogs bark?", "Yes"), ("Cats bark?", "No"), ("Birds fly?", "Yes"),
    ("1+1?", "2"), ("2*2?", "4"), ("10/2?", "5"),
]


def encode(text, max_len=16):
    tokens = [ord(c) for c in text[:max_len]]
    return tokens + [0] * (max_len - len(tokens))


def create_text_batch(pairs, batch_size=16):
    samples = random.choices(pairs, k=batch_size)
    inputs, targets = [], []
    for q, a in samples:
        text = f"{q}{a}"
        tokens = encode(text)
        inputs.append(tokens[:-1])
        targets.append(tokens[1:])
    return mx.array(inputs, dtype=mx.int32), mx.array(targets, dtype=mx.int32)


def ce_loss(logits, targets):
    logits_flat = logits.reshape(-1, 256)
    targets_flat = targets.reshape(-1)
    log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
    return -mx.mean(mx.take_along_axis(log_probs, targets_flat[:, None], axis=1))


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_math_tables(swarm, steps=3000, lr=1e-2):
    """Train the math worker's addition table"""
    print("\n📐 PHASE 1: Training Math Tables")
    print("-" * 60)
    
    math_worker = swarm.math_worker
    if math_worker is None:
        print("   No math worker found!")
        return
    
    optimizer = optim.AdamW(learning_rate=lr)
    
    for step in range(steps):
        batch = random.choices(MATH_ADD, k=32)
        a = mx.array([x[0] for x in batch], dtype=mx.int32)
        b = mx.array([x[1] for x in batch], dtype=mx.int32)
        targets = mx.array([x[2] for x in batch], dtype=mx.int32)
        
        def loss_fn(params):
            math_worker.tables.update(params)
            
            emb_a = math_worker.tables.add_embed(a)
            emb_b = math_worker.tables.add_embed(b)
            combined = mx.concatenate([emb_a, emb_b], axis=-1)
            h = nn.relu(math_worker.tables.add_fc1(combined))
            logits = math_worker.tables.add_fc2(h)
            
            log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
            return -mx.mean(mx.take_along_axis(log_probs, targets[:, None], axis=1))
        
        loss, grads = mx.value_and_grad(loss_fn)(math_worker.tables.trainable_parameters())
        optimizer.update(math_worker.tables, grads)
        mx.eval(math_worker.tables.parameters())
        
        if step % 1000 == 0 or step == steps - 1:
            print(f"   Step {step:4d} | Loss: {float(loss):.6f}")
    
    # Test
    correct = 0
    for a, b, expected in MATH_ADD:
        result = math_worker.tables.add(a, b)
        if result == expected:
            correct += 1
    
    print(f"   Math accuracy: {correct}/100 ({correct}%)")


def train_text_workers(swarm, steps=2000, lr=1e-3):
    """Train base + LoRAs on text data"""
    print("\n📚 PHASE 2: Training Text Workers")
    print("-" * 60)
    
    base_opt = optim.AdamW(learning_rate=lr)
    lora_opts = [optim.AdamW(learning_rate=lr) for _ in swarm.workers]
    
    for step in range(steps):
        x, y = create_text_batch(TEXT_PAIRS, batch_size=16)
        
        # Train base through first worker
        def base_loss_fn(params):
            swarm.base.update(params)
            logits = swarm.workers[0](x)
            return ce_loss(logits, y)
        
        loss, grads = mx.value_and_grad(base_loss_fn)(swarm.base.trainable_parameters())
        base_opt.update(swarm.base, grads)
        mx.eval(swarm.base.parameters())
        
        # Train each LoRA
        for worker, opt in zip(swarm.workers, lora_opts):
            def lora_loss_fn(params):
                worker.lora.update(params)
                logits = worker(x)
                return ce_loss(logits, y)
            
            _, grads = mx.value_and_grad(lora_loss_fn)(worker.lora.trainable_parameters())
            opt.update(worker.lora, grads)
            mx.eval(worker.lora.parameters())
        
        if step % 500 == 0 or step == steps - 1:
            print(f"   Step {step:4d} | Loss: {float(loss):.4f}")


def generate(worker, prompt: str, max_tokens: int = 10) -> str:
    tokens = [ord(c) for c in prompt]
    for _ in range(max_tokens):
        x = mx.array([tokens], dtype=mx.int32)
        logits = worker(x)
        next_token = int(mx.argmax(logits[0, -1]).item())
        if next_token == 0 or next_token == 10:
            break
        tokens.append(next_token)
    return "".join(chr(t) if 32 <= t < 127 else "" for t in tokens[len(prompt):]).strip()


def test_ensemble(swarm, pairs):
    """Test with debate"""
    correct = 0
    for q, a in pairs:
        answer, _, _ = swarm.answer(q)
        if answer.lower().startswith(a.lower()[:2]):
            correct += 1
    return correct / len(pairs)


def self_learn(swarm, questions, rounds=200):
    """Self-learning through debate"""
    print("\n🧠 PHASE 3: Self-Learning (Debate)")
    print("-" * 60)
    
    for r in range(rounds):
        q = random.choice(questions)
        answer, conf, method = swarm.answer(q)
        
        if r % 50 == 0:
            stats = swarm.brain.get_stats()
            print(f"   Round {r:3d} | Stored: {stats['stored']} | Unanimous: {stats['unanimous_rate']*100:.0f}%")


# ============================================================
# EVALUATION
# ============================================================

def run_evaluation(swarm):
    """Full evaluation suite"""
    print("\n" + "=" * 60)
    print("              EVALUATION SUITE")
    print("=" * 60)
    
    # Math evaluation
    print("\n📐 MATH (Single Digit)")
    print("-" * 40)
    
    math_tests = [
        ("3+5=", "8"), ("9+9=", "18"), ("0+0=", "0"),
        ("7+8=", "15"), ("4+6=", "10"), ("2+3=", "5"),
    ]
    
    math_correct = 0
    for q, expected in math_tests:
        answer, conf, method = swarm.answer(q)
        status = "✅" if answer == expected else "❌"
        print(f"   {q} → {answer} [{method}] {status}")
        if answer == expected:
            math_correct += 1
    
    print(f"\n   Accuracy: {math_correct}/{len(math_tests)} ({math_correct/len(math_tests)*100:.0f}%)")
    
    # Text evaluation
    print("\n📚 TEXT/FACTS")
    print("-" * 40)
    
    text_tests = [
        ("France?", "Paris"), ("Japan?", "Tokyo"),
        ("Fire?", "Hot"), ("Ice?", "Cold"),
    ]
    
    text_correct = 0
    for q, expected in text_tests:
        answer, conf, method = swarm.answer(q)
        status = "✅" if answer.lower().startswith(expected.lower()[:3]) else "❌"
        print(f"   {q} → '{answer}' [{method}] {status}")
        if answer.lower().startswith(expected.lower()[:3]):
            text_correct += 1
    
    print(f"\n   Accuracy: {text_correct}/{len(text_tests)} ({text_correct/len(text_tests)*100:.0f}%)")
    
    # Speed test
    print("\n⚡ SPEED")
    print("-" * 40)
    
    start = time.perf_counter()
    for _ in range(100):
        swarm.answer("3+5=")
    math_time = time.perf_counter() - start
    
    start = time.perf_counter()
    for _ in range(100):
        swarm.answer("France?")
    text_time = time.perf_counter() - start
    
    print(f"   Math: {100/math_time:.0f} q/s")
    print(f"   Text: {100/text_time:.0f} q/s")
    
    # Memory
    print("\n💾 MEMORY")
    print("-" * 40)
    
    stats = swarm.get_stats()
    print(f"   Base params: {stats['base_params']:,}")
    print(f"   Workers: {stats['workers']}")
    print(f"   Brain consensus: {stats['brain']['stored']}")
    
    return {
        "math_accuracy": math_correct / len(math_tests),
        "text_accuracy": text_correct / len(text_tests),
        "math_speed": 100 / math_time,
        "text_speed": 100 / text_time,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("     GHOST v20 - UNIFIED SWARM TRAINING")
    print("=" * 60)
    
    # Create swarm
    swarm = create_v20_swarm()
    
    stats = swarm.get_stats()
    print(f"\n📦 System: {stats['base_params']:,} base params, {stats['workers']} workers")
    
    # Phase 1: Math tables
    start = time.perf_counter()
    train_math_tables(swarm, steps=3000)
    
    # Phase 2: Text workers
    train_text_workers(swarm, steps=2000)
    
    # Phase 3: Self-learning
    all_questions = [q for q, _ in TEXT_PAIRS] + [f"{a}+{b}=" for a in range(10) for b in range(10)]
    self_learn(swarm, all_questions, rounds=200)
    
    train_time = time.perf_counter() - start
    print(f"\n⏱️ Total training time: {train_time:.1f}s")
    
    # Evaluation
    results = run_evaluation(swarm)
    
    # Summary
    print("\n" + "=" * 60)
    print("              FINAL SUMMARY")
    print("=" * 60)
    print(f"   Math Accuracy: {results['math_accuracy']*100:.0f}%")
    print(f"   Text Accuracy: {results['text_accuracy']*100:.0f}%")
    print(f"   Math Speed: {results['math_speed']:.0f} q/s")
    print(f"   Text Speed: {results['text_speed']:.0f} q/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
