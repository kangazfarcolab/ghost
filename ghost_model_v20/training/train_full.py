"""
Ghost v20 - Full Training with Multi-Digit Math
================================================
Complete training pipeline:
1. Train multi-digit carry table (any size numbers)
2. Train single-digit tables (fallback)
3. Train text workers (facts, geography)
4. Self-learning through debate
5. Full evaluation
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
from ghost_model_v20.core.multi_digit_math import MultiDigitMath


# ============================================================
# TRAINING DATA
# ============================================================

# Math tables
MATH_ADD = [(a, b, a + b) for a in range(10) for b in range(10)]
CARRY_DATA = [(a, b, c, (a+b+c) % 10, 1 if a+b+c >= 10 else 0) 
              for a in range(10) for b in range(10) for c in range(2)]

# Text/Facts
TEXT_PAIRS = [
    ("France?", "Paris"), ("Japan?", "Tokyo"), ("USA?", "Washington"),
    ("UK?", "London"), ("Germany?", "Berlin"), ("Italy?", "Rome"),
    ("Fire?", "Hot"), ("Ice?", "Cold"), ("Sky?", "Blue"), ("Grass?", "Green"),
    ("Dogs bark?", "Yes"), ("Cats bark?", "No"), ("Birds fly?", "Yes"),
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
# TRAINING PHASES
# ============================================================

def train_multi_digit_math(steps=3000, lr=1e-2):
    """Train multi-digit carry table"""
    print("\n📐 PHASE 1: Training Multi-Digit Carry Table")
    print("-" * 60)
    
    model = MultiDigitMath(hidden_dim=32)
    mx.eval(model.carry_table.parameters())
    
    optimizer = optim.AdamW(learning_rate=lr)
    
    for step in range(steps):
        batch = random.choices(CARRY_DATA, k=32)
        
        a = mx.array([x[0] for x in batch], dtype=mx.int32)
        b = mx.array([x[1] for x in batch], dtype=mx.int32)
        carry_in = mx.array([x[2] for x in batch], dtype=mx.int32)
        result_target = mx.array([x[3] for x in batch], dtype=mx.int32)
        carry_target = mx.array([x[4] for x in batch], dtype=mx.int32)
        
        def loss_fn(params):
            model.carry_table.update(params)
            result_logits, carry_logits = model.carry_table.forward_batch(a, b, carry_in)
            
            result_log_probs = mx.log(mx.softmax(result_logits, axis=-1) + 1e-10)
            result_loss = -mx.mean(mx.take_along_axis(result_log_probs, result_target[:, None], axis=1))
            
            carry_log_probs = mx.log(mx.softmax(carry_logits, axis=-1) + 1e-10)
            carry_loss = -mx.mean(mx.take_along_axis(carry_log_probs, carry_target[:, None], axis=1))
            
            return result_loss + carry_loss
        
        loss, grads = mx.value_and_grad(loss_fn)(model.carry_table.trainable_parameters())
        optimizer.update(model.carry_table, grads)
        mx.eval(model.carry_table.parameters())
        
        if step % 1000 == 0 or step == steps - 1:
            print(f"   Step {step:4d} | Loss: {float(loss):.6f}")
    
    # Test
    correct = 0
    for a, b, c, exp_r, exp_c in CARRY_DATA:
        result, carry = model.carry_table.add_with_carry(a, b, c)
        if result == exp_r and carry == exp_c:
            correct += 1
    
    print(f"   Carry table accuracy: {correct}/{len(CARRY_DATA)} ({correct/len(CARRY_DATA)*100:.1f}%)")
    return model


def train_single_digit_table(math_worker, steps=2000, lr=1e-2):
    """Train single digit addition for fallback"""
    print("\n📊 PHASE 2: Training Single-Digit Table")
    print("-" * 60)
    
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
        
        if step % 500 == 0 or step == steps - 1:
            print(f"   Step {step:4d} | Loss: {float(loss):.6f}")


def train_text_workers(swarm, steps=1500, lr=1e-3):
    """Train base + LoRAs on text data"""
    print("\n📚 PHASE 3: Training Text Workers")
    print("-" * 60)
    
    base_opt = optim.AdamW(learning_rate=lr)
    lora_opts = [optim.AdamW(learning_rate=lr) for _ in swarm.workers]
    
    for step in range(steps):
        x, y = create_text_batch(TEXT_PAIRS, batch_size=16)
        
        def base_loss_fn(params):
            swarm.base.update(params)
            logits = swarm.workers[0](x)
            return ce_loss(logits, y)
        
        loss, grads = mx.value_and_grad(base_loss_fn)(swarm.base.trainable_parameters())
        base_opt.update(swarm.base, grads)
        mx.eval(swarm.base.parameters())
        
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


# ============================================================
# EVALUATION
# ============================================================

def run_evaluation(swarm):
    print("\n" + "=" * 60)
    print("              EVALUATION SUITE")
    print("=" * 60)
    
    # Single-digit math
    print("\n📐 SINGLE-DIGIT MATH")
    print("-" * 40)
    
    single_tests = [
        ("3+5=", "8"), ("9+9=", "18"), ("0+0=", "0"),
        ("7+8=", "15"), ("4+6=", "10"),
    ]
    
    single_correct = 0
    for q, expected in single_tests:
        answer, conf, method = swarm.answer(q)
        status = "✅" if answer == expected else "❌"
        print(f"   {q} → {answer} [{method}] {status}")
        if answer == expected:
            single_correct += 1
    
    # Multi-digit math
    print("\n📐 MULTI-DIGIT MATH")
    print("-" * 40)
    
    multi_tests = [
        ("12+34=", "46"),
        ("99+1=", "100"),
        ("123+456=", "579"),
        ("1000+10000=", "11000"),
        ("999999+1=", "1000000"),
    ]
    
    multi_correct = 0
    for q, expected in multi_tests:
        answer, conf, method = swarm.answer(q)
        status = "✅" if answer == expected else "❌"
        print(f"   {q} → {answer} [{method}] {status}")
        if answer == expected:
            multi_correct += 1
    
    # Text
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
    
    # Speed
    print("\n⚡ SPEED")
    print("-" * 40)
    
    start = time.perf_counter()
    for _ in range(100):
        swarm.answer("3+5=")
    single_speed = 100 / (time.perf_counter() - start)
    
    start = time.perf_counter()
    for _ in range(100):
        swarm.answer("12345+67890=")
    multi_speed = 100 / (time.perf_counter() - start)
    
    print(f"   Single-digit: {single_speed:.0f} q/s")
    print(f"   Multi-digit: {multi_speed:.0f} q/s")
    
    return {
        "single_math": single_correct / len(single_tests),
        "multi_math": multi_correct / len(multi_tests),
        "text": text_correct / len(text_tests),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("     GHOST v20 - UNIFIED SWARM + MULTI-DIGIT MATH")
    print("=" * 60)
    
    # Create swarm
    swarm = create_v20_swarm()
    stats = swarm.get_stats()
    print(f"\n📦 System: {stats['base_params']:,} base params, {stats['workers']} workers")
    
    start = time.perf_counter()
    
    # Phase 1: Multi-digit carry table
    multi_digit_model = train_multi_digit_math(steps=3000)
    
    # Attach to math worker
    if swarm.math_worker:
        swarm.math_worker.multi_digit = multi_digit_model
        print("   ✅ Multi-digit math attached to swarm")
    
    # Phase 2: Single-digit table (backup)
    if swarm.math_worker:
        train_single_digit_table(swarm.math_worker, steps=2000)
    
    # Phase 3: Text workers
    train_text_workers(swarm, steps=1500)
    
    train_time = time.perf_counter() - start
    print(f"\n⏱️ Total training: {train_time:.1f}s")
    
    # Evaluation
    results = run_evaluation(swarm)
    
    # Summary
    print("\n" + "=" * 60)
    print("              FINAL SUMMARY")
    print("=" * 60)
    print(f"   Single-digit math: {results['single_math']*100:.0f}%")
    print(f"   Multi-digit math:  {results['multi_math']*100:.0f}%")
    print(f"   Text/Facts:        {results['text']*100:.0f}%")
    print(f"   Training time:     {train_time:.1f}s")
    print("=" * 60)
    
    if results['multi_math'] == 1.0:
        print("\n   🔥 BEST AT MATH! Handles any size numbers!")


if __name__ == "__main__":
    main()
