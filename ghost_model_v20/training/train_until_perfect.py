"""
Ghost v20 - Train Until Perfect (100%)
=======================================
Train all tables UNTIL they reach 100% accuracy.
Uses adaptive training with increasing steps.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import random
import time
import sys
import os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(base_path))

from ghost_model_v20.core.complete_math import (
    CompleteMathEngine, 
    create_carry_data, create_borrow_data, 
    create_mult_data, create_div_data
)


def train_until_perfect(model, table_name, data, forward_fn, max_attempts=10, base_steps=5000):
    """Train a table until 100% accuracy"""
    print(f"\n📐 Training {table_name} until 100%...")
    
    for attempt in range(max_attempts):
        steps = base_steps * (attempt + 1)  # More steps each attempt
        lr = 1e-2 / (attempt + 1)  # Lower LR each attempt
        
        opt = optim.AdamW(learning_rate=lr)
        table = getattr(model, table_name)
        
        for step in range(steps):
            batch = random.choices(data, k=64)  # Larger batch
            
            def loss_fn(params):
                table.update(params)
                return forward_fn(table, batch)
            
            loss, grads = mx.value_and_grad(loss_fn)(table.trainable_parameters())
            opt.update(table, grads)
            mx.eval(table.parameters())
            
            if step % 2000 == 0:
                print(f"   Attempt {attempt+1}, Step {step} | Loss: {float(loss):.6f}")
        
        # Test accuracy
        correct, total = test_table(model, table_name, data)
        accuracy = correct / total * 100
        print(f"   Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        
        if correct == total:
            print(f"   ✅ {table_name} reached 100%!")
            return True
    
    print(f"   ⚠️ {table_name} max attempts reached")
    return False


def test_table(model, table_name, data):
    if table_name == "carry_table":
        correct = sum(1 for a,b,c,r,co in data 
                      if model.carry_table.add_with_carry(a,b,c) == (r,co))
    elif table_name == "borrow_table":
        correct = sum(1 for a,b,bo,r,boo in data 
                      if model.borrow_table.sub_with_borrow(a,b,bo) == (r,boo))
    elif table_name == "mult_table":
        correct = sum(1 for a,b,p in data 
                      if model.mult_table.multiply(a,b) == p)
    elif table_name == "div_table":
        correct = sum(1 for d,v,q,r in data 
                      if model.div_table.divide(d,v) == (q,r))
    else:
        return 0, len(data)
    return correct, len(data)


def carry_forward(table, batch):
    a = mx.array([x[0] for x in batch], dtype=mx.int32)
    b = mx.array([x[1] for x in batch], dtype=mx.int32)
    c = mx.array([x[2] for x in batch], dtype=mx.int32)
    rt = mx.array([x[3] for x in batch], dtype=mx.int32)
    ct = mx.array([x[4] for x in batch], dtype=mx.int32)
    rl, cl = table.forward_batch(a, b, c)
    l1 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(rl, axis=-1)+1e-10), rt[:, None], axis=1))
    l2 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(cl, axis=-1)+1e-10), ct[:, None], axis=1))
    return l1 + l2


def borrow_forward(table, batch):
    a = mx.array([x[0] for x in batch], dtype=mx.int32)
    b = mx.array([x[1] for x in batch], dtype=mx.int32)
    bo = mx.array([x[2] for x in batch], dtype=mx.int32)
    rt = mx.array([x[3] for x in batch], dtype=mx.int32)
    bt = mx.array([x[4] for x in batch], dtype=mx.int32)
    rl, bl = table.forward_batch(a, b, bo)
    l1 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(rl, axis=-1)+1e-10), rt[:, None], axis=1))
    l2 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(bl, axis=-1)+1e-10), bt[:, None], axis=1))
    return l1 + l2


def mult_forward(table, batch):
    a = mx.array([x[0] for x in batch], dtype=mx.int32)
    b = mx.array([x[1] for x in batch], dtype=mx.int32)
    targets = mx.array([x[2] for x in batch], dtype=mx.int32)
    logits = table.forward_batch(a, b)
    return -mx.mean(mx.take_along_axis(mx.log(mx.softmax(logits, axis=-1)+1e-10), targets[:, None], axis=1))


def div_forward(table, batch):
    d = mx.array([x[0] for x in batch], dtype=mx.int32)
    v = mx.array([x[1] for x in batch], dtype=mx.int32)
    qt = mx.array([x[2] for x in batch], dtype=mx.int32)
    rt = mx.array([x[3] for x in batch], dtype=mx.int32)
    ql, rl = table.forward_batch(d, v)
    l1 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(ql, axis=-1)+1e-10), qt[:, None], axis=1))
    l2 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(rl, axis=-1)+1e-10), rt[:, None], axis=1))
    return l1 + l2


def main():
    print("=" * 70)
    print("     GHOST v20 - TRAINING UNTIL 100% ON ALL OPERATIONS")
    print("=" * 70)
    
    model = CompleteMathEngine(hidden_dim=48)  # Slightly larger
    mx.eval(model.parameters())
    
    print(f"\n📦 Model: {model.count_params():,} params")
    
    start = time.perf_counter()
    
    # Train each table until perfect
    carry_data = create_carry_data()
    borrow_data = create_borrow_data()
    mult_data = create_mult_data()
    div_data = create_div_data()
    
    train_until_perfect(model, "carry_table", carry_data, carry_forward, base_steps=5000)
    train_until_perfect(model, "borrow_table", borrow_data, borrow_forward, base_steps=5000)
    train_until_perfect(model, "mult_table", mult_data, mult_forward, base_steps=5000)
    train_until_perfect(model, "div_table", div_data, div_forward, base_steps=8000)
    
    train_time = time.perf_counter() - start
    
    print("\n" + "=" * 70)
    print("              FINAL TEST")
    print("=" * 70)
    
    # Test all operations
    print("\n➕ Addition:")
    add_tests = [("999", "1"), ("12345", "67890"), ("999999", "1")]
    for a, b in add_tests:
        expected = str(int(a) + int(b))
        result = model.add(a, b)
        print(f"   {a} + {b} = {result} {'✅' if result == expected else '❌'}")
    
    print("\n➖ Subtraction:")
    sub_tests = [("1000", "1"), ("80235", "67890"), ("1000000", "1")]
    for a, b in sub_tests:
        expected = str(int(a) - int(b))
        result = model.subtract(a, b)
        print(f"   {a} - {b} = {result} {'✅' if result == expected else '❌'}")
    
    print("\n✖️ Multiplication:")
    mul_tests = [("12", "12"), ("99", "99"), ("25", "4")]
    for a, b in mul_tests:
        expected = str(int(a) * int(b))
        result = model.multiply(a, b)
        print(f"   {a} × {b} = {result} {'✅' if result == expected else '❌'}")
    
    print("\n➗ Division:")
    div_tests = [("10", "3"), ("100", "7"), ("144", "12"), ("999", "9")]
    for a, b in div_tests:
        exp_q, exp_r = str(int(a) // int(b)), str(int(a) % int(b))
        q, r = model.divide(a, b)
        status = "✅" if q == exp_q and r == exp_r else "❌"
        print(f"   {a} ÷ {b} = {q} R {r} (expected: {exp_q} R {exp_r}) {status}")
    
    print("\n" + "=" * 70)
    print(f"   Total training time: {train_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
