"""
Ghost v12 - Comprehensive Benchmark
====================================
Tests: Speed, Resources, Q&A Accuracy
"""

import sys
import os
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

from ghost_model_v13.core.ghost_worker_v12 import GhostWorkerV13
from ghost_model_v13.training.dataset_qna import QnADataset


def generate_answer(model, prompt, max_tokens=30):
    """Generate answer from prompt"""
    tokens = [ord(c) for c in prompt]
    x = mx.array([tokens], dtype=mx.int32)
    
    generated = ""
    for _ in range(max_tokens):
        logits = model(x, use_memory=False, use_routing=False)
        probs = mx.softmax(logits[:, -1, :], axis=-1)
        next_tok = int(mx.argmax(probs, axis=-1).item())
        
        if next_tok == ord('\n') or next_tok == 0:
            break
            
        char = chr(next_tok) if 32 <= next_tok <= 126 else ''
        generated += char
        x = mx.concatenate([x, mx.array([[next_tok]])], axis=1)
    
    return generated.strip()


def run_benchmark():
    print("=" * 70)
    print("           GHOST v12 - COMPREHENSIVE BENCHMARK")
    print("=" * 70)
    print()
    
    # =========================================================================
    # 1. MODEL INITIALIZATION
    # =========================================================================
    print("📦 MODEL INFO")
    print("-" * 70)
    
    model = GhostWorkerV13(dim=256, num_layers=6)
    mx.eval(model.parameters())
    
    storage = model.estimate_storage()
    
    print(f"   Parameters: {model.count_params():,}")
    print(f"   Layers: {model.num_layers}")
    print(f"   Dimension: {model.dim}")
    print(f"   Features: {model.FEATURES}")
    print(f"   Storage: {storage['total_mb']:.2f} MB (compressed)")
    print()
    
    # =========================================================================
    # 2. TRAINING BENCHMARK
    # =========================================================================
    print("🏋️ TRAINING BENCHMARK")
    print("-" * 70)
    
    dataset = QnADataset(batch_size=8, seq_len=64)
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)
    
    def loss_fn(params, x, y):
        model.update(params)
        logits = model(x, use_memory=False, use_routing=False)
        return nn.losses.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction='mean')
    
    loss_and_grad = mx.value_and_grad(loss_fn)
    
    # Warmup
    for _ in range(5):
        batch = dataset.next_batch()
        loss, grads = loss_and_grad(model.trainable_parameters(), batch[:, :-1], batch[:, 1:])
        optimizer.update(model, grads)
        mx.eval(model.parameters())
    
    # Benchmark training
    train_steps = 100
    total_tokens = 0
    start_time = time.time()
    
    for step in range(train_steps):
        batch = dataset.next_batch()
        bx = batch[:, :-1]
        by = batch[:, 1:]
        loss, grads = loss_and_grad(model.trainable_parameters(), bx, by)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total_tokens += bx.shape[0] * bx.shape[1]
    
    train_time = time.time() - start_time
    train_tps = total_tokens / train_time
    
    print(f"   Steps: {train_steps}")
    print(f"   Time: {train_time:.2f}s")
    print(f"   Speed: {train_tps:,.0f} tokens/sec")
    print(f"   Final Loss: {loss.item():.4f}")
    print()
    
    # Continue training to lower loss for Q&A
    print("   [Training more for Q&A accuracy...]")
    for step in range(1900):  # Total 2000 steps
        batch = dataset.next_batch()
        bx = batch[:, :-1]
        by = batch[:, 1:]
        loss, grads = loss_and_grad(model.trainable_parameters(), bx, by)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
    
    print(f"   After 2000 steps Loss: {loss.item():.4f}")
    print()
    
    # =========================================================================
    # 3. INFERENCE BENCHMARK
    # =========================================================================
    print("⚡ INFERENCE BENCHMARK")
    print("-" * 70)
    
    test_prompt = "Q: What is 2 + 2?\nA:"
    tokens = [ord(c) for c in test_prompt]
    x = mx.array([tokens], dtype=mx.int32)
    
    # Warmup
    for _ in range(5):
        _ = model(x, use_memory=False, use_routing=False)
        mx.eval(_)
    
    # Benchmark
    inference_runs = 100
    start_time = time.time()
    
    for _ in range(inference_runs):
        logits = model(x, use_memory=False, use_routing=False)
        mx.eval(logits)
    
    inference_time = time.time() - start_time
    inference_tps = (inference_runs * len(tokens)) / inference_time
    latency_ms = (inference_time / inference_runs) * 1000
    
    print(f"   Runs: {inference_runs}")
    print(f"   Prompt Length: {len(tokens)} tokens")
    print(f"   Time: {inference_time:.2f}s")
    print(f"   Speed: {inference_tps:,.0f} tokens/sec")
    print(f"   Latency: {latency_ms:.2f} ms/inference")
    print()
    
    # =========================================================================
    # 4. Q&A ACCURACY TEST (20 Questions)
    # =========================================================================
    print("🧠 Q&A ACCURACY TEST (20 Questions)")
    print("-" * 70)
    
    test_questions = [
        ("What is 2 + 2?", "4"),
        ("What is the capital of France?", "Paris"),
        ("Is fire hot or cold?", "Hot"),
        ("What color is the sky?", "Blue"),
        ("What follows Monday?", "Tuesday"),
        ("Do cats bark?", "No"),
        ("Can birds fly?", "Yes"),
        ("Sun rises in the?", "East"),
        ("Sun sets in the?", "West"),
        ("Ice is solid or liquid?", "Solid"),
        ("How many legs does a spider have?", "8"),
        ("How many legs does a dog have?", "4"),
        ("What number comes after 5?", "6"),
        ("What number comes before 10?", "9"),
        ("Is 4 an even number?", "Yes"),
        ("Is 7 an even number?", "No"),
        ("Red and Blue make?", "Purple"),
        ("Blue and Yellow make?", "Green"),
        ("Who is the ghost?", "V12"),
        ("What language is this?", "English"),
    ]
    
    correct = 0
    results = []
    
    for question, expected in test_questions:
        prompt = f"Q: {question}\nA:"
        answer = generate_answer(model, prompt)
        is_correct = answer.strip().lower() == expected.lower()
        correct += int(is_correct)
        status = "✅" if is_correct else "❌"
        results.append((question, expected, answer, status))
    
    # Print results
    print(f"   {'Question':<40} {'Expected':<12} {'Got':<12} {'Status'}")
    print("   " + "-" * 70)
    for q, exp, got, status in results:
        q_short = q[:38] + ".." if len(q) > 40 else q
        print(f"   {q_short:<40} {exp:<12} {got:<12} {status}")
    
    print()
    print("-" * 70)
    accuracy = (correct / len(test_questions)) * 100
    print(f"   ACCURACY: {correct}/{len(test_questions)} = {accuracy:.1f}%")
    print()
    
    # =========================================================================
    # 5. SUMMARY
    # =========================================================================
    print("=" * 70)
    print("                         SUMMARY")
    print("=" * 70)
    print(f"   Model Size:       {storage['total_mb']:.2f} MB")
    print(f"   Parameters:       {model.count_params():,}")
    print(f"   Training Speed:   {train_tps:,.0f} tok/s")
    print(f"   Inference Speed:  {inference_tps:,.0f} tok/s")
    print(f"   Inference Latency: {latency_ms:.2f} ms")
    print(f"   Q&A Accuracy:     {accuracy:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
