"""
Ghost v7 vs v10 Ultimate Benchmark
==================================
- 25 Q&A training pairs
- Speed comparison
- Generalization test (5 unseen questions)
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ghost_model_v7.core.ghost_worker import GhostWorker as GhostV7
from ghost_model_v10.core.ghost_worker_v10 import GhostWorkerV10 as GhostV10


# ============================================================================
# 25 TRAINING Q&A PAIRS
# ============================================================================

TRAIN_QA = [
    # Kubernetes (8)
    ("list pods", "kubectl get pods"),
    ("list namespaces", "kubectl get ns"),
    ("describe pod", "kubectl describe pod"),
    ("get logs", "kubectl logs"),
    ("delete pod", "kubectl delete pod"),
    ("list services", "kubectl get svc"),
    ("get deployments", "kubectl get deploy"),
    ("scale deployment", "kubectl scale deploy"),
    # Docker (6)
    ("list containers", "docker ps"),
    ("run nginx", "docker run nginx"),
    ("build image", "docker build"),
    ("stop container", "docker stop"),
    ("remove container", "docker rm"),
    ("list images", "docker images"),
    # Git (6)
    ("clone repo", "git clone"),
    ("push changes", "git push"),
    ("pull changes", "git pull"),
    ("create branch", "git checkout -b"),
    ("commit changes", "git commit"),
    ("check status", "git status"),
    # AWS (5)
    ("list buckets", "aws s3 ls"),
    ("copy to s3", "aws s3 cp"),
    ("list ec2", "aws ec2 describe-instances"),
    ("create bucket", "aws s3 mb"),
    ("sync folder", "aws s3 sync"),
]

# 5 UNSEEN QUESTIONS (for generalization)
TEST_QA = [
    ("show pods", "kubectl get pods"),  # Variation of "list pods"
    ("start nginx", "docker run nginx"),  # Variation of "run nginx"
    ("download repo", "git clone"),  # Variation of "clone repo"
    ("upload to s3", "aws s3 cp"),  # Variation of "copy to s3"
    ("list docker", "docker ps"),  # Variation of "list containers"
]


def encode(text, maxlen=40):
    b = [ord(c) for c in text]
    return (b + [0]*maxlen)[:maxlen]


def generate(model, question, maxlen=25, use_mem=False, use_route=True):
    prompt = f"Q:{question} A:"
    tokens = [ord(c) for c in prompt]
    
    for _ in range(maxlen):
        padded = (tokens + [0]*40)[:40]
        x = mx.array([padded], dtype=mx.int32)
        
        # v7 vs v10 call signature
        if hasattr(model, 'FEATURES'):  # v10
            logits = model(x, use_memory=use_mem, use_routing=use_route)
        else:  # v7
            logits = model(x)
        
        pos = min(len(tokens)-1, logits.shape[1]-1)
        next_tok = int(mx.argmax(logits[0, pos]).item())
        
        if next_tok == 0 or next_tok == ord('\n'):
            break
        tokens.append(next_tok)
    
    full = ''.join(chr(t) if 32 <= t < 127 else '' for t in tokens)
    return full.split('A:')[1].strip() if 'A:' in full else full


def train_model(model, qa_pairs, epochs=200, lr=5e-3, name="Model"):
    opt = optim.Adam(learning_rate=lr)
    print(f"\nTraining {name} ({epochs} epochs on {len(qa_pairs)} pairs)...")
    
    start = time.time()
    
    for epoch in range(epochs):
        total_loss = 0
        for q, a in qa_pairs:
            full = f"Q:{q} A:{a}"
            enc = encode(full)
            x = mx.array([enc], dtype=mx.int32)
            tgt = mx.array([enc[1:] + [0]], dtype=mx.int32)
            
            def loss_fn(m):
                if hasattr(m, 'FEATURES'):
                    logits = m(x[:, :-1], use_memory=False, use_routing=True)
                else:
                    logits = m(x[:, :-1])
                return mx.mean(nn.losses.cross_entropy(logits.reshape(-1, 256), tgt[:, :-1].reshape(-1)))
            
            loss, grads = mx.value_and_grad(loss_fn)(model)
            opt.update(model, grads)
            mx.eval(model.parameters())
            total_loss += float(loss.item())
        
        if (epoch+1) % 50 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/len(qa_pairs):.4f}")
    
    elapsed = time.time() - start
    return elapsed, total_loss/len(qa_pairs)


def evaluate(model, qa_pairs, name="Model"):
    correct = 0
    results = []
    
    for q, expected in qa_pairs:
        answer = generate(model, q)
        is_correct = expected.lower() in answer.lower()
        correct += int(is_correct)
        results.append((q, expected, answer, is_correct))
    
    return correct, len(qa_pairs), results


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Ghost v7 vs v10 Ultimate Benchmark")
    print("=" * 70)
    
    # Create models
    v7 = GhostV7(dim=256, num_layers=6)
    v10 = GhostV10(dim=256, num_layers=6)
    mx.eval(v7.parameters())
    mx.eval(v10.parameters())
    
    v7_params = sum(p.size for _, p in tree_flatten(v7.parameters()))
    v10_params = v10.count_params()
    
    print(f"\nv7:  {v7_params:,} params")
    print(f"v10: {v10_params:,} params ({v10.FEATURES} features)")
    
    # Train both
    v7_time, v7_loss = train_model(v7, TRAIN_QA, epochs=200, name="Ghost v7")
    v10_time, v10_loss = train_model(v10, TRAIN_QA, epochs=200, name="Ghost v10")
    
    # Evaluate on training set
    print("\n" + "=" * 70)
    print("Training Set Evaluation (25 questions)")
    print("=" * 70)
    
    v7_train_correct, v7_train_total, v7_train_res = evaluate(v7, TRAIN_QA, "v7")
    v10_train_correct, v10_train_total, v10_train_res = evaluate(v10, TRAIN_QA, "v10")
    
    print(f"v7:  {v7_train_correct}/{v7_train_total} ({v7_train_correct*100//v7_train_total}%)")
    print(f"v10: {v10_train_correct}/{v10_train_total} ({v10_train_correct*100//v10_train_total}%)")
    
    # Generalization test
    print("\n" + "=" * 70)
    print("Generalization Test (5 UNSEEN questions)")
    print("=" * 70)
    
    v7_gen_correct, v7_gen_total, v7_gen_res = evaluate(v7, TEST_QA, "v7")
    v10_gen_correct, v10_gen_total, v10_gen_res = evaluate(v10, TEST_QA, "v10")
    
    for q, expected, answer, is_correct in v10_gen_res:
        status = "✅" if is_correct else "❌"
        print(f"{status} Q: {q}")
        print(f"   Expected: {expected}")
        print(f"   v10 Got:  {answer}")
        print()
    
    print(f"v7 Generalization:  {v7_gen_correct}/{v7_gen_total} ({v7_gen_correct*100//v7_gen_total}%)")
    print(f"v10 Generalization: {v10_gen_correct}/{v10_gen_total} ({v10_gen_correct*100//v10_gen_total}%)")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25} {'v7':<15} {'v10':<15} {'Winner':<10}")
    print("-" * 70)
    print(f"{'Parameters':<25} {v7_params:,}  {v10_params:,}  {'v7' if v7_params < v10_params else 'v10'}")
    print(f"{'Training Time':<25} {v7_time:.1f}s       {v10_time:.1f}s       {'v7' if v7_time < v10_time else 'v10'}")
    print(f"{'Final Loss':<25} {v7_loss:.4f}      {v10_loss:.4f}      {'v7' if v7_loss < v10_loss else 'v10'}")
    print(f"{'Train Accuracy':<25} {v7_train_correct*100//v7_train_total}%          {v10_train_correct*100//v10_train_total}%          {'v7' if v7_train_correct > v10_train_correct else 'v10'}")
    print(f"{'Generalization':<25} {v7_gen_correct*100//v7_gen_total}%          {v10_gen_correct*100//v10_gen_total}%          {'v7' if v7_gen_correct > v10_gen_correct else 'v10'}")
    print("=" * 70)
