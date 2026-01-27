"""
SwarmMomentum vs Adam Comparison - v10-swarm
=============================================
Compare two training approaches:
1. Simple Adam (baseline)
2. SwarmMomentum (parallel gradients + consensus-based LR)
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import time
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ghost_model_v10.core.ghost_worker_v10_swarm import GhostWorkerV10Swarm


# ============================================================================
# DATASET
# ============================================================================

TRAIN_QA = [
    ('list pods', 'kubectl get pods'),
    ('list namespaces', 'kubectl get ns'),
    ('describe pod', 'kubectl describe pod'),
    ('get logs', 'kubectl logs'),
    ('delete pod', 'kubectl delete pod'),
    ('list services', 'kubectl get svc'),
    ('get deployments', 'kubectl get deploy'),
    ('scale deployment', 'kubectl scale deploy'),
    ('list containers', 'docker ps'),
    ('run nginx', 'docker run nginx'),
    ('build image', 'docker build'),
    ('stop container', 'docker stop'),
    ('remove container', 'docker rm'),
    ('list images', 'docker images'),
    ('clone repo', 'git clone'),
    ('push changes', 'git push'),
    ('pull changes', 'git pull'),
    ('create branch', 'git checkout -b'),
    ('commit changes', 'git commit'),
    ('check status', 'git status'),
    ('list buckets', 'aws s3 ls'),
    ('copy to s3', 'aws s3 cp'),
    ('list ec2', 'aws ec2 describe-instances'),
    ('create bucket', 'aws s3 mb'),
    ('sync folder', 'aws s3 sync'),
]

TEST_QA = [
    ('show pods', 'kubectl get pods'),
    ('start nginx', 'docker run nginx'),
    ('download repo', 'git clone'),
    ('upload to s3', 'aws s3 cp'),
    ('list docker', 'docker ps'),
]


def encode(text, maxlen=40):
    return ([ord(c) for c in text] + [0]*maxlen)[:maxlen]


def make_batch(qa_pairs, batch_size=4):
    pairs = random.sample(qa_pairs, min(batch_size, len(qa_pairs)))
    inputs, targets = [], []
    for q, a in pairs:
        enc = encode(f"Q:{q} A:{a}")
        inputs.append(enc)
        targets.append(enc[1:] + [0])
    return mx.array(inputs, dtype=mx.int32), mx.array(targets, dtype=mx.int32)


def generate(model, q, maxlen=25):
    tokens = [ord(c) for c in f"Q:{q} A:"]
    for _ in range(maxlen):
        x = mx.array([(tokens + [0]*40)[:40]], dtype=mx.int32)
        logits = model(x, use_memory=False, use_routing=True)
        next_tok = int(mx.argmax(logits[0, min(len(tokens)-1, 39)]).item())
        if next_tok == 0 or next_tok == ord('\n'):
            break
        tokens.append(next_tok)
    full = ''.join(chr(t) if 32 <= t < 127 else '' for t in tokens)
    return full.split('A:')[1].strip() if 'A:' in full else ''


def evaluate(model, qa_pairs):
    correct = sum(1 for q, exp in qa_pairs if exp.lower() in generate(model, q).lower())
    return correct, len(qa_pairs)


# ============================================================================
# TRAINERS
# ============================================================================

def train_adam(model, epochs=200):
    """Simple Adam training (baseline)."""
    opt = optim.Adam(learning_rate=5e-3)
    start = time.time()
    
    for epoch in range(epochs):
        for q, a in TRAIN_QA:
            enc = encode(f"Q:{q} A:{a}")
            x = mx.array([enc], dtype=mx.int32)
            tgt = mx.array([enc[1:] + [0]], dtype=mx.int32)
            
            def loss_fn(m):
                logits = m(x[:, :-1], use_memory=False, use_routing=True)
                return mx.mean(nn.losses.cross_entropy(logits.reshape(-1, 256), tgt[:, :-1].reshape(-1)))
            
            loss, grads = mx.value_and_grad(loss_fn)(model)
            opt.update(model, grads)
            mx.eval(model.parameters())
    
    return time.time() - start, float(loss.item())


def train_swarm_momentum(model, epochs=200, num_parallel=4):
    """SwarmMomentum training with parallel gradients + consensus-based LR."""
    base_lr = 5e-3
    opt = optim.Adam(learning_rate=base_lr)
    start = time.time()
    
    for epoch in range(epochs):
        # Compute gradients on MULTIPLE parallel batches
        all_grads = []
        all_losses = []
        
        for _ in range(num_parallel):
            x, tgt = make_batch(TRAIN_QA, batch_size=4)
            
            def loss_fn(m):
                logits = m(x[:, :-1], use_memory=False, use_routing=True)
                return mx.mean(nn.losses.cross_entropy(logits.reshape(-1, 256), tgt[:, :-1].reshape(-1)))
            
            loss, grads = mx.value_and_grad(loss_fn)(model)
            all_grads.append(grads)
            all_losses.append(float(loss.item()))
        
        # Average gradients
        def average_grads(grads_list):
            avg = {}
            for key in grads_list[0]:
                if isinstance(grads_list[0][key], dict):
                    avg[key] = average_grads([g[key] for g in grads_list])
                elif isinstance(grads_list[0][key], list):
                    avg[key] = [
                        average_grads([g[key][i] for g in grads_list])
                        for i in range(len(grads_list[0][key]))
                    ]
                else:
                    stacked = mx.stack([g[key] for g in grads_list])
                    avg[key] = mx.mean(stacked, axis=0)
            return avg
        
        avg_grads = average_grads(all_grads)
        
        # Compute consensus (variance of losses)
        loss_var = sum((l - sum(all_losses)/len(all_losses))**2 for l in all_losses) / len(all_losses)
        consensus = 1.0 / (1.0 + loss_var)  # High agreement = high consensus
        
        # Scale gradients by consensus
        def scale_grads(grads, scale):
            scaled = {}
            for key in grads:
                if isinstance(grads[key], dict):
                    scaled[key] = scale_grads(grads[key], scale)
                elif isinstance(grads[key], list):
                    scaled[key] = [scale_grads(g, scale) for g in grads[key]]
                else:
                    scaled[key] = grads[key] * scale
            return scaled
        
        lr_mult = 0.5 + consensus  # [0.5, 1.5] range
        scaled_grads = scale_grads(avg_grads, lr_mult)
        
        # Apply
        opt.update(model, scaled_grads)
        mx.eval(model.parameters())
    
    return time.time() - start, sum(all_losses) / len(all_losses)


# ============================================================================
# MAIN COMPARISON
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SwarmMomentum vs Adam Comparison - v10-swarm")
    print("=" * 70)
    
    EPOCHS = 200
    
    # Model 1: Adam
    print("\n[1] Training with Adam...")
    model_adam = GhostWorkerV10Swarm(dim=256, num_layers=6)
    mx.eval(model_adam.parameters())
    adam_time, adam_loss = train_adam(model_adam, epochs=EPOCHS)
    adam_train, _ = evaluate(model_adam, TRAIN_QA)
    adam_gen, _ = evaluate(model_adam, TEST_QA)
    print(f"    Time: {adam_time:.1f}s, Loss: {adam_loss:.4f}")
    print(f"    Train: {adam_train}/25, Gen: {adam_gen}/5")
    
    # Model 2: SwarmMomentum
    print("\n[2] Training with SwarmMomentum...")
    model_swarm = GhostWorkerV10Swarm(dim=256, num_layers=6)
    mx.eval(model_swarm.parameters())
    swarm_time, swarm_loss = train_swarm_momentum(model_swarm, epochs=EPOCHS, num_parallel=4)
    swarm_train, _ = evaluate(model_swarm, TRAIN_QA)
    swarm_gen, _ = evaluate(model_swarm, TEST_QA)
    print(f"    Time: {swarm_time:.1f}s, Loss: {swarm_loss:.4f}")
    print(f"    Train: {swarm_train}/25, Gen: {swarm_gen}/5")
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Metric':<20} {'Adam':<15} {'SwarmMomentum':<15} {'Winner':<10}")
    print("-" * 70)
    print(f"{'Training Time':<20} {adam_time:.1f}s          {swarm_time:.1f}s          {'Adam' if adam_time < swarm_time else 'Swarm'}")
    print(f"{'Final Loss':<20} {adam_loss:.4f}        {swarm_loss:.4f}        {'Adam' if adam_loss < swarm_loss else 'Swarm'}")
    print(f"{'Train Accuracy':<20} {adam_train*4}%           {swarm_train*4}%           {'Adam' if adam_train > swarm_train else 'Swarm'}")
    print(f"{'Generalization':<20} {adam_gen*20}%           {swarm_gen*20}%           {'Adam' if adam_gen > swarm_gen else 'Swarm'}")
    print("=" * 70)
