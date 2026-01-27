
import sys
import os
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Add paths
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

from ghost_model_v13.core.ghost_worker_v12 import GhostWorkerV13
from ghost_model_v13.training.dataset_qna import QnADataset
from ghost_model_v13.training.muon import Muon

class Config:
    dim = 256
    num_layers = 6
    batch_size = 16
    seq_len = 64
    steps = 500
    lr = 0.02 # Muon likes high LR (0.02 - 0.05)
    
def train_awakening():
    print("🧠 The Awakener: Intelligence Fix Protocol")
    print("==========================================")
    
    cfg = Config()
    
    # 1. Model
    model = GhostWorkerV13(dim=cfg.dim, num_layers=cfg.num_layers)
    mx.eval(model.parameters())
    print(f"👻 Ghost v12 Initialized ({model.count_params():,} params)")
    
    # 2. Data
    dataset = QnADataset(batch_size=cfg.batch_size, seq_len=cfg.seq_len)
    
    # 3. Optimizer (The Spark)
    # Using Muon for weights (2D), SGD for biases (1D) handled internally by our Muon class
    optimizer = Muon(learning_rate=cfg.lr, momentum=0.95)
    
    # Loss Function
    def loss_fn(model, x, y):
        logits = model(x, use_memory=False)
        # Cross Entropy
        loss = nn.losses.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction='mean')
        return loss

    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    
    # Training Loop
    print("\n🚀 Starting Muon Training...")
    start_time = time.time()
    
    for i in range(cfg.steps):
        # Get batch
        batch = dataset.next_batch()
        bx = batch[:, :-1]
        by = batch[:, 1:]
        
        loss, grads = loss_and_grad_fn(model, bx, by)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        if i % 10 == 0:
            print(f"Step {i:3d} | Loss: {loss.item():.4f}")
            
    print("✅ Training Complete!")
    
    # 4. Verification (The Test)
    print("\n🧪 Verification Test:")
    test_q = "Q: What is 2 + 2?\nA:"
    tokens = [ord(c) for c in test_q]
    x = mx.array([tokens], dtype=mx.int32)
    
    print(f"Prompt: {test_q}", end=" ")
    
    for _ in range(10):
        logits = model(x, use_memory=False)
        tok = mx.argmax(logits[:, -1, :], axis=-1).item()
        x = mx.concatenate([x, mx.array([[tok]])], axis=1)
        print(chr(tok), end="", flush=True)
        
    print("\n")
    
    # Save if successful
    if loss.item() < 0.5:
        print("💾 Saving Smart Weights...")
        os.makedirs("ghost_model_v13/checkpoints", exist_ok=True)
        model.save_weights("ghost_model_v13/checkpoints/ghost_v12_awakened.safetensors")

if __name__ == "__main__":
    train_awakening()
