
"""
Ghost v11 - Optimized Swarm Trainer (Vectorized)
================================================
Uses mx.vmap to run multiple "workers" in parallel on the GPU.
This eliminates Python loop overhead while preserving Sway Consensus logic.

Consensus = Agreement between gradients on different data batches.
"""

import sys
import os
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten, tree_unflatten

# Try to import datasets
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("WARNING: 'datasets' library not found. Using synthetic data.")

# ============================================================================
# CONFIG
# ============================================================================

class Config:
    dim = 256
    num_layers = 6
    memory_size = 512
    
    # Swarm Config
    num_workers = 4      # Number of parallel batches to compare
    batch_size_per_worker = 8 # Total effective batch = 4 * 8 = 32
    seq_len = 256
    
    steps = 1000
    lr = 3e-3 # Boosted SGD Speed (10x)
    weight_decay = 0.0 # Disable WD to prevent NaN updates
    
    
    
    # Data
    dataset_name = "roneneldan/TinyStories"
    use_synthetic = False # Set to True for speed benchmark only
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "../checkpoints")

# Add paths
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

from ghost_model_v11.core.ghost_worker_v11 import GhostWorkerV11

# ============================================================================
# DATA LOADER
# ============================================================================

class TinyStoriesLoader:
    def __init__(self, batch_size, seq_len, use_synthetic=False):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.use_synthetic = use_synthetic
        self.dataset = None
        self.iterator = None
        
        # 0. Try Parquet File (OpenWebText/Custom)
        parquet_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/openwebtext.parquet")
        
        if os.path.exists(parquet_path):
             print(f"📂 Loading Local Parquet: {parquet_path}...")
             self.dataset = load_dataset("parquet", data_files={'train': parquet_path}, split="train", streaming=True)
             self.iterator = iter(self.dataset)
        
        # 1. Try Offline Path
        elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/tinystories")):
             data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/tinystories")
             print(f"📂 Loading Offline Data from {data_path}...")
             from datasets import load_from_disk
             self.dataset = load_from_disk(data_path)["train"]
             self.iterator = iter(self.dataset)
        elif not use_synthetic and HAS_DATASETS:
            print(f"Loading {Config.dataset_name} (Streaming)...")
            try:
                self.dataset = load_dataset(Config.dataset_name, split="train", streaming=True)
                self.iterator = iter(self.dataset)
            except Exception as e:
                print(f"Failed to load dataset: {e}")
                self.use_synthetic = True
        else:
            self.use_synthetic = True
            
        if self.use_synthetic:
            print("⚠️ Using SYTHETIC data generator")
        
    def next_batch(self):
        """Get next batch of (x, y) pairs with shape [B, L+1]"""
        if self.use_synthetic:
            # Random bytes
            data = mx.random.randint(0, 256, (self.batch_size, self.seq_len + 1), dtype=mx.int32)
            return data
            
        text_batch = []
        try:
            for _ in range(self.batch_size):
                item = next(self.iterator)
                text = item['text']
                # Truncate/Pad logic
                bytes_list = [ord(c) for c in text[:self.seq_len]]
                if len(bytes_list) < self.seq_len:
                    bytes_list += [0] * (self.seq_len - len(bytes_list))
                text_batch.append(bytes_list)
        except StopIteration:
            self.iterator = iter(self.dataset)
            return self.next_batch()
            
        # Create targets by shifting (simple next token prediction)
        # We need seq_len + 1 to get input and target properly
        # For now, we pad with 0 to get +1 length or just re-fetch
        # Simplification: just duplicate last token or pad 0
        
        # Proper way: fetch seq_len + 1 chars
        # But for valid bytes we might just take seq_len and shift
        # Input: 0..N-1, Target: 1..N
        
        # Re-tokenizing for length + 1
        data_list = []
        for t in text_batch:
            # We already padded to seq_len. 
            # Let's pad one more 0 for target shift
            data_list.append(t + [0])
            
        return mx.array(data_list, dtype=mx.int32)

# ============================================================================
# TRAINER
# ============================================================================

class SwarmTrainerOptimized:
    def __init__(self, cfg):
        self.cfg = cfg
        
        print(f"Initializing Ghost v11 (Vectorized)...")
        self.model = GhostWorkerV11(
            dim=cfg.dim, 
            num_layers=cfg.num_layers,
            memory_size=cfg.memory_size
        )
        mx.eval(self.model.parameters())
        print(f"Params: {self.model.count_params():,}")
        
        # Define a simple learning rate schedule (constant for now)
        self.lr_schedule = optim.linear_schedule(cfg.lr, cfg.lr, cfg.steps)
        # AdamW failed even with high epsilon. SGD is the only stable choice.
        self.optimizer = optim.SGD(learning_rate=self.lr_schedule, momentum=0.9)
        
        # Total batch size = workers * per_worker
        total_batch = cfg.num_workers * cfg.batch_size_per_worker
        self.loader = TinyStoriesLoader(total_batch, cfg.seq_len, use_synthetic=cfg.use_synthetic)
        self.step_num = 0

    def loss_fn(self, params, x, y):
        """
        Pure function for vmap.
        Args:
            params: Model parameters
            x: Input [B, L]
            y: Target [B, L]
        """
        self.model.update(params)
        # Memory frozen (use_memory=False) to ensure pure function
        logits = self.model(x, use_memory=False, use_routing=True)
        
        # Cross Entropy
        # Check logits range
        logits_max = mx.max(mx.abs(logits))
        
        loss = nn.losses.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction='mean')
        return loss, logits_max

    def step(self, batch_x, batch_y):
        """
        Optimized Step with Vectorized Loss and Consensus
        """
        # 0. Integrity Check (Pre-Compute) - DISABLED for Production
        # for path, p in tree_flatten(self.model.parameters()):
        #      if mx.any(mx.isnan(p)).item() or mx.any(mx.isinf(p)).item():
        #          print(f"💀 FATAL: Parameter Corruption BEFORE step {self.step_num} in {path}")
        #          sys.exit(1)

        # 1. Compute Gradients (Vectorized over workers)
        # vmap over axis 0 of input tensors (workers), broadcast params (None)
        # Returns: losses [Num_Workers], grads [Num_Workers, ...]
        loss_and_grad_fn = mx.value_and_grad(self.loss_fn)
        vmap_fn = mx.vmap(loss_and_grad_fn, in_axes=(None, 0, 0))
        
        (losses, logits_maxes), grads = vmap_fn(self.model.trainable_parameters(), batch_x, batch_y)
        
        # 2. Compute Consensus (GPU accelerated)
        flat_grads_list = tree_flatten(grads) # list of (path, tensor)
        
        # Collect all parameter gradients into one giant matrix [Num_Workers, Total_Params]
        param_vecs = []
        for _, g in flat_grads_list:
            # g: [Num_Workers, ...]
            g_flat = g.reshape(self.cfg.num_workers, -1)
            param_vecs.append(g_flat)
        
        # [Num_Workers, Total_Params]
        all_grads_matrix = mx.concatenate(param_vecs, axis=1)
        
        # Normalize rows
        norms = mx.linalg.norm(all_grads_matrix, axis=1, keepdims=True)
        normalized_grads = all_grads_matrix / (norms + 1e-8)
        
        # Compute pairwise cosine similarity matrix: G @ G.T
        # [Num_Workers, Num_Workers]
        sim_matrix = normalized_grads @ normalized_grads.T
        
        # Average off-diagonal elements
        mask = 1 - mx.eye(self.cfg.num_workers)
        avg_sim = (mx.sum(sim_matrix * mask) / (mx.sum(mask) + 1e-6)).item()
        if mx.isnan(mx.array(avg_sim)):
             avg_sim = 0.0
        
        consensus = (avg_sim + 1) / 2 # 0..1 range
        
        # 3. Aggregate Gradients
        # Simple mean across workers
        # grads is a tree. We want to reduce it over axis 0.
        avg_grads = tree_map(lambda x: mx.mean(x, axis=0), grads)
        
        # 4. Adaptive Update
        # 4. Adaptive Update
        # Feature Restored: Conservative Swarm Acceleration
        # Max Boost 1.5x (Safe) instead of 2.2x
        lr_mult = 1.0 + (consensus * 0.5) 
        
        # Warmup (Slower: 500 steps)
        warmup_steps = 500
        if self.step_num < warmup_steps:
            warmup_factor = (self.step_num + 1) / warmup_steps
            lr_mult *= warmup_factor
            
        self.optimizer.learning_rate = self.cfg.lr * lr_mult
        
        # Gradient Clipping
        # tree_flatten returns list of (key, value)
        flat_grads = tree_flatten(avg_grads) 
        # Extract just the tensors
        flattened = [g.flatten() for _, g in flat_grads]
        concat_grads = mx.concatenate(flattened)
        norm = mx.linalg.norm(concat_grads)
        
        if mx.isnan(norm):
            print(f"⚠️ NaN Gradient detected! Skipping step {self.step_num}")
            self.step_num += 1
            return mx.array(0.0), consensus, self.optimizer.learning_rate, norm, mx.array(0.0)
            
        max_norm = 0.5 # Tighter clipping
        if norm > max_norm:
            scale = max_norm / (norm + 1e-6)
            avg_grads = tree_map(lambda x: x * scale, avg_grads)
            
        self.optimizer.update(self.model, avg_grads)
        self.step_num += 1
        
        return mx.mean(losses), consensus, self.optimizer.learning_rate, norm, mx.mean(logits_maxes)

    def run(self):
        print(f"🚀 Starting Optimized Swarm Training")
        print(f"   Workers: {self.cfg.num_workers}")
        print(f"   Batch/Worker: {self.cfg.batch_size_per_worker}")
        print(f"   Dataset: {'Synthetic' if self.loader.use_synthetic else 'TinyStories'}")
        print("-" * 60)
        
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        start_time = time.time()
        
        for i in range(self.cfg.steps):
            # Get batch [Total_Batch, Seq_Len+1]
            raw_batch = self.loader.next_batch()
            
            # Split input/target
            inp = raw_batch[:, :-1]
            tgt = raw_batch[:, 1:]
            
            # Reshape for vmap: [Num_Workers, Batch_Per_Worker, Seq_Len]
            # Assumes Total_Batch is divisible by Num_Workers
            shape = (self.cfg.num_workers, self.cfg.batch_size_per_worker, self.cfg.seq_len)
            bx = inp.reshape(shape)
            by = tgt.reshape(shape)
            
            loss, cons, lr, norm, lmax = self.step(bx, by)
            mx.eval(self.model.parameters(), self.optimizer.state)
            
            if i % 10 == 0:
                elapsed = time.time() - start_time
                dt = elapsed / (i + 1)
                toks_per_sec = (self.cfg.num_workers * self.cfg.batch_size_per_worker * self.cfg.seq_len) / dt
                
                # Handle NaN for print formatting
                loss_val = loss.item()
                norm_val = norm.item() if hasattr(norm, 'item') else norm
                lmax_val = lmax.item()
                print(f"Step {i:4d} | Loss: {loss_val:.4f} | Cons: {cons:.2f} | LR: {lr:.5f} | Norm: {norm_val:.2f} | LMax: {lmax_val:.1f} | {toks_per_sec:.0f} tok/s")
                
                # Checkpoint
                if i > 0 and i % 100 == 0:
                     self.model.save_weights(os.path.join(self.cfg.checkpoint_dir, f"swarm_step_{i}.safetensors"))
                     
        print("✅ Training Complete!")
        self.model.save_weights(os.path.join(self.cfg.checkpoint_dir, "ghost_v11_perceptual_final.safetensors"))

if __name__ == "__main__":
    cfg = Config()
    trainer = SwarmTrainerOptimized(cfg)
    trainer.run()
