"""
ghost_v7_worker.py - Ghost v7 Worker Tier (6M params)

The "Army" - Fast, parallel workers for simple tasks.
Same as v6, optimized for command generation.

Parameters: ~6.58M
Best for: kubectl, aws cli, simple commands
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import math
import os
import sys

# Add ghost_model paths
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ghost_model_path = os.path.join(base_path, 'ghost_model')
sys.path.insert(0, ghost_model_path)
sys.path.insert(0, base_path)

from ghost_model import RMSNorm
from mamba_ssm import MambaSSM, MambaConfig


class FastSparseAttention(nn.Module):
    """Optimized sparse attention with larger stride."""
    
    def __init__(self, dim, stride=128):
        super().__init__()
        self.stride = stride
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = RMSNorm(dim)
    
    def __call__(self, x):
        B, L, D = x.shape
        x_norm = self.norm(x)
        
        qkv = self.qkv(x_norm)
        Q, K, V = mx.split(qkv, 3, axis=-1)
        
        stride = min(self.stride, max(1, L))
        indices = list(range(0, L, stride))
        K_sparse = K[:, indices, :]
        V_sparse = V[:, indices, :]
        
        scores = mx.matmul(Q, K_sparse.transpose(0, 2, 1)) / math.sqrt(D)
        weights = mx.softmax(scores, axis=-1)
        out = mx.matmul(weights, V_sparse)
        
        return self.out_proj(out)


class SharedMemory:
    """Simple shared memory."""
    
    def __init__(self, dim):
        self.dim = dim
        self._keys = []
        self._values = []
    
    def store(self, key_vec, value_vec):
        self._keys.append(key_vec)
        self._values.append(value_vec)
    
    def query(self, query_vec):
        if len(self._keys) == 0:
            return None
        K = mx.stack(self._keys, axis=0)
        V = mx.stack(self._values, axis=0)
        scores = mx.matmul(query_vec, K.T) / math.sqrt(self.dim)
        weights = mx.softmax(scores, axis=-1)
        return mx.matmul(weights, V)
    
    def clear(self):
        self._keys = []
        self._values = []


class GhostWorker(nn.Module):
    """
    Ghost v7 Worker (6M params)
    
    Fast, parallel workers for simple command generation.
    Identical architecture to v6.
    """
    
    TIER = "worker"
    PARAMS = "6M"
    
    def __init__(self, dim=256, num_layers=6):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # Tokenizer
        self.byte_embed = nn.Embedding(256, dim)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim)
        
        # Routing
        self.depth_predictor = nn.Linear(dim, 1)
        self.byte_importance = nn.Embedding(256, 1)
        self.surprise_predictor = nn.Linear(dim, 256)
        
        # Memory (shared, not per-layer)
        self.memory = SharedMemory(dim)
        self.mem_query_proj = nn.Linear(dim, dim)
        self.mem_gate = nn.Linear(dim * 2, 1)
        
        # Layers
        self.layers = []
        for i in range(num_layers):
            layer = {
                'norm1': RMSNorm(dim),
                'mamba': MambaSSM(dim, MambaConfig()),
                'norm2': RMSNorm(dim),
                'ffn': nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                ),
            }
            if i in [3, 5]:
                layer['sparse_attn'] = FastSparseAttention(dim, stride=128)
            self.layers.append(layer)
        
        # Output
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, 256)
    
    def store_fact(self, key_bytes, value_bytes):
        key_vec = mx.mean(self.byte_embed(mx.array(key_bytes, dtype=mx.int32)), axis=0)
        value_vec = mx.mean(self.byte_embed(mx.array(value_bytes, dtype=mx.int32)), axis=0)
        self.memory.store(key_vec, value_vec)
    
    def query_memory(self, h):
        if len(self.memory._keys) == 0:
            return mx.zeros_like(h)
        
        B, L, D = h.shape
        query = mx.mean(h, axis=1)
        query = self.mem_query_proj(query)
        
        results = []
        for b in range(B):
            mem_out = self.memory.query(query[b])
            results.append(mem_out if mem_out is not None else mx.zeros((D,)))
        
        mem_output = mx.stack(results, axis=0)[:, None, :]
        mem_broadcast = mx.broadcast_to(mem_output, h.shape)
        
        gate_input = mx.concatenate([h, mem_broadcast], axis=-1)
        gate = mx.sigmoid(self.mem_gate(gate_input))
        
        return gate * mem_broadcast
    
    def __call__(self, x):
        B, L = x.shape
        
        h = self.byte_embed(x)
        h = nn.silu(self.conv1d(h)[:, :L, :])
        
        depth_score = self.depth_predictor(h).squeeze(-1)
        byte_score = self.byte_importance(x).squeeze(-1)
        depths = mx.sigmoid(depth_score + byte_score) * self.num_layers
        
        surprise_logits = self.surprise_predictor(h)
        surprise_probs = nn.softmax(surprise_logits, axis=-1)
        if L > 1:
            next_x = mx.concatenate([x[:, 1:], x[:, -1:]], axis=1)
            correct_prob = mx.take_along_axis(surprise_probs, next_x[:, :, None], axis=-1).squeeze(-1)
            surprise_mask = (1.0 - correct_prob > 0.5).astype(mx.float32)
        else:
            surprise_mask = mx.ones((B, L))
        
        for i, layer in enumerate(self.layers):
            depth_mask = mx.sigmoid((depths - i) * 5)
            combined_mask = (depth_mask * surprise_mask).reshape(B, L, 1)
            
            h = h + layer['mamba'](layer['norm1'](h)) * combined_mask
            
            if i in [2, 4]:
                h = h + self.query_memory(h)
            
            if 'sparse_attn' in layer:
                h = h + layer['sparse_attn'](h) * combined_mask
            
            h = h + layer['ffn'](layer['norm2'](h)) * combined_mask
        
        return self.output(self.norm(h))
    
    def count_params(self):
        return sum(p.size for _, p in tree_flatten(self.parameters()))


if __name__ == "__main__":
    print("Ghost v7 Worker - 6M params")
    model = GhostWorker(dim=256, num_layers=6)
    mx.eval(model.parameters())
    print(f"Params: {model.count_params():,}")
    
    x = mx.array([[ord(c) for c in "Hello"]], dtype=mx.int32)
    out = model(x)
    print(f"Forward: {x.shape} → {out.shape}")
    print("✅ Worker tier ready!")
