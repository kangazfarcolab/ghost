"""
ghost_v7_expert.py - Ghost v7 Expert Tier (25M params)

The "Specialists" - Domain experts for explanations and medium tasks.
Scaled up from Worker: dim 256→512, layers 6→8.

Parameters: ~25M
Best for: Explanations, debugging, domain expertise
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


class ExpertSparseAttention(nn.Module):
    """Multi-head sparse attention for Expert tier."""
    
    def __init__(self, dim, num_heads=8, stride=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.stride = stride
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = RMSNorm(dim)
    
    def __call__(self, x):
        B, L, D = x.shape
        x_norm = self.norm(x)
        
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Sparse keys/values
        stride = min(self.stride, max(1, L))
        indices = list(range(0, L, stride))
        K_sparse = K[:, :, indices, :]
        V_sparse = V[:, :, indices, :]
        
        # Attention with scaling
        scale = math.sqrt(self.head_dim)
        scores = mx.matmul(Q, K_sparse.transpose(0, 1, 3, 2)) / scale
        weights = mx.softmax(scores, axis=-1)
        out = mx.matmul(weights, V_sparse)
        
        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.out_proj(out)


class SharedMemory:
    """Enhanced shared memory for Expert tier."""
    
    def __init__(self, dim):
        self.dim = dim
        self._keys = []
        self._values = []
        self._timestamps = []
    
    def store(self, key_vec, value_vec, timestamp=None):
        self._keys.append(key_vec)
        self._values.append(value_vec)
        self._timestamps.append(timestamp or len(self._keys))
    
    def query(self, query_vec, top_k=5):
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
        self._timestamps = []


class GhostExpert(nn.Module):
    """
    Ghost v7 Expert (25M params)
    
    Domain specialists for explanations and medium-length outputs.
    Scaled up: dim 256→512, layers 6→8, multi-head attention.
    
    Changes from Worker:
    - Larger embedding dimension (512)
    - More layers (8)
    - Multi-head attention (8 heads)
    - Denser attention stride (64)
    - More memory query layers
    """
    
    TIER = "expert"
    PARAMS = "25M"
    
    def __init__(self, dim=448, num_layers=8, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Tokenizer
        self.byte_embed = nn.Embedding(256, dim)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim)
        
        # Routing (enhanced)
        self.depth_predictor = nn.Linear(dim, 1)
        self.byte_importance = nn.Embedding(256, 1)
        self.surprise_predictor = nn.Linear(dim, 256)
        
        # Memory (enhanced)
        self.memory = SharedMemory(dim)
        self.mem_query_proj = nn.Linear(dim, dim)
        self.mem_key_proj = nn.Linear(dim, dim)  # Additional for better retrieval
        self.mem_gate = nn.Linear(dim * 2, 1)
        
        # Layers (more, with attention at more positions)
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
            # Sparse attention at layers 2, 4, 6 (more frequent)
            if i in [2, 4, 6]:
                layer['sparse_attn'] = ExpertSparseAttention(dim, num_heads=num_heads, stride=64)
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
            
            # Memory query at layers 1, 3, 5, 7 (more frequent)
            if i in [1, 3, 5, 7]:
                h = h + self.query_memory(h)
            
            if 'sparse_attn' in layer:
                h = h + layer['sparse_attn'](h) * combined_mask
            
            h = h + layer['ffn'](layer['norm2'](h)) * combined_mask
        
        return self.output(self.norm(h))
    
    def count_params(self):
        return sum(p.size for _, p in tree_flatten(self.parameters()))


if __name__ == "__main__":
    print("Ghost v7 Expert - 25M params")
    model = GhostExpert(dim=512, num_layers=8, num_heads=8)
    mx.eval(model.parameters())
    print(f"Params: {model.count_params():,}")
    
    x = mx.array([[ord(c) for c in "Hello"]], dtype=mx.int32)
    out = model(x)
    print(f"Forward: {x.shape} → {out.shape}")
    print("✅ Expert tier ready!")
