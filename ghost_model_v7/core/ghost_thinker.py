"""
ghost_v7_thinker.py - Ghost v7 Thinker Tier (50M params)

The "Brains" - Heavy thinkers for planning and complex reasoning.
Scaled up from Expert: dim 512→768, layers 8→12.

Parameters: ~50M
Best for: Planning, synthesis, long document generation
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


class ThinkerMultiHeadAttention(nn.Module):
    """Full multi-head attention for Thinker tier (denser, more heads)."""
    
    def __init__(self, dim, num_heads=12, stride=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.stride = stride
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = RMSNorm(dim)
        
        # Rotary position encoding for better long-range
        self.rope_scale = 1.0 / (10000 ** (mx.arange(0, self.head_dim, 2) / self.head_dim))
    
    def apply_rope(self, x, seq_len):
        """Apply rotary position embeddings."""
        positions = mx.arange(seq_len)[:, None]
        freqs = positions * self.rope_scale[None, :]
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        
        # Split x into pairs
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        
        # Apply rotation
        x_rot = mx.concatenate([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], axis=-1)
        
        return x_rot
    
    def __call__(self, x):
        B, L, D = x.shape
        x_norm = self.norm(x)
        
        Q = self.q_proj(x_norm).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = self.k_proj(x_norm).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = self.v_proj(x_norm).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Sparse keys/values (denser stride for Thinker)
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


class HierarchicalMemory:
    """Enhanced hierarchical memory for Thinker tier."""
    
    def __init__(self, dim):
        self.dim = dim
        # Short-term (recent context)
        self._short_keys = []
        self._short_values = []
        self._short_max = 100
        # Long-term (persistent facts)
        self._long_keys = []
        self._long_values = []
    
    def store_short(self, key_vec, value_vec):
        self._short_keys.append(key_vec)
        self._short_values.append(value_vec)
        # Evict oldest if full
        if len(self._short_keys) > self._short_max:
            self._short_keys.pop(0)
            self._short_values.pop(0)
    
    def store_long(self, key_vec, value_vec):
        self._long_keys.append(key_vec)
        self._long_values.append(value_vec)
    
    def query(self, query_vec, use_long=True):
        all_keys = self._short_keys + (self._long_keys if use_long else [])
        all_values = self._short_values + (self._long_values if use_long else [])
        
        if len(all_keys) == 0:
            return None
        
        K = mx.stack(all_keys, axis=0)
        V = mx.stack(all_values, axis=0)
        scores = mx.matmul(query_vec, K.T) / math.sqrt(self.dim)
        weights = mx.softmax(scores, axis=-1)
        return mx.matmul(weights, V)
    
    def clear_short(self):
        self._short_keys = []
        self._short_values = []
    
    def clear_all(self):
        self._short_keys = []
        self._short_values = []
        self._long_keys = []
        self._long_values = []


class GhostThinker(nn.Module):
    """
    Ghost v7 Thinker (50M params)
    
    Heavy thinkers for planning, synthesis, and long document generation.
    Scaled up: dim 512→768, layers 8→12, 12 attention heads.
    
    Changes from Expert:
    - Larger embedding dimension (768)
    - More layers (12)
    - More attention heads (12)
    - Denser attention stride (32)
    - Hierarchical memory (short-term + long-term)
    - Rotary position embeddings
    """
    
    TIER = "thinker"
    PARAMS = "50M"
    
    def __init__(self, dim=512, num_layers=10, num_heads=8):
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
        
        # Hierarchical Memory
        self.memory = HierarchicalMemory(dim)
        self.mem_query_proj = nn.Linear(dim, dim)
        self.mem_key_proj = nn.Linear(dim, dim)
        self.mem_value_proj = nn.Linear(dim, dim)
        self.mem_gate = nn.Linear(dim * 2, 1)
        
        # Layers (most, with frequent attention)
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
            # Attention at every other layer (more frequent for reasoning)
            if i % 2 == 0:
                layer['attn'] = ThinkerMultiHeadAttention(dim, num_heads=num_heads, stride=32)
            self.layers.append(layer)
        
        # Output
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, 256)
    
    def store_fact(self, key_bytes, value_bytes, long_term=False):
        key_vec = mx.mean(self.byte_embed(mx.array(key_bytes, dtype=mx.int32)), axis=0)
        value_vec = mx.mean(self.byte_embed(mx.array(value_bytes, dtype=mx.int32)), axis=0)
        if long_term:
            self.memory.store_long(key_vec, value_vec)
        else:
            self.memory.store_short(key_vec, value_vec)
    
    def query_memory(self, h, use_long=True):
        if len(self.memory._short_keys) == 0 and len(self.memory._long_keys) == 0:
            return mx.zeros_like(h)
        
        B, L, D = h.shape
        query = mx.mean(h, axis=1)
        query = self.mem_query_proj(query)
        
        results = []
        for b in range(B):
            mem_out = self.memory.query(query[b], use_long=use_long)
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
            
            # Memory query at layers 1, 3, 5, 7, 9, 11 (every other)
            if i % 2 == 1:
                h = h + self.query_memory(h)
            
            if 'attn' in layer:
                h = h + layer['attn'](h) * combined_mask
            
            h = h + layer['ffn'](layer['norm2'](h)) * combined_mask
        
        return self.output(self.norm(h))
    
    def count_params(self):
        return sum(p.size for _, p in tree_flatten(self.parameters()))


if __name__ == "__main__":
    print("Ghost v7 Thinker - 50M params")
    model = GhostThinker(dim=768, num_layers=12, num_heads=12)
    mx.eval(model.parameters())
    print(f"Params: {model.count_params():,}")
    
    x = mx.array([[ord(c) for c in "Hello"]], dtype=mx.int32)
    out = model(x)
    print(f"Forward: {x.shape} → {out.shape}")
    print("✅ Thinker tier ready!")
