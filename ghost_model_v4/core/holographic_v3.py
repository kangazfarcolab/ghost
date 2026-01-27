"""
holographic_v3.py - Stronger Memory Integration

The v2 issue: Memory is added but model ignores it.
Solution: Force memory usage by making it part of the main path,
not just a residual addition.

Key changes:
1. Memory retrieval as a separate "lookup" layer
2. Concatenate memory output with hidden states
3. More training steps
"""

import mlx.core as mx
import mlx.nn as nn
import math
import os
import sys

ghost_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ghost_model')
sys.path.insert(0, ghost_model_path)

from ghost_model import RMSNorm


class SimpleMemory:
    """
    Simple key-value memory without nn.Module inheritance issues.
    """
    def __init__(self, dim):
        self.dim = dim
        self.keys = []
        self.values = []
    
    def store(self, key_vec, value_vec):
        self.keys.append(key_vec)
        self.values.append(value_vec)
    
    def retrieve(self, query, temperature=0.5):
        if len(self.keys) == 0:
            return mx.zeros((query.shape[0], query.shape[1], self.dim))
        
        # Stack memory
        K = mx.stack(self.keys, axis=0)  # [N, dim]
        V = mx.stack(self.values, axis=0)  # [N, dim]
        
        # query: [B, L, dim]
        B, L, D = query.shape
        
        # Attention: [B, L, N]
        scores = mx.matmul(query, K.T) / (math.sqrt(D) * temperature)
        weights = mx.softmax(scores, axis=-1)
        
        # Retrieve: [B, L, dim]
        return mx.matmul(weights, V)


class MemoryEnhancedModel(nn.Module):
    """
    Ghost model where memory output is CONCATENATED with hidden states,
    forcing the model to use it.
    """
    
    def __init__(self, dim=256, num_layers=4):
        super().__init__()
        self.dim = dim
        
        # Memory (not nn.Module)
        self.memory = SimpleMemory(dim)
        
        # Encoder for storing facts
        self.fact_embed = nn.Embedding(256, dim)
        
        # Main model
        self.byte_embed = nn.Embedding(256, dim)
        
        # Query projection for memory lookup
        self.query_proj = nn.Linear(dim, dim)
        
        # Fusion layer: combines hidden + memory
        self.fusion = nn.Linear(dim * 2, dim)
        
        # Layers
        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                'norm': RMSNorm(dim),
                'ffn': nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
            })
        
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, 256)
    
    def store_fact(self, key_bytes, value_bytes):
        """Store a fact."""
        key_x = mx.array(key_bytes, dtype=mx.int32)
        value_x = mx.array(value_bytes, dtype=mx.int32)
        
        key_vec = mx.mean(self.fact_embed(key_x), axis=0)
        value_vec = mx.mean(self.fact_embed(value_x), axis=0)
        
        self.memory.store(key_vec, value_vec)
    
    def __call__(self, x):
        B, L = x.shape
        
        # Embed input
        h = self.byte_embed(x)  # [B, L, dim]
        
        # Query memory
        query = self.query_proj(h)  # [B, L, dim]
        mem_out = self.memory.retrieve(query)  # [B, L, dim]
        
        # Fuse hidden and memory
        h = self.fusion(mx.concatenate([h, mem_out], axis=-1))  # [B, L, dim]
        
        # Process through layers
        for layer in self.layers:
            h = h + layer['ffn'](layer['norm'](h))
        
        h = self.norm(h)
        return self.output(h)
    
    def count_params(self):
        from mlx.utils import tree_flatten
        return sum(p.size for _, p in tree_flatten(self.parameters()))
