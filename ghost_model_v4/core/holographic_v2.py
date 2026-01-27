"""
holographic_v2.py - Fixed Holographic Memory with Cross-Attention

Key Changes from v1:
1. Add cross-attention layer to query memory
2. Make retrieval differentiable via soft attention
3. Gated fusion with hidden states
4. End-to-end trainable

The magic: Model LEARNS when and how to query memory.
"""

import mlx.core as mx
import mlx.nn as nn
import math
import os
import sys

ghost_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ghost_model')
sys.path.insert(0, ghost_model_path)

from ghost_model import RMSNorm


class HolographicMemory(nn.Module):
    """
    Improved holographic memory with differentiable retrieval.
    
    Key insight: Store facts as key-value pairs in high-dimensional space,
    retrieve via learned attention over stored facts.
    """
    
    def __init__(self, dim=256, num_slots=100):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        
        # Use a list to store facts (not nn.Module attributes)
        self._stored_keys = []
        self._stored_values = []
        
        # Byte encoder for facts
        self.byte_embed = nn.Embedding(256, dim)
        self.encoder = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
    
    def encode_sequence(self, byte_seq):
        """
        Encode a byte sequence into a single vector.
        byte_seq: list of int (0-255)
        """
        x = mx.array(byte_seq, dtype=mx.int32)
        h = self.byte_embed(x)  # [L, dim]
        h = self.encoder(h)
        # Pool: mean over sequence
        return mx.mean(h, axis=0)  # [dim]
    
    def store(self, key_bytes, value_bytes):
        """
        Store a key-value fact.
        key_bytes: list of int (e.g., [ord(c) for c in "France capital"])
        value_bytes: list of int (e.g., [ord(c) for c in "Paris"])
        """
        if len(self._stored_keys) >= self.num_slots:
            print("Warning: Memory full!")
            return
        
        key_vec = self.encode_sequence(key_bytes)
        value_vec = self.encode_sequence(value_bytes)
        
        self._stored_keys.append(key_vec)
        self._stored_values.append(value_vec)
    
    def retrieve(self, query_vec, temperature=0.1):
        """
        Retrieve from memory using soft attention.
        query_vec: [B, L, dim] or [dim]
        Returns: [B, L, dim] or [dim] weighted sum of values
        """
        if len(self._stored_keys) == 0:
            # No facts stored, return zeros
            return mx.zeros_like(query_vec)
        
        # Stack stored keys and values
        stored_keys = mx.stack(self._stored_keys, axis=0)    # [N, dim]
        stored_values = mx.stack(self._stored_values, axis=0)  # [N, dim]
        
        # Handle different input shapes
        single = query_vec.ndim == 1
        if single:
            query_vec = query_vec[None, None, :]  # [1, 1, dim]
        elif query_vec.ndim == 2:
            query_vec = query_vec[None, :, :]  # [1, L, dim]
        
        B, L, D = query_vec.shape
        
        # Compute attention scores
        # query: [B, L, D], keys: [N, D]
        # scores: [B, L, N]
        scores = mx.matmul(query_vec, stored_keys.T) / math.sqrt(D)
        scores = scores / temperature  # Sharpen
        
        # Soft attention weights
        weights = mx.softmax(scores, axis=-1)  # [B, L, N]
        
        # Weighted sum of values
        # weights: [B, L, N], values: [N, D]
        retrieved = mx.matmul(weights, stored_values)  # [B, L, D]
        
        if single:
            return retrieved[0, 0, :]  # [dim]
        return retrieved


class HolographicAttention(nn.Module):
    """
    Cross-attention layer that queries holographic memory.
    
    This is the key fix: instead of just storing facts,
    the model LEARNS to query them via attention.
    """
    
    def __init__(self, dim, memory_dim=256, num_slots=100):
        super().__init__()
        self.dim = dim
        
        # Memory bank
        self.memory = HolographicMemory(dim=memory_dim, num_slots=num_slots)
        
        # Query projection: hidden states → memory queries
        self.query_proj = nn.Linear(dim, memory_dim)
        
        # Value projection: memory output → hidden dim
        self.value_proj = nn.Linear(memory_dim, dim)
        
        # Gating: learn when to use memory vs. internal knowledge
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # Layer norm for stability
        self.norm = RMSNorm(dim)
    
    def store_fact(self, key_bytes, value_bytes):
        """Store a fact in memory."""
        self.memory.store(key_bytes, value_bytes)
    
    def __call__(self, h):
        """
        Query memory and fuse with hidden states.
        h: [B, L, dim] hidden states
        Returns: [B, L, dim] enhanced hidden states
        """
        # Normalize input
        h_norm = self.norm(h)
        
        # Project to query space
        queries = self.query_proj(h_norm)  # [B, L, memory_dim]
        
        # Retrieve from memory
        retrieved = self.memory.retrieve(queries)  # [B, L, memory_dim]
        
        # Project back to hidden dim
        mem_out = self.value_proj(retrieved)  # [B, L, dim]
        
        # Gated fusion: learn when to use memory
        gate_input = mx.concatenate([h, mem_out], axis=-1)  # [B, L, dim*2]
        gate = self.gate(gate_input)  # [B, L, dim], values 0-1
        
        # Blend: gate=1 means use memory, gate=0 means ignore
        output = h + gate * mem_out
        
        return output


class GhostV4WithMemory(nn.Module):
    """
    Ghost v4 with working Holographic Memory.
    
    Architecture:
    Input → StateSpace Tokenizer → Sparse Mamba Layers → Holographic Attention → Output
    """
    
    def __init__(self, dim=256, num_layers=6, num_memory_slots=100):
        super().__init__()
        self.dim = dim
        
        # Byte embedding
        self.byte_embed = nn.Embedding(256, dim)
        
        # Simple layers (for this test, no full Mamba to speed things up)
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
        
        # NEW: Holographic Memory with cross-attention
        self.memory_attn = HolographicAttention(dim, dim, num_memory_slots)
        
        # Output
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, 256)
    
    def store_fact(self, key_bytes, value_bytes):
        """Store a fact in holographic memory."""
        self.memory_attn.store_fact(key_bytes, value_bytes)
    
    def __call__(self, x):
        B, L = x.shape
        
        h = self.byte_embed(x)
        
        for layer in self.layers:
            h = h + layer['ffn'](layer['norm'](h))
        
        # Query holographic memory
        h = self.memory_attn(h)
        
        h = self.norm(h)
        return self.output(h)
    
    def count_params(self):
        from mlx.utils import tree_flatten
        return sum(p.size for _, p in tree_flatten(self.parameters()))
