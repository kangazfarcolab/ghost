"""
memory_augmented_ghost.py - Proper Memory Integration

Based on user's excellent analysis of the bugs:
1. Memory must be QUERIED in forward pass
2. Use cross-attention (proven to work, differentiable)
3. Memory gate learns WHEN to use memory

This is how modern RAG systems work.
"""

import mlx.core as mx
import mlx.nn as nn
import math
import os
import sys

ghost_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ghost_model')
sys.path.insert(0, ghost_model_path)

from ghost_model import RMSNorm


class MemoryAugmentedGhost(nn.Module):
    """
    Ghost model that learns WHEN and HOW to query memory.
    
    Key innovations:
    1. Memory gate: learns when to use memory vs internal knowledge
    2. Cross-attention: attends to relevant memory slots
    3. End-to-end trainable
    """
    
    def __init__(self, dim=256, num_layers=4, num_memory_slots=100):
        super().__init__()
        self.dim = dim
        self.num_memory_slots = num_memory_slots
        
        self.byte_embed = nn.Embedding(256, dim)
        
        # Main processing layers
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
        
        # Memory system - stored as lists (not nn.Module attrs)
        self._memory_keys = []
        self._memory_values = []
        
        # Memory attention (cross-attention to memory)
        self.memory_query = nn.Linear(dim, dim)
        self.memory_key_proj = nn.Linear(dim, dim)
        self.memory_value_proj = nn.Linear(dim, dim)
        
        # Memory gate: "should I use memory here?"
        self.memory_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )
        
        # Output
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, 256)
    
    def store_fact(self, question_bytes, answer_bytes):
        """Store Q->A mapping in memory"""
        # Encode question as key
        q_embed = self.byte_embed(mx.array(question_bytes, dtype=mx.int32))
        key = mx.mean(q_embed, axis=0)
        
        # Encode answer as value
        a_embed = self.byte_embed(mx.array(answer_bytes, dtype=mx.int32))
        value = mx.mean(a_embed, axis=0)
        
        # Store
        self._memory_keys.append(key)
        self._memory_values.append(value)
    
    def query_memory(self, h):
        """
        Cross-attention to memory bank.
        h: [B, L, dim]
        returns: [B, L, dim] memory contribution
        """
        B, L, D = h.shape
        
        if len(self._memory_keys) == 0:
            return mx.zeros_like(h)
        
        # Stack memory
        mem_keys = mx.stack(self._memory_keys, axis=0)  # [num_facts, dim]
        mem_values = mx.stack(self._memory_values, axis=0)  # [num_facts, dim]
        
        # Project hidden states to queries
        Q = self.memory_query(h)  # [B, L, dim]
        
        # Project memory
        K = self.memory_key_proj(mem_keys)  # [num_facts, dim]
        V = self.memory_value_proj(mem_values)  # [num_facts, dim]
        
        # Cross-attention: Q attends to K, retrieves V
        # [B, L, dim] @ [dim, num_facts] -> [B, L, num_facts]
        attn_scores = mx.matmul(Q, K.T) / math.sqrt(D)
        attn_weights = mx.softmax(attn_scores, axis=-1)  # [B, L, num_facts]
        
        # [B, L, num_facts] @ [num_facts, dim] -> [B, L, dim]
        retrieved = mx.matmul(attn_weights, V)
        
        return retrieved
    
    def __call__(self, x):
        B, L = x.shape
        h = self.byte_embed(x)
        
        # Process through layers
        for layer in self.layers:
            h = h + layer['ffn'](layer['norm'](h))
        
        # Memory gate: learn when to query (0=ignore, 1=use)
        gate = mx.sigmoid(self.memory_gate(h))  # [B, L, 1]
        
        # Query memory - THIS IS THE KEY FIX!
        memory_out = self.query_memory(h)  # [B, L, dim]
        
        # Gated combination: h + gate * memory
        h = h + gate * memory_out
        
        h = self.norm(h)
        return self.output(h)
    
    def count_params(self):
        from mlx.utils import tree_flatten
        return sum(p.size for _, p in tree_flatten(self.parameters()))
