"""
ghost_v4_with_memory.py - Add Memory to Proven Ghost v4

Strategy: Use the PROVEN Ghost v3 Ultimate (100% accuracy)
and add memory augmentation ON TOP of it.

This avoids the issue of simplified models not learning.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import math
import time
import os
import sys

ghost_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ghost_model')
sys.path.insert(0, ghost_model_path)

from ghost_model import RMSNorm
from mamba_ssm import MambaSSM, MambaConfig


# Import the proven working model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ghost_model_v3', 'core'))


class SimpleFactMemory:
    """Simple fact memory that works."""
    
    def __init__(self):
        self._keys = []
        self._values = []
    
    def store(self, key_vec, value_vec):
        self._keys.append(key_vec)
        self._values.append(value_vec)
    
    def query(self, query_vec, temperature=0.5):
        """Soft attention over stored facts."""
        if len(self._keys) == 0:
            return None
        
        K = mx.stack(self._keys, axis=0)  # [N, D]
        V = mx.stack(self._values, axis=0)  # [N, D]
        
        # query: [D]
        D = query_vec.shape[-1]
        scores = mx.matmul(query_vec, K.T) / (math.sqrt(D) * temperature)  # [N]
        weights = mx.softmax(scores)  # [N]
        
        return mx.matmul(weights, V)  # [D]


class GhostV4WithMemory(nn.Module):
    """
    Full Ghost v4 with Memory Augmentation.
    
    Uses the proven architecture:
    - State-Space Tokenization
    - Sparse Mamba Layers
    - Predictive Coding
    
    PLUS: Memory augmentation via cross-attention.
    """
    
    def __init__(self, dim=256, num_layers=6):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # === PROVEN ARCHITECTURE FROM V3 ULTIMATE ===
        
        # State-Space Tokenizer
        self.byte_embed = nn.Embedding(256, dim)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim)
        
        # Depth Router for sparse processing
        self.depth_predictor = nn.Linear(dim, 1)
        self.byte_importance = nn.Embedding(256, 1)
        
        # Surprise detector
        self.surprise_predictor = nn.Linear(dim, 256)
        
        # Full Mamba layers
        self.layers = []
        for i in range(num_layers):
            self.layers.append({
                'norm1': RMSNorm(dim),
                'mamba': MambaSSM(dim, MambaConfig()),
                'norm2': RMSNorm(dim),
                'ffn': nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
            })
        
        # === NEW: MEMORY AUGMENTATION ===
        self.memory = SimpleFactMemory()
        
        # Memory projections
        self.mem_query_proj = nn.Linear(dim, dim)
        self.mem_value_proj = nn.Linear(dim, dim)
        
        # Memory gate
        self.mem_gate = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )
        
        # Output
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, 256)
    
    def store_fact(self, key_bytes, value_bytes):
        """Store a fact in memory."""
        key_vec = mx.mean(self.byte_embed(mx.array(key_bytes, dtype=mx.int32)), axis=0)
        value_vec = mx.mean(self.byte_embed(mx.array(value_bytes, dtype=mx.int32)), axis=0)
        self.memory.store(key_vec, value_vec)
    
    def __call__(self, x):
        B, L = x.shape
        
        # State-Space Tokenization
        h = self.byte_embed(x)
        h = nn.silu(self.conv1d(h)[:, :L, :])
        
        # Get routing
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
        
        # Process through Mamba layers with sparsity
        for i, layer in enumerate(self.layers):
            depth_mask = mx.sigmoid((depths - i) * 5)
            combined_mask = (depth_mask * surprise_mask).reshape(B, L, 1)
            
            h = h + layer['mamba'](layer['norm1'](h)) * combined_mask
            h = h + layer['ffn'](layer['norm2'](h)) * combined_mask
        
        # === MEMORY QUERY ===
        if len(self.memory._keys) > 0:
            # Use mean hidden state as query
            query = mx.mean(h, axis=1)  # [B, dim]
            query = self.mem_query_proj(query)  # [B, dim]
            
            # Query memory for each batch item
            mem_results = []
            for b in range(B):
                mem_out = self.memory.query(query[b])
                if mem_out is not None:
                    mem_results.append(self.mem_value_proj(mem_out))
                else:
                    mem_results.append(mx.zeros((self.dim,)))
            
            mem_output = mx.stack(mem_results, axis=0)  # [B, dim]
            
            # Gated addition to hidden states
            # Expand to all positions
            mem_expanded = mem_output[:, None, :]  # [B, 1, dim]
            gate_input = mx.concatenate([h, mx.broadcast_to(mem_expanded, h.shape)], axis=-1)
            gate = mx.sigmoid(self.mem_gate(gate_input))  # [B, L, 1]
            
            h = h + gate * mem_expanded
        
        h = self.norm(h)
        return self.output(h)
    
    def count_params(self):
        return sum(p.size for _, p in tree_flatten(self.parameters()))


if __name__ == "__main__":
    print("GhostV4WithMemory - Testing initialization...")
    model = GhostV4WithMemory(dim=256, num_layers=6)
    mx.eval(model.parameters())
    print(f"Params: {model.count_params():,}")
    
    # Quick test
    x = mx.array([[ord(c) for c in "Hello"]], dtype=mx.int32)
    out = model(x)
    print(f"Forward pass: {x.shape} -> {out.shape}")
    print("✅ Works!")
