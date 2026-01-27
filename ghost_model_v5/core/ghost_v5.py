"""
ghost_v5.py - Ghost Model v5: The Next Evolution

New features over v4:
1. Sparse Attention (every 2 layers) - Global context
2. Per-Layer Memory Query - Each layer queries memory differently
3. Hybrid Architecture - Mamba (local) + Attention (global)

Architecture:
    Input → SST → Routing → [Mamba + Memory + SparseAttn?]×6 → Output
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import math
import os
import sys

# Add both ghost_model paths
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ghost_model_path = os.path.join(base_path, 'ghost_model')
sys.path.insert(0, ghost_model_path)
sys.path.insert(0, base_path)

# Import from ghost_model
from ghost_model import RMSNorm
from mamba_ssm import MambaSSM, MambaConfig


class SparseAttention(nn.Module):
    """
    Efficient attention that only attends to every Nth position.
    O(N * N/stride) instead of O(N²)
    """
    
    def __init__(self, dim, num_heads=4, stride=64):
        super().__init__()
        self.dim = dim
        self.stride = stride
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = RMSNorm(dim)
    
    def __call__(self, x):
        B, L, D = x.shape
        
        # Normalize input
        x_norm = self.norm(x)
        
        # All positions create queries
        Q = self.q_proj(x_norm)  # [B, L, D]
        
        # Only every stride-th position is a key/value (minimum 1)
        stride = min(self.stride, L)
        indices = list(range(0, L, stride))
        if len(indices) == 0:
            indices = [0]
        
        x_sparse = x_norm[:, indices, :]  # [B, num_sparse, D]
        
        K = self.k_proj(x_sparse)  # [B, num_sparse, D]
        V = self.v_proj(x_sparse)  # [B, num_sparse, D]
        
        # Attention: each position attends to sparse keys
        scores = mx.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(self.head_dim)
        weights = mx.softmax(scores, axis=-1)
        
        out = mx.matmul(weights, V)
        
        return self.out_proj(out)


class LayerwiseMemory:
    """
    Memory that can be queried at each layer with layer-specific projections.
    Not an nn.Module to avoid attribute issues.
    """
    
    def __init__(self, dim, num_layers):
        self.dim = dim
        self.num_layers = num_layers
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
        
        D = query_vec.shape[-1]
        scores = mx.matmul(query_vec, K.T) / (math.sqrt(D) * temperature)
        weights = mx.softmax(scores, axis=-1)
        
        return mx.matmul(weights, V)


class GhostModelV5(nn.Module):
    """
    Ghost Model v5 - Hybrid Architecture
    
    Features:
    - State-Space Tokenization (from v4)
    - Sparse Byte Routing (from v4)
    - Predictive Coding (from v4)
    - Sparse Attention (NEW - every 2 layers)
    - Per-Layer Memory (NEW - each layer queries memory)
    """
    
    def __init__(self, dim=256, num_layers=6, memory_slots=100, attention_stride=64):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.attention_stride = attention_stride
        
        # === STATE-SPACE TOKENIZER ===
        self.byte_embed = nn.Embedding(256, dim)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim)
        
        # === ROUTING ===
        self.depth_predictor = nn.Linear(dim, 1)
        self.byte_importance = nn.Embedding(256, 1)
        
        # === SURPRISE DETECTOR ===
        self.surprise_predictor = nn.Linear(dim, 256)
        
        # === MEMORY SYSTEM ===
        self.memory = LayerwiseMemory(dim, num_layers)
        
        # Per-layer memory projections
        self.mem_query_projs = [nn.Linear(dim, dim) for _ in range(num_layers)]
        self.mem_gates = [nn.Linear(dim * 2, 1) for _ in range(num_layers)]
        
        # === HYBRID LAYERS ===
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
            
            # Add sparse attention every 2 layers
            if i % 2 == 1:
                layer['sparse_attn'] = SparseAttention(dim, stride=attention_stride)
            
            self.layers.append(layer)
        
        # === OUTPUT ===
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, 256)
    
    def store_fact(self, key_bytes, value_bytes):
        """Store a fact in memory."""
        key_vec = mx.mean(self.byte_embed(mx.array(key_bytes, dtype=mx.int32)), axis=0)
        value_vec = mx.mean(self.byte_embed(mx.array(value_bytes, dtype=mx.int32)), axis=0)
        self.memory.store(key_vec, value_vec)
    
    def query_memory_at_layer(self, h, layer_idx):
        """Query memory with layer-specific projection."""
        if len(self.memory._keys) == 0:
            return mx.zeros_like(h)
        
        B, L, D = h.shape
        
        # Use mean hidden state as query
        query = mx.mean(h, axis=1)  # [B, D]
        query = self.mem_query_projs[layer_idx](query)  # [B, D]
        
        # Query memory for each batch
        results = []
        for b in range(B):
            mem_out = self.memory.query(query[b])
            results.append(mem_out if mem_out is not None else mx.zeros((D,)))
        
        mem_output = mx.stack(results, axis=0)  # [B, D]
        mem_expanded = mem_output[:, None, :]  # [B, 1, D]
        mem_broadcast = mx.broadcast_to(mem_expanded, h.shape)  # [B, L, D]
        
        # Gated fusion
        gate_input = mx.concatenate([h, mem_broadcast], axis=-1)  # [B, L, D*2]
        gate = mx.sigmoid(self.mem_gates[layer_idx](gate_input))  # [B, L, 1]
        
        return gate * mem_broadcast
    
    def __call__(self, x):
        B, L = x.shape
        
        # === STATE-SPACE TOKENIZATION ===
        h = self.byte_embed(x)
        h = nn.silu(self.conv1d(h)[:, :L, :])
        
        # === COMPUTE ROUTING ===
        depth_score = self.depth_predictor(h).squeeze(-1)
        byte_score = self.byte_importance(x).squeeze(-1)
        depths = mx.sigmoid(depth_score + byte_score) * self.num_layers
        
        # === SURPRISE DETECTION ===
        surprise_logits = self.surprise_predictor(h)
        surprise_probs = nn.softmax(surprise_logits, axis=-1)
        if L > 1:
            next_x = mx.concatenate([x[:, 1:], x[:, -1:]], axis=1)
            correct_prob = mx.take_along_axis(surprise_probs, next_x[:, :, None], axis=-1).squeeze(-1)
            surprise_mask = (1.0 - correct_prob > 0.5).astype(mx.float32)
        else:
            surprise_mask = mx.ones((B, L))
        
        # === HYBRID LAYERS ===
        for i, layer in enumerate(self.layers):
            # Compute masks
            depth_mask = mx.sigmoid((depths - i) * 5)
            combined_mask = (depth_mask * surprise_mask).reshape(B, L, 1)
            
            # 1. Mamba (local processing)
            h = h + layer['mamba'](layer['norm1'](h)) * combined_mask
            
            # 2. Memory query (per-layer)
            mem_contrib = self.query_memory_at_layer(h, i)
            h = h + mem_contrib
            
            # 3. Sparse Attention (every 2 layers)
            if 'sparse_attn' in layer:
                h = h + layer['sparse_attn'](h) * combined_mask
            
            # 4. FFN
            h = h + layer['ffn'](layer['norm2'](h)) * combined_mask
        
        # === OUTPUT ===
        h = self.norm(h)
        return self.output(h)
    
    def count_params(self):
        return sum(p.size for _, p in tree_flatten(self.parameters()))


if __name__ == "__main__":
    print("=" * 60)
    print("GHOST MODEL v5 - Hybrid Architecture")
    print("=" * 60)
    
    model = GhostModelV5(dim=256, num_layers=6)
    mx.eval(model.parameters())
    
    print(f"Parameters: {model.count_params():,}")
    print("\nFeatures:")
    print("  ✅ State-Space Tokenization")
    print("  ✅ Sparse Byte Routing")
    print("  ✅ Predictive Coding")
    print("  ✅ Per-Layer Memory Query (NEW)")
    print("  ✅ Sparse Attention (NEW)")
    
    # Test forward pass
    x = mx.array([[ord(c) for c in "Hello world"]], dtype=mx.int32)
    out = model(x)
    print(f"\nForward pass: {x.shape} → {out.shape}")
    print("✅ Model works!")
