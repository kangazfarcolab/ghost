"""
Ghost Worker v10 Ultimate - ALL Features Combined
==================================================
The ultimate Ghost model combining ALL features:

From v7:
- Conv1D Tokenizer
- Depth Predictor + Byte Importance
- Surprise Predictor (Predictive Coding)
- Per-Layer Memory Query (layers 2,4)
- Sparse Attention (layers 3,5)

From v8:
- Binary Mamba (1-bit)
- BitFFN (1-bit MLP)

From v9:
- Mixture of Depths (MoD)
- Memory Bank

Total: 9 Novel Features
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import math
import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

# Import Binary Mamba components
from ghost_model_v8.core.binary_mamba import BinaryMamba, BinaryMambaConfig, BitLinear, ste_quantize
from ghost_model_v8.core.adaptive_depth import RMSNorm

# Import MoD
from ghost_model_v9.core.mixture_of_depths import MoDRouter
from ghost_model_v9.core.memory_mamba import MemoryBank


# ============================================================================
# COMPONENTS
# ============================================================================

class BitFFN(nn.Module):
    """FFN with 1-bit weights (from v8)."""
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden = dim * mult
        self.gate_proj = BitLinear(dim, hidden)
        self.up_proj = BitLinear(dim, hidden)
        self.down_proj = BitLinear(hidden, dim)
    
    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class BitMoE(nn.Module):
    """
    Mixture of Experts with 1-bit weights (from v4).
    8 experts, top-2 routing = 4x capacity.
    """
    def __init__(self, dim, num_experts=8, top_k=2, mult=4):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.router = nn.Linear(dim, num_experts)
        
        # Experts (BitFFN)
        self.experts = [BitFFN(dim, mult) for _ in range(num_experts)]
    
    def __call__(self, x):
        B, L, D = x.shape
        
        # Router logits
        router_logits = self.router(x)  # [B, L, num_experts]
        router_probs = mx.softmax(router_logits, axis=-1)
        
        # Get top-k experts using argsort (MLX doesn't have topk with indices)
        sorted_indices = mx.argsort(router_probs, axis=-1)  # Ascending
        top_k_indices = sorted_indices[:, :, -self.top_k:]  # Take last k (highest)
        
        # Gather top-k probs
        top_k_probs = mx.take_along_axis(router_probs, top_k_indices, axis=-1)
        top_k_probs = top_k_probs / mx.sum(top_k_probs, axis=-1, keepdims=True)  # Normalize
        
        # Simplified: compute weighted sum of top-k experts
        # More efficient than computing all experts
        output = mx.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]  # [B, L]
            weight = top_k_probs[:, :, k:k+1]    # [B, L, 1]
            
            # For each sample, apply the selected expert
            # This is simplified - in practice you'd batch by expert
            for e in range(self.num_experts):
                mask = (expert_idx == e).astype(mx.float32)[:, :, None]  # [B, L, 1]
                if mx.sum(mask) > 0:
                    expert_out = self.experts[e](x)
                    output = output + expert_out * mask * weight
        
        return output


class BitSparseAttention(nn.Module):
    """Sparse attention with 1-bit weights (from v7+v8)."""
    def __init__(self, dim, stride=64):
        super().__init__()
        self.stride = stride
        self.dim = dim
        self.qkv = BitLinear(dim, dim * 3)
        self.out_proj = BitLinear(dim, dim)
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


# ============================================================================
# GHOST WORKER V10 ULTIMATE
# ============================================================================

class GhostWorkerV10(nn.Module):
    """
    Ghost v10 Ultimate - ALL Features Combined
    
    Novel Features (10 total):
    1. Conv1D Tokenizer (v7)
    2. Depth Predictor + Byte Importance (v7)
    3. Surprise Predictor (v7)
    4. Per-Layer Memory Query (v7)
    5. Sparse Attention (v7+v8)
    6. Binary Mamba (v8)
    7. BitFFN (v8)
    8. Mixture of Depths (v9)
    9. Memory Bank (v9)
    10. MoE - 8 Experts, Top-2 (v4)
    """
    
    TIER = "worker"
    VERSION = "v10"
    FEATURES = 10
    
    def __init__(self, dim=256, num_layers=6, memory_size=256):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # ===== FROM v7: Tokenizer =====
        self.byte_embed = nn.Embedding(256, dim)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim)
        
        # ===== FROM v7: Depth Predictor + Byte Importance =====
        self.depth_predictor = nn.Linear(dim, 1)
        self.byte_importance = nn.Embedding(256, 1)
        
        # ===== FROM v7: Surprise Predictor =====
        self.surprise_predictor = nn.Linear(dim, 256)
        
        # ===== FROM v9: MoD Router =====
        self.mod_router = MoDRouter(dim, num_layers, min_layers=2, capacity_factor=0.5)
        
        # ===== FROM v9: Memory Projections =====
        self.mem_query_proj = nn.Linear(dim, dim)
        self.mem_key_proj = nn.Linear(dim, dim)
        self.mem_value_proj = nn.Linear(dim, dim)
        self.mem_importance = nn.Linear(dim, 1)
        self.mem_gate = nn.Linear(dim * 2, 1)
        
        # ===== LAYERS: Binary Mamba + MoE + Sparse Attention =====
        self.layers = []
        for i in range(num_layers):
            layer = {
                'norm1': RMSNorm(dim),
                'mamba': BinaryMamba(dim, BinaryMambaConfig()),
                'norm2': RMSNorm(dim),
            }
            # FROM v4: MoE at deeper layers (3, 5) for 4x capacity
            if i in [2, 4]:
                layer['ffn'] = BitMoE(dim, num_experts=8, top_k=2)
                layer['sparse_attn'] = BitSparseAttention(dim, stride=64)
            else:
                layer['ffn'] = BitFFN(dim)
            # FROM v7: Per-Layer Memory Query at layers 2, 4
            if i in [1, 3]:
                layer['mem_query'] = nn.Linear(dim, dim)
                layer['mem_gate'] = nn.Linear(dim * 2, 1)
            self.layers.append(layer)
        
        # Output
        self.norm = RMSNorm(dim)
        self.output = BitLinear(dim, 256)
    
    # Memory stored outside module tree
    _memory_banks = {}
    _stats_store = {}
    
    @property
    def memory(self):
        if id(self) not in GhostWorkerV10._memory_banks:
            GhostWorkerV10._memory_banks[id(self)] = MemoryBank(self.dim, 256)
        return GhostWorkerV10._memory_banks[id(self)]
    
    @property
    def _stats(self):
        if id(self) not in GhostWorkerV10._stats_store:
            GhostWorkerV10._stats_store[id(self)] = {
                'depth_skipped': 0, 'surprise_skipped': 0, 'mod_skipped': 0, 'total': 0
            }
        return GhostWorkerV10._stats_store[id(self)]
    
    def _query_memory_at_layer(self, h, layer):
        """Per-layer memory query (from v7)."""
        if self.memory.size == 0:
            return mx.zeros_like(h)
        
        B, L, D = h.shape
        query = mx.mean(h, axis=1)  # [B, D]
        query = layer['mem_query'](query)
        
        results = []
        for b in range(B):
            mem_out = self.memory.retrieve(query[b], top_k=4)
            results.append(mem_out)
        
        mem_output = mx.stack(results, axis=0)[:, None, :]
        mem_broadcast = mx.broadcast_to(mem_output, h.shape)
        
        gate_input = mx.concatenate([h, mem_broadcast], axis=-1)
        gate = mx.sigmoid(layer['mem_gate'](gate_input))
        return gate * mem_broadcast
    
    def _store_memory(self, h):
        """Store important tokens to memory (from v9)."""
        B, L, D = h.shape
        importance = mx.sigmoid(self.mem_importance(h)).squeeze(-1)
        for b in range(B):
            for t in range(L):
                imp = float(importance[b, t].item())
                if imp > 0.7:
                    key = self.mem_key_proj(h[b, t])
                    val = self.mem_value_proj(h[b, t])
                    self.memory.store(key, val, imp)
    
    def _retrieve_memory(self, h):
        """Global memory retrieval (from v9)."""
        B, L, D = h.shape
        if self.memory.size == 0:
            return mx.zeros_like(h)
        
        results = []
        for b in range(B):
            batch_res = []
            for t in range(L):
                q = self.mem_query_proj(h[b, t])
                batch_res.append(self.memory.retrieve(q, top_k=4))
            results.append(mx.stack(batch_res, axis=0))
        return mx.stack(results, axis=0)
    
    def __call__(self, x, use_memory=True, use_routing=True):
        """
        Forward pass with ALL features.
        
        Args:
            x: [B, L] input bytes
            use_memory: Enable memory (disable for training grad)
            use_routing: Enable depth/surprise/MoD routing
        """
        B, L = x.shape
        
        # ===== Conv1D Tokenizer (v7) =====
        h = self.byte_embed(x)
        h = nn.silu(self.conv1d(h)[:, :L, :])
        
        # ===== Depth Predictor + Byte Importance (v7) =====
        if use_routing:
            depth_score = self.depth_predictor(h).squeeze(-1)
            byte_score = self.byte_importance(x).squeeze(-1)
            depths = mx.sigmoid(depth_score + byte_score) * self.num_layers
        else:
            depths = mx.ones((B, L)) * self.num_layers
        
        # ===== Surprise Predictor (v7) =====
        if use_routing and L > 1:
            surprise_logits = self.surprise_predictor(h)
            surprise_probs = mx.softmax(surprise_logits, axis=-1)
            next_x = mx.concatenate([x[:, 1:], x[:, -1:]], axis=1)
            correct_prob = mx.take_along_axis(surprise_probs, next_x[:, :, None], axis=-1).squeeze(-1)
            surprise_mask = (1.0 - correct_prob > 0.5).astype(mx.float32)
        else:
            surprise_mask = mx.ones((B, L))
        
        # ===== Memory Retrieval (v9) =====
        if use_memory and self.memory.size > 0:
            mem_ctx = self._retrieve_memory(h)
            gate_in = mx.concatenate([h, mem_ctx], axis=-1)
            gate = mx.sigmoid(self.mem_gate(gate_in))
            h = h + gate * mem_ctx
        
        # ===== Process Layers =====
        for i, layer in enumerate(self.layers):
            # Depth mask (v7)
            depth_mask = mx.sigmoid((depths - i) * 5)
            
            # MoD mask (v9)
            if use_routing and i >= 2:
                mod_mask, _ = self.mod_router(h, i)
            else:
                mod_mask = mx.ones((B, L))
            
            # Combined mask
            combined_mask = (depth_mask * surprise_mask * mod_mask).reshape(B, L, 1)
            
            # Track stats
            if use_routing:
                self._stats['total'] += B * L
                self._stats['depth_skipped'] += int((1 - mx.mean(depth_mask).item()) * B * L)
                self._stats['surprise_skipped'] += int((1 - mx.mean(surprise_mask).item()) * B * L)
                self._stats['mod_skipped'] += int((1 - mx.mean(mod_mask).item()) * B * L)
            
            # Binary Mamba (v8)
            h = h + layer['mamba'](layer['norm1'](h)) * combined_mask
            
            # Per-Layer Memory (v7) at layers 2, 4
            if 'mem_query' in layer and use_memory:
                h = h + self._query_memory_at_layer(h, layer)
            
            # Sparse Attention (v7+v8) at layers 3, 5
            if 'sparse_attn' in layer:
                h = h + layer['sparse_attn'](h) * combined_mask
            
            # BitFFN (v8)
            h = h + layer['ffn'](layer['norm2'](h)) * combined_mask
        
        # Store to memory (v9)
        if use_memory:
            self._store_memory(h)
        
        return self.output(self.norm(h))
    
    def count_params(self):
        return sum(p.size for _, p in tree_flatten(self.parameters()))
    
    def estimate_memory(self):
        params = self.count_params()
        return {
            'params': params,
            'binary_mb': params * 0.2 / 1024 / 1024,
            'float32_mb': params * 4 / 1024 / 1024
        }
    
    def get_stats(self):
        total = self._stats['total']
        if total == 0:
            return {'compute_ratio': 1.0}
        return {
            'compute_ratio': 1.0 - (self._stats['mod_skipped'] / total),
            'depth_skip_ratio': self._stats['depth_skipped'] / total,
            'surprise_skip_ratio': self._stats['surprise_skipped'] / total,
            'mod_skip_ratio': self._stats['mod_skipped'] / total,
        }
    
    def clear_memory(self):
        self.memory.clear()
        for k in self._stats:
            self._stats[k] = 0


GhostWorker = GhostWorkerV10


if __name__ == "__main__":
    print("Ghost v10 Ultimate - ALL Features")
    print("=" * 50)
    
    model = GhostWorkerV10(dim=256, num_layers=6)
    mx.eval(model.parameters())
    
    mem = model.estimate_memory()
    print(f"Parameters: {mem['params']:,}")
    print(f"Binary size: {mem['binary_mb']:.2f} MB")
    print(f"Features: {model.FEATURES}")
    
    x = mx.array([[ord(c) for c in "kubectl get pods"]], dtype=mx.int32)
    out = model(x)
    print(f"\nForward: {x.shape} → {out.shape}")
    
    stats = model.get_stats()
    print(f"Compute ratio: {stats['compute_ratio']*100:.1f}%")
    
    print("\n✅ Ghost v10 Ultimate ready!")
