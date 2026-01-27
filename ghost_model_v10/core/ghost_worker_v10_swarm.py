"""
Ghost Worker v10-Swarm - Optimized for Ghost Swarm Architecture
================================================================
9 Novel Features (no MoE - swarm provides external routing)

Features:
1. Conv1D Tokenizer (v7)
2. Depth Predictor + Byte Importance (v7)
3. Surprise Predictor (v7)
4. Per-Layer Memory Query (v7)
5. Sparse Attention (v7+v8)
6. Binary Mamba (v8)
7. BitFFN (v8)
8. Mixture of Depths (v9)
9. Memory Bank (v9)

Params: ~8M (vs 19M with MoE)
Designed for: Worker tier in Ghost Swarm
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import math
import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

from ghost_model_v8.core.binary_mamba import BinaryMamba, BinaryMambaConfig, BitLinear
from ghost_model_v8.core.adaptive_depth import RMSNorm
from ghost_model_v9.core.mixture_of_depths import MoDRouter
from ghost_model_v9.core.memory_mamba import MemoryBank


class BitFFN(nn.Module):
    """FFN with 1-bit weights."""
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden = dim * mult
        self.gate_proj = BitLinear(dim, hidden)
        self.up_proj = BitLinear(dim, hidden)
        self.down_proj = BitLinear(hidden, dim)
    
    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class BitSparseAttention(nn.Module):
    """Sparse attention with 1-bit weights."""
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


class GhostWorkerV10Swarm(nn.Module):
    """
    Ghost v10-Swarm Worker - 9 Features, No MoE
    
    Optimized for Ghost Swarm architecture where each worker
    specializes in one domain. MoE is provided externally by the swarm.
    """
    
    TIER = "worker"
    VERSION = "v10-swarm"
    FEATURES = 9
    
    def __init__(self, dim=256, num_layers=6, memory_size=256):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # Conv1D Tokenizer (v7)
        self.byte_embed = nn.Embedding(256, dim)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim)
        
        # Depth Predictor + Byte Importance (v7)
        self.depth_predictor = nn.Linear(dim, 1)
        self.byte_importance = nn.Embedding(256, 1)
        
        # Surprise Predictor (v7)
        self.surprise_predictor = nn.Linear(dim, 256)
        
        # MoD Router (v9)
        self.mod_router = MoDRouter(dim, num_layers, min_layers=2, capacity_factor=0.5)
        
        # Memory Projections (v9)
        self.mem_query_proj = nn.Linear(dim, dim)
        self.mem_key_proj = nn.Linear(dim, dim)
        self.mem_value_proj = nn.Linear(dim, dim)
        self.mem_importance = nn.Linear(dim, 1)
        self.mem_gate = nn.Linear(dim * 2, 1)
        
        # Layers: Binary Mamba + BitFFN + Sparse Attention (NO MoE!)
        self.layers = []
        for i in range(num_layers):
            layer = {
                'norm1': RMSNorm(dim),
                'mamba': BinaryMamba(dim, BinaryMambaConfig()),
                'norm2': RMSNorm(dim),
                'ffn': BitFFN(dim),  # Simple BitFFN, no MoE
            }
            # Sparse Attention at layers 3, 5
            if i in [2, 4]:
                layer['sparse_attn'] = BitSparseAttention(dim, stride=64)
            # Per-Layer Memory Query at layers 2, 4
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
        if id(self) not in GhostWorkerV10Swarm._memory_banks:
            GhostWorkerV10Swarm._memory_banks[id(self)] = MemoryBank(self.dim, 256)
        return GhostWorkerV10Swarm._memory_banks[id(self)]
    
    @property
    def _stats(self):
        if id(self) not in GhostWorkerV10Swarm._stats_store:
            GhostWorkerV10Swarm._stats_store[id(self)] = {
                'depth_skipped': 0, 'surprise_skipped': 0, 'mod_skipped': 0, 'total': 0
            }
        return GhostWorkerV10Swarm._stats_store[id(self)]
    
    def _query_memory_at_layer(self, h, layer):
        if self.memory.size == 0:
            return mx.zeros_like(h)
        B, L, D = h.shape
        query = mx.mean(h, axis=1)
        query = layer['mem_query'](query)
        results = [self.memory.retrieve(query[b], top_k=4) for b in range(B)]
        mem_output = mx.stack(results, axis=0)[:, None, :]
        mem_broadcast = mx.broadcast_to(mem_output, h.shape)
        gate_input = mx.concatenate([h, mem_broadcast], axis=-1)
        gate = mx.sigmoid(layer['mem_gate'](gate_input))
        return gate * mem_broadcast
    
    def _store_memory(self, h):
        B, L, D = h.shape
        importance = mx.sigmoid(self.mem_importance(h)).squeeze(-1)
        for b in range(B):
            for t in range(L):
                imp = float(importance[b, t].item())
                if imp > 0.7:
                    self.memory.store(self.mem_key_proj(h[b, t]), self.mem_value_proj(h[b, t]), imp)
    
    def _retrieve_memory(self, h):
        B, L, D = h.shape
        if self.memory.size == 0:
            return mx.zeros_like(h)
        results = []
        for b in range(B):
            batch_res = [self.memory.retrieve(self.mem_query_proj(h[b, t]), top_k=4) for t in range(L)]
            results.append(mx.stack(batch_res, axis=0))
        return mx.stack(results, axis=0)
    
    def __call__(self, x, use_memory=True, use_routing=True):
        B, L = x.shape
        
        # Conv1D Tokenizer
        h = self.byte_embed(x)
        h = nn.silu(self.conv1d(h)[:, :L, :])
        
        # Depth Predictor + Byte Importance
        if use_routing:
            depth_score = self.depth_predictor(h).squeeze(-1)
            byte_score = self.byte_importance(x).squeeze(-1)
            depths = mx.sigmoid(depth_score + byte_score) * self.num_layers
        else:
            depths = mx.ones((B, L)) * self.num_layers
        
        # Surprise Predictor
        if use_routing and L > 1:
            surprise_logits = self.surprise_predictor(h)
            surprise_probs = mx.softmax(surprise_logits, axis=-1)
            next_x = mx.concatenate([x[:, 1:], x[:, -1:]], axis=1)
            correct_prob = mx.take_along_axis(surprise_probs, next_x[:, :, None], axis=-1).squeeze(-1)
            surprise_mask = (1.0 - correct_prob > 0.5).astype(mx.float32)
        else:
            surprise_mask = mx.ones((B, L))
        
        # Memory Retrieval
        if use_memory and self.memory.size > 0:
            mem_ctx = self._retrieve_memory(h)
            gate_in = mx.concatenate([h, mem_ctx], axis=-1)
            gate = mx.sigmoid(self.mem_gate(gate_in))
            h = h + gate * mem_ctx
        
        # Process Layers
        for i, layer in enumerate(self.layers):
            depth_mask = mx.sigmoid((depths - i) * 5)
            
            if use_routing and i >= 2:
                mod_mask, _ = self.mod_router(h, i)
            else:
                mod_mask = mx.ones((B, L))
            
            combined_mask = (depth_mask * surprise_mask * mod_mask).reshape(B, L, 1)
            
            if use_routing:
                self._stats['total'] += B * L
                self._stats['depth_skipped'] += int((1 - mx.mean(depth_mask).item()) * B * L)
                self._stats['mod_skipped'] += int((1 - mx.mean(mod_mask).item()) * B * L)
            
            h = h + layer['mamba'](layer['norm1'](h)) * combined_mask
            
            if 'mem_query' in layer and use_memory:
                h = h + self._query_memory_at_layer(h, layer)
            
            if 'sparse_attn' in layer:
                h = h + layer['sparse_attn'](h) * combined_mask
            
            h = h + layer['ffn'](layer['norm2'](h)) * combined_mask
        
        if use_memory:
            self._store_memory(h)
        
        return self.output(self.norm(h))
    
    def count_params(self):
        return sum(p.size for _, p in tree_flatten(self.parameters()))
    
    def estimate_memory(self):
        params = self.count_params()
        return {'params': params, 'binary_mb': params * 0.2 / 1024 / 1024}
    
    def get_stats(self):
        total = self._stats['total']
        if total == 0:
            return {'compute_ratio': 1.0}
        return {
            'compute_ratio': 1.0 - (self._stats['mod_skipped'] / total),
            'depth_skip_ratio': self._stats['depth_skipped'] / total,
            'mod_skip_ratio': self._stats['mod_skipped'] / total,
        }
    
    def clear_memory(self):
        self.memory.clear()
        for k in self._stats:
            self._stats[k] = 0


GhostWorker = GhostWorkerV10Swarm


if __name__ == "__main__":
    print("Ghost v10-Swarm Worker")
    print("=" * 50)
    
    model = GhostWorkerV10Swarm(dim=256, num_layers=6)
    mx.eval(model.parameters())
    
    mem = model.estimate_memory()
    print(f"Params: {mem['params']:,}")
    print(f"Binary: {mem['binary_mb']:.2f} MB")
    print(f"Features: {model.FEATURES}")
    
    x = mx.array([[ord(c) for c in "kubectl get pods"]], dtype=mx.int32)
    out = model(x)
    print(f"Forward: {x.shape} → {out.shape}")
    print("✅ Ready!")
