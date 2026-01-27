"""
Ghost Worker v9 - MoD + Memory + Binary Mamba (Standalone)
==========================================================
Standalone implementation (no v8 inheritance) for gradient compatibility.
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
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden = dim * mult
        self.gate_proj = BitLinear(dim, hidden)
        self.up_proj = BitLinear(dim, hidden)
        self.down_proj = BitLinear(hidden, dim)
    
    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class GhostWorkerV9(nn.Module):
    """Ghost v9 - MoD + Memory + Binary Mamba (standalone)."""
    
    TIER = "worker"
    VERSION = "v9"
    
    def __init__(self, dim=256, num_layers=6, capacity_factor=0.5, memory_size=256):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # Embedding
        self.byte_embed = nn.Embedding(256, dim)
        
        # MoD Router
        self.mod_router = MoDRouter(dim, num_layers, min_layers=2, capacity_factor=capacity_factor)
        
        # Layers
        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                'norm1': RMSNorm(dim),
                'mamba': BinaryMamba(dim, BinaryMambaConfig()),
                'norm2': RMSNorm(dim),
                'ffn': BitFFN(dim),
            })
        
        # Memory projections (trainable)
        self.mem_query = nn.Linear(dim, dim)
        self.mem_key = nn.Linear(dim, dim)
        self.mem_value = nn.Linear(dim, dim)
        self.mem_importance = nn.Linear(dim, 1)
        self.mem_gate = nn.Linear(dim * 2, 1)
        
        # Output
        self.norm = RMSNorm(dim)
        self.output = BitLinear(dim, 256)
    
    # Memory bank stored OUTSIDE of __init__ to avoid tree_flatten
    _memory_banks = {}
    _mod_stats_store = {}
    
    @property
    def memory(self):
        if id(self) not in GhostWorkerV9._memory_banks:
            GhostWorkerV9._memory_banks[id(self)] = MemoryBank(self.dim, 256)
        return GhostWorkerV9._memory_banks[id(self)]
    
    @property 
    def _mod_stats(self):
        if id(self) not in GhostWorkerV9._mod_stats_store:
            GhostWorkerV9._mod_stats_store[id(self)] = {'skipped': 0, 'total': 0}
        return GhostWorkerV9._mod_stats_store[id(self)]
    
    def __call__(self, x, use_memory=True, use_mod=True):
        B, L = x.shape
        h = self.byte_embed(x)
        
        # Memory retrieval
        if use_memory and self.memory.size > 0:
            mem_ctx = self._retrieve_memory(h)
            gate_in = mx.concatenate([h, mem_ctx], axis=-1)
            gate = mx.sigmoid(self.mem_gate(gate_in))
            h = h + gate * mem_ctx
        
        # Layers with MoD
        for i, layer in enumerate(self.layers):
            if use_mod and i >= 2:
                mask, _ = self.mod_router(h, i)
                self._mod_stats['total'] += B * L
                self._mod_stats['skipped'] += int((1 - mx.mean(mask).item()) * B * L)
            else:
                mask = mx.ones((B, L))
            
            mask_3d = mask.reshape(B, L, 1)
            h = h + layer['mamba'](layer['norm1'](h)) * mask_3d
            h = h + layer['ffn'](layer['norm2'](h)) * mask_3d
        
        # Store to memory
        if use_memory:
            self._store_memory(h)
        
        return self.output(self.norm(h))
    
    def _retrieve_memory(self, x):
        B, L, D = x.shape
        if self.memory.size == 0:
            return mx.zeros_like(x)
        results = []
        for b in range(B):
            batch_res = []
            for t in range(L):
                q = self.mem_query(x[b, t])
                batch_res.append(self.memory.retrieve(q, top_k=4))
            results.append(mx.stack(batch_res, axis=0))
        return mx.stack(results, axis=0)
    
    def _store_memory(self, x):
        B, L, D = x.shape
        imp = mx.sigmoid(self.mem_importance(x)).squeeze(-1)
        for b in range(B):
            for t in range(L):
                if float(imp[b, t].item()) > 0.7:
                    self.memory.store(self.mem_key(x[b, t]), self.mem_value(x[b, t]), float(imp[b, t].item()))
    
    def count_params(self):
        return sum(p.size for _, p in tree_flatten(self.parameters()))
    
    def estimate_memory(self):
        params = self.count_params()
        return {'params': params, 'binary_mb': params * 0.2 / 1024 / 1024}
    
    def get_mod_stats(self):
        total = self._mod_stats['total']
        if total == 0:
            return {'compute_ratio': 1.0}
        return {'compute_ratio': 1.0 - self._mod_stats['skipped'] / total}
    
    def clear_memory(self):
        self.memory.clear()
        self._mod_stats.clear()
        self._mod_stats['skipped'] = 0
        self._mod_stats['total'] = 0


GhostWorker = GhostWorkerV9


if __name__ == "__main__":
    print("Ghost v9 Test")
    model = GhostWorkerV9(dim=256, num_layers=6)
    mx.eval(model.parameters())
    print(f"Params: {model.count_params():,}")
    x = mx.array([[ord(c) for c in "hello world"]], dtype=mx.int32)
    out = model(x)
    print(f"Forward: {x.shape} → {out.shape}")
    print("✅ Ready!")
