"""
Ghost Worker v11 - Ultra Compression
=====================================
11 Novel Features with Ternary + Codebook compression.

Target: 1B params in 250MB with 92-95% quality retention.

Features:
1. Conv1D Tokenizer (v7)
2. Depth Predictor + Byte Importance (v7)
3. Surprise Predictor (v7)
4. Per-Layer Memory Query (v7)
5. Sparse Attention (v7)
6. Ternary Mamba (v11 NEW)
7. Codebook FFN (v11 NEW)
8. MoD - Mixture of Depths (v9)
9. Memory Bank (v9)
10. SwarmMomentum compatible (v10)
11. Adaptive Codebook (v11 NEW)
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import math
import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

from ghost_model_v12.core.ternary_linear import TernaryLinear
from ghost_model_v12.core.learned_codebook import CodebookLinear, LearnedCodebook
from ghost_model_v12.core.ternary_mamba import TernaryMamba, TernaryMambaConfig
from ghost_model_v12.core.mixture_of_depths import MoDRouter
from ghost_model_v12.core.cognitive_memory import CognitiveMemory


class RMSNorm(nn.Module):
    """RMSNorm for stable training"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))
    
    def __call__(self, x):
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class CodebookFFN(nn.Module):
    """FFN with ternary + codebook compression"""
    def __init__(self, dim, mult=4, codebook_size=256):
        super().__init__()
        hidden = dim * mult
        self.gate_proj = CodebookLinear(dim, hidden, codebook_size)
        self.up_proj = CodebookLinear(dim, hidden, codebook_size)
        self.down_proj = CodebookLinear(hidden, dim, codebook_size)
        # self.gate_proj = nn.Linear(dim, hidden, bias=False)
        # self.up_proj = nn.Linear(dim, hidden, bias=False)
        # self.down_proj = nn.Linear(hidden, dim, bias=False)
    
    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TernarySparseAttention(nn.Module):
    """Sparse attention with ternary weights"""
    def __init__(self, dim, stride=64, codebook_size=256):
        super().__init__()
        self.stride = stride
        self.dim = dim
        self.qkv = CodebookLinear(dim, dim * 3, codebook_size)
        self.out_proj = CodebookLinear(dim, dim, codebook_size)
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


class GhostWorkerV12(nn.Module):
    """
    Ghost v12: The Swarm Worker (Perceptual Layer) - STABILIZED BASE
    Includes:
    - TernaryMamba (2-bit SSM)
    - CodebookFFN (Clipped)
    - Latent Attention
    """
    
    TIER = "worker"
    VERSION = "v12"
    FEATURES = 11
    
    def __init__(self, dim=256, num_layers=6, memory_size=512, codebook_size=256):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.memory_size = memory_size
        self.codebook_size = codebook_size
        
        # Conv1D Tokenizer (v7)
        self.byte_embed = nn.Embedding(256, dim)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim)
        
        # Depth Predictor + Byte Importance (v7)
        self.depth_predictor = TernaryLinear(dim, 1)
        self.byte_importance = nn.Embedding(256, 1)
        
        # Surprise Predictor (v7)
        self.surprise_predictor = TernaryLinear(dim, 256)
        
        # MoD Router (v9)
        self.mod_router = MoDRouter(dim, num_layers, min_layers=2, capacity_factor=0.5)
        
        # Memory Projections (v9)
        self.mem_query_proj = nn.Linear(dim, dim, bias=False)
        self.mem_query_proj.weight = mx.eye(dim)
        self.mem_key_proj = nn.Linear(dim, dim, bias=False)
        self.mem_key_proj.weight = mx.eye(dim) # Identity init for one-shot
        self.mem_value_proj = nn.Linear(dim, dim, bias=False)
        self.mem_value_proj.weight = mx.eye(dim)
        self.mem_importance = TernaryLinear(dim, 1)
        self.mem_gate = TernaryLinear(dim * 2, 1)
        
        # Layers: Ternary Mamba + Codebook FFN
        self.layers = []
        for i in range(num_layers):
            layer = {
                'norm1': RMSNorm(dim),
                'mamba': TernaryMamba(dim, TernaryMambaConfig()),
                'norm2': RMSNorm(dim),
                'ffn': CodebookFFN(dim, codebook_size=codebook_size),
            }
            # Sparse Attention at layers 3, 5
            if i in [2, 4]:
                layer['sparse_attn'] = TernarySparseAttention(dim, stride=64, codebook_size=codebook_size)
            # Per-Layer Memory Query at layers 2, 4
            if i in [1, 3]:
                layer['mem_query'] = TernaryLinear(dim, dim)
                layer['mem_gate'] = TernaryLinear(dim * 2, 1)
            self.layers.append(layer)
        
        # Output
        self.norm = RMSNorm(dim)
        self.output = CodebookLinear(dim, 256, codebook_size)
    
    # Memory stored outside module tree
    _memory_banks = {}
    _stats_store = {}
    
    @property
    def memory(self):
        if id(self) not in GhostWorkerV12._memory_banks:
            GhostWorkerV12._memory_banks[id(self)] = CognitiveMemory(self.dim, max_entries=self.memory_size)
        return GhostWorkerV12._memory_banks[id(self)]
    
    @property
    def _stats(self):
        if id(self) not in GhostWorkerV12._stats_store:
            GhostWorkerV12._stats_store[id(self)] = {
                'depth_skipped': 0, 'surprise_skipped': 0, 'mod_skipped': 0, 'total': 0
            }
        return GhostWorkerV12._stats_store[id(self)]
    
    def _query_memory_at_layer(self, h, layer):
        if self.memory.size == 0:
            return mx.zeros_like(h)
        B, L, D = h.shape
        # Use mean pooling for querying at layer level
        query_context = mx.mean(h, axis=1) # [B, D]
        query_concept = layer['mem_query'](query_context) # [B, D]
        
        results = []
        for b in range(B):
            # Associative recall: concept + context
            res, _ = self.memory.recall_associative(query_concept[b], query_context[b], top_k=4)
            results.append(res)
            
        mem_output = mx.stack(results, axis=0)[:, None, :]
        mem_broadcast = mx.broadcast_to(mem_output, h.shape)
        gate_input = mx.concatenate([h, mem_broadcast], axis=-1)
        gate = mx.sigmoid(layer['mem_gate'](gate_input))
        return gate * mem_broadcast
    
    def _store_memory(self, h, curiosity_score=None):
        B, L, D = h.shape
        importance = mx.sigmoid(self.mem_importance(h)).squeeze(-1)
        
        if curiosity_score is None:
            curiosity_score = mx.zeros((B, L))
            
        for b in range(B):
            for t in range(L):
                # Importance = max(Learned Importance, Curiosity)
                # If we explicit learned to store it OR if we are confused -> Store it
                imp = max(float(importance[b, t].item()), float(curiosity_score[b, t].item()))
                
                if imp > 0.7:
                    # One-Shot Storage: Concept, Context, Outcome
                    concept = self.mem_key_proj(h[b, t])
                    context = h[b, t]
                    outcome = self.mem_value_proj(h[b, t])
                    
                    self.memory.store_one_shot(concept, context, outcome, surprise_score=imp)
    
    def _retrieve_memory(self, h):
        B, L, D = h.shape
        if self.memory.size == 0:
            return mx.zeros_like(h)
        results = []
        for b in range(B):
            batch_res = []
            for t in range(L):
                # Retrieve by current concept and context
                concept = self.mem_query_proj(h[b, t])
                context = h[b, t]
                res, _ = self.memory.recall_associative(concept, context, top_k=4)
                batch_res.append(res)
            results.append(mx.stack(batch_res, axis=0))
        return mx.stack(results, axis=0)
    
    def __call__(self, x, use_memory=True, use_routing=True):
        B, L = x.shape
        
        # Conv1D Tokenizer
        h = self.byte_embed(x)
        h = nn.silu(self.conv1d(h)[:, :L, :])
        
        # Depth Predictor
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
            
            # Curiosity: High entropy = confused = store this moment
            # Entropy H(p) = -sum(p * log(p))
            # Max H(p) for 256 classes is log(256) = 5.545
            entropy = -mx.sum(surprise_probs * mx.log(surprise_probs + 1e-9), axis=-1)
            curiosity_score = entropy / 5.545 # Normalize 0-1
        else:
            surprise_mask = mx.ones((B, L))
            curiosity_score = mx.zeros((B, L))
        
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
                # Optimized: Avoid .item() syncs during training
                # self._stats['total'] += B * L
                # self._stats['depth_skipped'] += int((1 - mx.mean(depth_mask).item()) * B * L)
                # self._stats['mod_skipped'] += int((1 - mx.mean(mod_mask).item()) * B * L)
                pass
            
            h = h + layer['mamba'](layer['norm1'](h)) * combined_mask
            
            if 'mem_query' in layer and use_memory:
                h = h + self._query_memory_at_layer(h, layer)
            
            if 'sparse_attn' in layer:
                h = h + layer['sparse_attn'](h) * combined_mask
            
            h = h + layer['ffn'](layer['norm2'](h)) * combined_mask
        
        if use_memory:
            curiosity_score = mx.maximum(curiosity_score, 0.0)
            self._store_memory(h, curiosity_score)
            
        return self.output(self.norm(h))

    def generate_step(self, x, cache=None):
        """
        Fast Generation Step using KV-Caching (SSM State).
        x: [B, 1] (Token indices)
        cache: List of states [Stem_Conv_Buf, Layer_0_Cache, Layer_1_Cache, ...]
        """
        B, L = x.shape
        
        # Initialize Cache if None
        if cache is None:
            # Stem Conv Buffer: kernel=4 -> stores last 3 inputs
            # MLX Conv1d weight shape: [out, in, k]
            conv_buf_len = self.conv1d.weight.shape[2] - 1
            stem_cache = mx.zeros((B, conv_buf_len, self.dim))
            layer_caches = [None] * self.num_layers
            cache = [stem_cache] + layer_caches

        # Unpack Cache
        stem_cache = cache[0]
        layer_caches = cache[1:]
        new_layer_caches = []
        
        # 1. Stem (Embed + Conv)
        h = self.byte_embed(x) # [B, 1, D]
        
        # Stem Conv Step
        stem_in = mx.concatenate([stem_cache, h], axis=1) # [B, 4, D]
        stem_out = self.conv1d(stem_in)[:, -1:, :] # [B, 1, D]
        h = nn.silu(stem_out)
        new_stem_cache = stem_in[:, 1:, :] 
        
        # 2. Layers
        for i, layer in enumerate(self.layers):
            # Norm
            res_input = h
            h = layer['norm1'](h)
            
            # Mamba Step
            mamba_out, new_m_cache = layer['mamba'].step(h, cache=layer_caches[i])
            new_layer_caches.append(new_m_cache)
            
            h = res_input + mamba_out
            
            # Memory Query (Stateless)
            if 'mem_query' in layer and self.memory.size > 0:
                 mem_ctx = self._query_memory_at_layer(h, layer)
                 h = h + mem_ctx
            
            # Sparse Attn (Stub: run identity or single token)
            if 'sparse_attn' in layer:
                 h = h + layer['sparse_attn'](h)

            # FFN (Stateless)
            res_input = h
            h = layer['norm2'](h)
            h = h + layer['ffn'](h)
            
        # 3. Output
        out = self.output(self.norm(h))
        
        return out, [new_stem_cache] + new_layer_caches

    def count_params(self):
        return sum(p.size for _, p in tree_flatten(self.parameters()))
    
    def estimate_storage(self):
        """Estimate storage with ternary + codebook compression"""
        params = self.count_params()
        
        ternary_bytes = params * 2 / 8  # 2 bits per weight
        codebook_bytes = self.num_layers * 256 * 4  # 256 entries × 4 bytes × layers
        
        return {
            'params': params,
            'fp32_mb': params * 4 / 1024 / 1024,
            'ternary_mb': ternary_bytes / 1024 / 1024,
            'codebook_kb': codebook_bytes / 1024,
            'total_mb': (ternary_bytes + codebook_bytes) / 1024 / 1024,
        }
    
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
            
    def sleep(self, cycles=10):
        """
        Trigger 'Sleep Learning': consolidate memories.
        
        This organizes the memory space to make retrieval more accurate.
        Call this during idle time.
        """
        loss = 0.0
        if self.memory.size > 1:
            loss = self.memory.consolidate_memories_contrastive(num_pairs=cycles)
        return loss


# Alias
GhostWorker = GhostWorkerV12


if __name__ == "__main__":
    print("Ghost v12 - Ultra Compression Worker")
    print("=" * 50)
    
    model = GhostWorkerV12(dim=256, num_layers=6)
    mx.eval(model.parameters())
    
    storage = model.estimate_storage()
    print(f"Params: {storage['params']:,}")
    print(f"FP32 size: {storage['fp32_mb']:.2f} MB")
    print(f"Ternary size: {storage['ternary_mb']:.2f} MB")
    print(f"Codebook: {storage['codebook_kb']:.1f} KB")
    print(f"Total compressed: {storage['total_mb']:.2f} MB")
    print(f"Features: {model.FEATURES}")
    
    x = mx.array([[ord(c) for c in "kubectl get pods"]], dtype=mx.int32)
    out = model(x, use_memory=False)
    print(f"\nForward: {x.shape} → {out.shape}")
    print("✅ v12 Ready!")
