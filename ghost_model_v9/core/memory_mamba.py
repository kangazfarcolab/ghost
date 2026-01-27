"""
Memory Mamba - Persistent Memory for Ghost v9
==============================================
Fixed-size memory bank that persists across sequences.
Store important tokens, retrieve relevant context.

Benefit: Infinite effective context via memory retrieval
"""

import mlx.core as mx
import mlx.nn as nn
import math


class MemoryBank:
    """
    Fixed-size memory bank for storing key-value pairs.
    
    Uses FIFO eviction when full.
    Retrieval via dot-product attention.
    """
    
    def __init__(self, dim, max_entries=256):
        self.dim = dim
        self.max_entries = max_entries
        
        # Memory storage
        self.keys = []      # List of [D] vectors
        self.values = []    # List of [D] vectors
        self.importance = []  # Importance scores for eviction
    
    def store(self, key, value, importance=1.0):
        """
        Store a key-value pair in memory.
        
        Args:
            key: [D] vector for retrieval
            value: [D] vector to return
            importance: higher = less likely to evict
        """
        self.keys.append(key)
        self.values.append(value)
        self.importance.append(importance)
        
        # Evict if over capacity (remove lowest importance)
        if len(self.keys) > self.max_entries:
            min_idx = self.importance.index(min(self.importance))
            del self.keys[min_idx]
            del self.values[min_idx]
            del self.importance[min_idx]
    
    def retrieve(self, query, top_k=4):
        """
        Retrieve top-k most similar entries.
        
        Args:
            query: [D] vector
            top_k: number of entries to return
        
        Returns:
            values: weighted sum of top-k values
        """
        if len(self.keys) == 0:
            return mx.zeros((self.dim,))
        
        # Stack keys
        K = mx.stack(self.keys, axis=0)  # [N, D]
        V = mx.stack(self.values, axis=0)  # [N, D]
        
        # Compute similarities
        scores = mx.matmul(query[None, :], K.T).squeeze(0)  # [N]
        scores = scores / math.sqrt(self.dim)
        
        # Softmax over top-k
        k = min(top_k, len(self.keys))
        top_indices = mx.argsort(scores)[-k:]
        top_scores = scores[top_indices]
        top_values = V[top_indices]
        
        weights = mx.softmax(top_scores, axis=0)
        result = mx.sum(top_values * weights[:, None], axis=0)
        
        return result
    
    def clear(self):
        """Clear all memory."""
        self.keys = []
        self.values = []
        self.importance = []
    
    @property
    def size(self):
        return len(self.keys)


class MemoryMamba(nn.Module):
    """
    Mamba layer with persistent memory integration.
    
    Before processing: retrieve relevant context from memory
    After processing: optionally store important outputs
    """
    
    def __init__(self, dim, mamba_layer, memory_size=256):
        super().__init__()
        self.dim = dim
        self.mamba = mamba_layer
        
        # Memory projections
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        
        # Importance predictor (which tokens to store)
        self.importance_pred = nn.Linear(dim, 1)
        
        # Memory gate (how much to use memory vs input)
        self.memory_gate = nn.Linear(dim * 2, 1)
        
        # Memory bank
        self.memory = MemoryBank(dim, max_entries=memory_size)
    
    def retrieve_memory(self, x):
        """
        Retrieve relevant context from memory for each position.
        
        Args:
            x: [B, L, D] input
        
        Returns:
            mem_context: [B, L, D] memory-augmented context
        """
        B, L, D = x.shape
        
        if self.memory.size == 0:
            return mx.zeros_like(x)
        
        results = []
        for b in range(B):
            batch_results = []
            for t in range(L):
                query = self.query_proj(x[b, t])
                mem_val = self.memory.retrieve(query, top_k=4)
                batch_results.append(mem_val)
            results.append(mx.stack(batch_results, axis=0))
        
        return mx.stack(results, axis=0)
    
    def store_to_memory(self, x, threshold=0.7):
        """
        Store important tokens to memory.
        
        Args:
            x: [B, L, D] output from layer
            threshold: importance threshold for storage
        """
        B, L, D = x.shape
        
        # Compute importance scores
        importance = mx.sigmoid(self.importance_pred(x)).squeeze(-1)  # [B, L]
        
        for b in range(B):
            for t in range(L):
                imp = float(importance[b, t].item())
                if imp > threshold:
                    key = self.key_proj(x[b, t])
                    value = self.value_proj(x[b, t])
                    self.memory.store(key, value, imp)
    
    def __call__(self, x, use_memory=True, store_memory=True):
        """
        Forward with memory integration.
        
        Args:
            x: [B, L, D] input
            use_memory: whether to retrieve from memory
            store_memory: whether to store outputs to memory
        """
        B, L, D = x.shape
        
        # Retrieve memory context
        if use_memory and self.memory.size > 0:
            mem_context = self.retrieve_memory(x)
            
            # Gate: decide how much memory to use
            gate_input = mx.concatenate([x, mem_context], axis=-1)
            gate = mx.sigmoid(self.memory_gate(gate_input))
            
            # Blend input with memory
            x_augmented = x + gate * mem_context
        else:
            x_augmented = x
        
        # Process through Mamba
        out = self.mamba(x_augmented)
        
        # Store important outputs
        if store_memory:
            self.store_to_memory(out, threshold=0.7)
        
        return out
    
    def clear_memory(self):
        """Clear the memory bank."""
        self.memory.clear()


if __name__ == "__main__":
    from mlx.utils import tree_flatten
    
    print("Memory Mamba Test")
    print("=" * 40)
    
    dim = 256
    
    # Create a simple Mamba-like layer for testing
    class SimpleMamba(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.proj = nn.Linear(dim, dim)
        def __call__(self, x):
            return self.proj(x)
    
    mamba = SimpleMamba(dim)
    mem_mamba = MemoryMamba(dim, mamba, memory_size=64)
    mx.eval(mem_mamba.parameters())
    
    # Test
    x = mx.random.normal((2, 16, dim))
    
    # First pass (no memory)
    out1 = mem_mamba(x)
    print(f"Pass 1: memory size = {mem_mamba.memory.size}")
    
    # Second pass (with memory)
    x2 = mx.random.normal((2, 16, dim))
    out2 = mem_mamba(x2)
    print(f"Pass 2: memory size = {mem_mamba.memory.size}")
    
    # Third pass
    x3 = mx.random.normal((2, 16, dim))
    out3 = mem_mamba(x3)
    print(f"Pass 3: memory size = {mem_mamba.memory.size}")
    
    print(f"\nOutput shape: {out3.shape}")
    print("✅ Memory Mamba ready!")
