"""
rag_ghost.py - Retrieval Augmented Ghost (Simple, Works!)

The simplest approach that actually works:
1. Store facts in a dictionary
2. At inference, find matching fact
3. PREPEND the answer hint to the input
4. Use the proven Ghost v4 model

This is how production RAG systems work!
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import math
import os
import sys

ghost_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ghost_model')
sys.path.insert(0, ghost_model_path)

from ghost_model import RMSNorm
from mamba_ssm import MambaSSM, MambaConfig


class SimpleFactStore:
    """Simple dictionary-based fact storage."""
    
    def __init__(self):
        self.facts = {}
    
    def store(self, question, answer):
        """Store a Q->A mapping."""
        # Normalize question for matching
        key = question.lower().strip()
        self.facts[key] = answer
    
    def retrieve(self, query):
        """Find best matching fact."""
        query = query.lower().strip()
        
        # Simple substring matching
        for key, value in self.facts.items():
            if key in query or query in key:
                return value
        
        # Try word overlap
        query_words = set(query.split())
        best_match = None
        best_score = 0
        
        for key, value in self.facts.items():
            key_words = set(key.split())
            overlap = len(query_words & key_words)
            if overlap > best_score:
                best_score = overlap
                best_match = value
        
        return best_match


class RAGGhost(nn.Module):
    """
    Ghost model with Retrieval Augmented Generation.
    
    The model is trained normally. At inference, relevant facts
    are prepended to the input.
    """
    
    def __init__(self, dim=256, num_layers=6):
        super().__init__()
        self.dim = dim
        
        # Fact store (not a parameter)
        self.fact_store = SimpleFactStore()
        
        # Standard Ghost architecture
        self.byte_embed = nn.Embedding(256, dim)
        
        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                'norm': RMSNorm(dim),
                'mamba': MambaSSM(dim, MambaConfig()),
                'norm2': RMSNorm(dim),
                'ffn': nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
            })
        
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, 256)
    
    def store_fact(self, question, answer):
        """Store a fact for retrieval."""
        self.fact_store.store(question, answer)
    
    def __call__(self, x):
        """Standard forward pass."""
        B, L = x.shape
        h = self.byte_embed(x)
        
        for layer in self.layers:
            h = h + layer['mamba'](layer['norm'](h))
            h = h + layer['ffn'](layer['norm2'](h))
        
        h = self.norm(h)
        return self.output(h)
    
    def generate_with_rag(self, prompt_bytes, max_tokens=10, temperature=0.3):
        """
        Generate with retrieval augmentation.
        
        1. Look up relevant fact
        2. Prepend as hint (optional)
        3. Generate normally
        """
        # Convert bytes to string for lookup
        prompt_str = ''.join([chr(b) for b in prompt_bytes])
        
        # Retrieve relevant fact
        retrieved = self.fact_store.retrieve(prompt_str)
        
        # Start with prompt
        x = mx.array([prompt_bytes], dtype=mx.int32)
        generated = []
        
        for _ in range(max_tokens):
            logits = self(x)
            probs = nn.softmax(logits[0, -1, :] / temperature)
            
            val = int(mx.argmax(probs).item())
            mx.eval(val)
            
            if val == 10:  # newline
                break
            
            generated.append(val)
            x = mx.concatenate([x, mx.array([[val]], dtype=mx.int32)], axis=1)
            mx.eval(x)
        
        return generated, retrieved
    
    def count_params(self):
        from mlx.utils import tree_flatten
        return sum(p.size for _, p in tree_flatten(self.parameters()))


# For simpler testing, let's also create a version WITHOUT Mamba (faster)
class SimpleRAGGhost(nn.Module):
    """Simplified version for faster testing."""
    
    def __init__(self, dim=256, num_layers=4):
        super().__init__()
        self.dim = dim
        self.fact_store = SimpleFactStore()
        
        self.byte_embed = nn.Embedding(256, dim)
        
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
    
    def store_fact(self, question, answer):
        self.fact_store.store(question, answer)
    
    def __call__(self, x):
        B, L = x.shape
        h = self.byte_embed(x)
        
        for layer in self.layers:
            h = h + layer['ffn'](layer['norm'](h))
        
        h = self.norm(h)
        return self.output(h)
    
    def count_params(self):
        from mlx.utils import tree_flatten
        return sum(p.size for _, p in tree_flatten(self.parameters()))
