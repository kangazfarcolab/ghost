"""
memory.py - Shared Knowledge Lake for Ghost Swarm

Concept: A centralized vector store that all agents can read/write to.
For now, implemented as a simple in-memory list with semantic matching mock.
"""

import time
import json
import mlx.core as mx

class SharedMemory:
    def __init__(self):
        self.facts = []  # List of {"txt": str, "vector": array, "source": str}
        self.logs = []
    
    def write(self, text, source_agent, embedding=None):
        """Agent writes a fact to shared memory."""
        entry = {
            "text": text,
            "source": source_agent,
            "timestamp": time.time(),
            "embedding": embedding
        }
        self.facts.append(entry)
        self.log(f"[{source_agent}] Wrote: {text[:50]}...")
    
    def query(self, query_text, limit=3):
        """Find relevant facts (Mock semantic search)."""
        # In real version, we'd use cosine similarity on embeddings
        # Here we just do keyword matching for the mock
        results = []
        q_words = set(query_text.lower().split())
        
        for fact in self.facts:
            f_words = set(fact["text"].lower().split())
            overlap = len(q_words.intersection(f_words))
            if overlap > 0:
                results.append((overlap, fact))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:limit]]
    
    def log(self, message):
        """Centralized logging."""
        self.logs.append(f"{time.strftime('%H:%M:%S')} - {message}")
        print(f"🧠 [Memory] {message}")

# Global instance
lake = SharedMemory()
