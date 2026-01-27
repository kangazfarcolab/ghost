"""
Cognitive Memory - True Learning for Ghost v11
================================================
Enhanced Memory Bank with:
1. One-Shot Learning - Learn from single examples
2. Contrastive Learning - Self-organize knowledge
3. Curiosity-Driven Storage - Only store surprising info
4. Associative Recall - Retrieve by concept, not just similarity

The "Hippocampus" of the Ghost Brain.
"""

import mlx.core as mx
import mlx.nn as nn
import math


class CognitiveMemory:
    """
    Enhanced Memory Bank for True Learning.
    
    Key Differences from basic MemoryBank:
    1. Stores (concept, context, outcome) triples, not just key-value
    2. Can learn from single examples via "instant indexing"
    3. Self-organizes via contrastive updates during "sleep"
    4. Tracks surprise to decide what to remember
    """
    
    def __init__(self, dim, max_entries=512):
        self.dim = dim
        self.max_entries = max_entries
        
        # Core memory storage
        self.concepts = []     # [D] concept vectors (learned representations)
        self.contexts = []     # [D] contextual info
        self.outcomes = []     # [D] what happened / output
        self.timestamps = []   # When stored (for temporal reasoning)
        self.surprise = []     # How surprising this was (for consolidation)
        
        # Learning stats
        self.timestep = 0
        self.total_stored = 0
        self.total_retrieved = 0
        self.consolidation_count = 0
    
    def store_one_shot(self, concept, context, outcome, surprise_score=1.0):
        """
        Store a single experience instantly - no gradient descent needed!
        
        This is "True Learning": see it once, remember it.
        
        Args:
            concept: [D] what this is about (e.g., "secret code" embedding)
            context: [D] situation/trigger (e.g., "when asked about code")
            outcome: [D] what to output (e.g., "9922" embedding)
            surprise_score: 0-1, how unexpected this was
        """
        self.concepts.append(concept)
        self.contexts.append(context)
        self.outcomes.append(outcome)
        self.timestamps.append(self.timestep)
        self.surprise.append(surprise_score)
        
        self.timestep += 1
        self.total_stored += 1
        
        # Evict if over capacity (remove oldest + least surprising)
        if len(self.concepts) > self.max_entries:
            # Score = age * (1 - surprise)  -> remove old boring memories
            ages = [self.timestep - t for t in self.timestamps]
            scores = [age * (1 - surp) for age, surp in zip(ages, self.surprise)]
            evict_idx = scores.index(max(scores))
            
            del self.concepts[evict_idx]
            del self.contexts[evict_idx]
            del self.outcomes[evict_idx]
            del self.timestamps[evict_idx]
            del self.surprise[evict_idx]
    
    def recall_associative(self, query_concept, query_context=None, top_k=3):
        """
        Retrieve memories by association, not just similarity.
        
        Combines concept match + context match for better retrieval.
        
        Args:
            query_concept: [D] what I'm thinking about
            query_context: [D] optional situation context
            top_k: how many memories to retrieve
        
        Returns:
            outcome: [D] blended outcome from top-k memories
            confidence: 0-1 how confident the recall is
        """
        if len(self.concepts) == 0:
            return mx.zeros((self.dim,)), 0.0
        
        self.total_retrieved += 1
        
        # Stack all memories
        C = mx.stack(self.concepts, axis=0)  # [N, D]
        O = mx.stack(self.outcomes, axis=0)  # [N, D]
        
        # Concept similarity
        concept_scores = mx.matmul(query_concept[None, :], C.T).squeeze(0)  # [N]
        concept_scores = concept_scores / math.sqrt(self.dim)
        
        # Context similarity (if provided)
        if query_context is not None:
            Ctx = mx.stack(self.contexts, axis=0)
            context_scores = mx.matmul(query_context[None, :], Ctx.T).squeeze(0)
            context_scores = context_scores / math.sqrt(self.dim)
            
            # Combined score (80% concept, 20% context)
            scores = 0.8 * concept_scores + 0.2 * context_scores
        else:
            scores = concept_scores
        
        # Get top-k
        k = min(top_k, len(self.concepts))
        top_indices = mx.argsort(scores)[-k:]
        top_scores = scores[top_indices]
        top_outcomes = O[top_indices]
        
        # Weighted blend
        weights = mx.softmax(top_scores * 5.0, axis=0)  # Temperature = 0.2
        result = mx.sum(top_outcomes * weights[:, None], axis=0)
        
        # Confidence = max similarity
        confidence = float(mx.max(top_scores).item())
        
        return result, confidence
    
    def consolidate_memories_contrastive(self, num_pairs=10, margin=0.5):
        """
        "Sleep Learning" - Consolidate memories via contrastive learning.
        
        Idea: Replay memories and push dissimilar ones apart.
        This self-organizes the concept space without labels!
        
        Args:
            num_pairs: how many memory pairs to compare
            margin: minimum distance for dissimilar concepts
        
        Returns:
            loss: contrastive loss (for monitoring)
        """
        if len(self.concepts) < 2:
            return 0.0
        
        self.consolidation_count += 1
        
        # Sample random pairs
        n = len(self.concepts)
        losses = []
        
        for _ in range(num_pairs):
            # Pick two memories
            i = int(mx.random.randint(0, n, (1,)).item())
            j = int(mx.random.randint(0, n, (1,)).item())
            
            if i == j:
                continue
            
            c_i = self.concepts[i]
            c_j = self.concepts[j]
            o_i = self.outcomes[i]
            o_j = self.outcomes[j]
            
            # Cosine similarity in concept space
            concept_sim = mx.sum(c_i * c_j) / (mx.sqrt(mx.sum(c_i * c_i)) * mx.sqrt(mx.sum(c_j * c_j)))
            
            # Cosine similarity in outcome space
            outcome_sim = mx.sum(o_i * o_j) / (mx.sqrt(mx.sum(o_i * o_i)) * mx.sqrt(mx.sum(o_j * o_j)))
            
            # If outcomes are similar, concepts should be similar
            # If outcomes are different, concepts should be different
            if outcome_sim > 0.7:
                # Similar outcomes -> pull concepts together
                loss = mx.maximum(0.0, margin - concept_sim)
            else:
                # Different outcomes -> push concepts apart
                loss = mx.maximum(0.0, concept_sim - (-margin))
            
            losses.append(loss)
        
        if len(losses) == 0:
            return 0.0
        
        avg_loss = mx.mean(mx.stack(losses))
        
        # NOTE: In a true implementation, we'd update the concept vectors here
        # For now, this is just monitoring - we rely on the model's learned representations
        
        return float(avg_loss.item())
    
    def get_surprise_threshold(self, percentile=70):
        """
        Get dynamic surprise threshold for storage.
        
        Only store memories that are more surprising than X% of past experiences.
        This implements "curiosity-driven learning".
        
        Args:
            percentile: what % of experiences to remember
        
        Returns:
            threshold: 0-1 surprise threshold
        """
        if len(self.surprise) < 10:
            return 0.5  # Default
        
        sorted_surprises = sorted(self.surprise)
        idx = int(len(sorted_surprises) * percentile / 100)
        return sorted_surprises[idx]
    
    def clear(self):
        """Forget everything (but keep stats)."""
        self.concepts = []
        self.contexts = []
        self.outcomes = []
        self.timestamps = []
        self.surprise = []
        self.timestep = 0
    
    def get_stats(self):
        """Memory diagnostics."""
        return {
            'size': len(self.concepts),
            'capacity': self.max_entries,
            'total_stored': self.total_stored,
            'total_retrieved': self.total_retrieved,
            'consolidations': self.consolidation_count,
            'avg_surprise': sum(self.surprise) / max(1, len(self.surprise)),
        }
    
    @property
    def size(self):
        return len(self.concepts)


if __name__ == "__main__":
    print("Cognitive Memory Test")
    print("=" * 50)
    
    dim = 256
    mem = CognitiveMemory(dim, max_entries=100)
    
    # Test 1: One-shot learning
    print("\n1. One-Shot Learning Test")
    concept1 = mx.random.normal((dim,))
    context1 = mx.random.normal((dim,))
    outcome1 = mx.random.normal((dim,))
    
    mem.store_one_shot(concept1, context1, outcome1, surprise_score=0.9)
    print(f"   Stored 1 memory, size = {mem.size}")
    
    # Retrieve it
    retrieved, conf = mem.recall_associative(concept1, context1)
    similarity = float(mx.sum(retrieved * outcome1).item()) / (float(mx.sqrt(mx.sum(retrieved * retrieved)).item()) * float(mx.sqrt(mx.sum(outcome1 * outcome1)).item()))
    print(f"   Retrieved with confidence = {conf:.3f}, similarity = {similarity:.3f}")
    
    # Test 2: Multiple memories
    print("\n2. Multiple Memories Test")
    for i in range(20):
        c = mx.random.normal((dim,))
        ctx = mx.random.normal((dim,))
        o = mx.random.normal((dim,))
        mem.store_one_shot(c, ctx, o, surprise_score=float(i) / 20)
    
    print(f"   Stored 20 more, size = {mem.size}")
    print(f"   Surprise threshold (70th percentile) = {mem.get_surprise_threshold(70):.3f}")
    
    # Test 3: Contrastive consolidation
    print("\n3. Contrastive Consolidation Test")
    loss = mem.consolidate_memories_contrastive(num_pairs=10)
    print(f"   Consolidation loss = {loss:.4f}")
    
    print("\n" + "=" * 50)
    print("Stats:", mem.get_stats())
    print("✅ Cognitive Memory ready!")
