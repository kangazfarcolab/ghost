"""
Ghost v20 - Unified Swarm Architecture
=======================================
The complete system:
- Shared base (800K params)
- 4 specialized workers (LoRA each)
- Math worker with learned digit tables
- Debate + consensus for all queries
- All knowledge in weights
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Tuple, Dict, Optional
import re


# ============================================================
# SHARED BASE (from v16)
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))
    
    def __call__(self, x):
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class SimpleMamba(nn.Module):
    def __init__(self, dim: int, expand: int = 2):
        super().__init__()
        self.inner_dim = dim * expand
        self.in_proj = nn.Linear(dim, self.inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(self.inner_dim, self.inner_dim, kernel_size=4, padding=3, groups=self.inner_dim)
        self.dt_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)
        self.D = mx.ones((self.inner_dim,))
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)
    
    def __call__(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_in, z = mx.split(xz, 2, axis=-1)
        x_conv = nn.silu(self.conv1d(x_in)[:, :L, :])
        dt = nn.softplus(self.dt_proj(x_conv))
        y = x_conv * mx.sigmoid(dt) + x_conv * self.D
        y = y * nn.silu(z)
        return self.out_proj(y)


class SharedBase(nn.Module):
    """Base model shared by all workers"""
    
    def __init__(self, dim: int = 128, num_layers: int = 4, vocab_size: int = 256):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, dim)
        self.max_len = 128
        self.pos_embed = mx.zeros((1, self.max_len, dim))
        
        self.blocks = [
            {'norm': RMSNorm(dim), 'mamba': SimpleMamba(dim)}
            for _ in range(num_layers)
        ]
        
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
    
    def forward_embed(self, x):
        B, L = x.shape
        h = self.embed(x)
        if L <= self.max_len:
            h = h + self.pos_embed[:, :L, :]
        return h
    
    def forward_blocks(self, h, lora=None):
        for i, block in enumerate(self.blocks):
            h = h + block['mamba'](block['norm'](h))
            if lora is not None:
                h = lora.inject(h, i)
        return h
    
    def forward_output(self, h):
        return self.output(self.norm(h))
    
    def __call__(self, x, lora=None):
        h = self.forward_embed(x)
        h = self.forward_blocks(h, lora)
        return self.forward_output(h)
    
    def count_params(self):
        from mlx.utils import tree_flatten
        return sum(p.size for _, p in tree_flatten(self.parameters()))


# ============================================================
# LORA ADAPTER
# ============================================================

class LoRAAdapter(nn.Module):
    def __init__(self, dim: int, num_layers: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.num_layers = num_layers
        self.scale = alpha / rank
        
        self.down = [nn.Linear(dim, rank, bias=False) for _ in range(num_layers)]
        self.up = [nn.Linear(rank, dim, bias=False) for _ in range(num_layers)]
        
        for u in self.up:
            u.weight = mx.zeros_like(u.weight)
    
    def inject(self, h, layer_idx):
        if layer_idx >= self.num_layers:
            return h
        return h + self.up[layer_idx](self.down[layer_idx](h)) * self.scale
    
    def count_params(self):
        from mlx.utils import tree_flatten
        return sum(p.size for _, p in tree_flatten(self.parameters()))


# ============================================================
# LEARNED MATH TABLES (from v19)
# ============================================================

class LearnedMathTables(nn.Module):
    """Addition and subtraction tables - all in weights"""
    
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        
        # Addition: 0-9 + 0-9 = 0-18
        self.add_embed = nn.Embedding(10, hidden_dim)
        self.add_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.add_fc2 = nn.Linear(hidden_dim, 19)
        
        # Subtraction: 0-18 - 0-9 = -9 to +18
        self.sub_embed = nn.Embedding(19, hidden_dim)
        self.sub_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.sub_fc2 = nn.Linear(hidden_dim, 28)  # -9 to +18
    
    def add(self, a: int, b: int) -> int:
        """Single digit addition"""
        a_arr = mx.array([a], dtype=mx.int32)
        b_arr = mx.array([b], dtype=mx.int32)
        
        emb = mx.concatenate([self.add_embed(a_arr), self.add_embed(b_arr)], axis=-1)
        h = nn.relu(self.add_fc1(emb))
        return int(mx.argmax(self.add_fc2(h)[0]).item())
    
    def sub(self, a: int, b: int) -> int:
        """Single digit subtraction"""
        a_arr = mx.array([min(a, 18)], dtype=mx.int32)
        b_arr = mx.array([min(b, 9)], dtype=mx.int32)
        
        emb = mx.concatenate([self.sub_embed(a_arr), self.add_embed(b_arr)], axis=-1)
        h = nn.relu(self.sub_fc1(emb))
        return int(mx.argmax(self.sub_fc2(h)[0]).item()) - 9
    
    def count_params(self):
        from mlx.utils import tree_flatten
        return sum(p.size for _, p in tree_flatten(self.parameters()))


# ============================================================
# SPECIALIZED WORKERS
# ============================================================

class TextWorker(nn.Module):
    """Standard text worker with LoRA"""
    
    def __init__(self, base: SharedBase, specialty: str = "text"):
        super().__init__()
        self.base = base
        self.specialty = specialty
        self.lora = LoRAAdapter(base.dim, base.num_layers, rank=8)
        self.confidence = nn.Linear(base.dim, 1)
    
    def __call__(self, x):
        return self.base(x, self.lora)
    
    def forward_with_confidence(self, x):
        h = self.base.forward_embed(x)
        h = self.base.forward_blocks(h, self.lora)
        logits = self.base.forward_output(h)
        conf = mx.sigmoid(self.confidence(mx.mean(h, axis=1)))
        return logits, conf


class MathWorker(nn.Module):
    """Math worker with multi-digit support"""
    
    def __init__(self, base: SharedBase):
        super().__init__()
        self.base = base
        self.specialty = "math"
        self.lora = LoRAAdapter(base.dim, base.num_layers, rank=8)
        self.tables = LearnedMathTables(hidden_dim=32)
        self.confidence = nn.Linear(base.dim, 1)
        
        # Multi-digit support
        self.multi_digit = None  # Will be set after training
    
    def parse_math(self, text: str) -> Optional[Tuple[str, str, str]]:
        """Parse math expression (any size numbers)"""
        match = re.match(r'^(\d+)([+\-])(\d+)=$', text.strip())
        if match:
            return match.group(1), match.group(2), match.group(3)
        return None
    
    def answer_math(self, text: str) -> Optional[str]:
        """Answer math using learned tables"""
        parsed = self.parse_math(text)
        if not parsed:
            return None
        
        a_str, op, b_str = parsed
        
        # If we have multi-digit support, use it
        if self.multi_digit is not None and op == '+':
            return self.multi_digit.add(a_str, b_str)
        
        # Fall back to single-digit table
        a, b = int(a_str), int(b_str)
        if a <= 9 and b <= 9:
            if op == '+':
                return str(self.tables.add(a, b))
            elif op == '-':
                return str(self.tables.sub(a, b))
        
        return None
    
    def __call__(self, x):
        return self.base(x, self.lora)
    
    def forward_with_confidence(self, x):
        h = self.base.forward_embed(x)
        h = self.base.forward_blocks(h, self.lora)
        logits = self.base.forward_output(h)
        conf = mx.sigmoid(self.confidence(mx.mean(h, axis=1)))
        return logits, conf


# ============================================================
# SHARED BRAIN
# ============================================================

class SharedBrain:
    """Consensus memory"""
    
    def __init__(self):
        self.consensus: Dict[str, Tuple[str, float]] = {}
        self.total_debates = 0
        self.unanimous = 0
    
    def store(self, question: str, answer: str, confidence: float, unanimous: bool):
        self.consensus[question] = (answer, confidence)
        self.total_debates += 1
        if unanimous:
            self.unanimous += 1
    
    def recall(self, question: str) -> Optional[Tuple[str, float]]:
        return self.consensus.get(question)
    
    def get_stats(self):
        return {
            "stored": len(self.consensus),
            "debates": self.total_debates,
            "unanimous_rate": self.unanimous / max(1, self.total_debates)
        }


# ============================================================
# UNIFIED SWARM
# ============================================================

class UnifiedSwarm:
    """The complete v20 system"""
    
    def __init__(self, base: SharedBase, workers: List, brain: SharedBrain):
        self.base = base
        self.workers = workers
        self.brain = brain
        
        # Find math worker
        self.math_worker = None
        for w in workers:
            if isinstance(w, MathWorker):
                self.math_worker = w
                break
    
    def generate(self, worker, prompt: str, max_tokens: int = 10) -> str:
        """Generate text from a worker"""
        tokens = [ord(c) for c in prompt]
        for _ in range(max_tokens):
            x = mx.array([tokens], dtype=mx.int32)
            logits = worker(x)
            next_token = int(mx.argmax(logits[0, -1]).item())
            if next_token == 0 or next_token == 10:
                break
            tokens.append(next_token)
        return "".join(chr(t) if 32 <= t < 127 else "" for t in tokens[len(prompt):]).strip()
    
    def answer(self, question: str) -> Tuple[str, float, str]:
        """
        Answer a question through debate.
        Returns: (answer, confidence, method)
        """
        # Check brain first
        cached = self.brain.recall(question)
        if cached and cached[1] > 0.9:
            return cached[0], cached[1], "cached"
        
        # Check if it's math (single digit)
        if self.math_worker:
            math_answer = self.math_worker.answer_math(question)
            if math_answer is not None:
                self.brain.store(question, math_answer, 1.0, True)
                return math_answer, 1.0, "math"
        
        # Otherwise: debate among all workers
        predictions = []
        confidences = []
        
        for worker in self.workers:
            pred = self.generate(worker, question)
            predictions.append(pred)
            # Simple confidence: longer = more confident
            conf = min(1.0, len(pred) / 10.0)
            confidences.append(conf)
        
        # Majority vote
        from collections import Counter
        votes = Counter(predictions)
        winner = votes.most_common(1)[0][0] if votes else ""
        
        # Average confidence
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Check if unanimous
        unanimous = len(set(predictions)) == 1
        
        # Store in brain
        self.brain.store(question, winner, avg_conf, unanimous)
        
        return winner, avg_conf, "debate"
    
    def get_stats(self) -> dict:
        return {
            "base_params": self.base.count_params(),
            "workers": len(self.workers),
            "brain": self.brain.get_stats()
        }


def create_v20_swarm() -> UnifiedSwarm:
    """Factory function to create complete v20 system"""
    
    # Shared base
    base = SharedBase(dim=128, num_layers=4)
    mx.eval(base.parameters())
    
    # Specialized workers
    workers = [
        TextWorker(base, specialty="text"),
        TextWorker(base, specialty="facts"),
        MathWorker(base),
        TextWorker(base, specialty="general"),
    ]
    
    for w in workers:
        mx.eval(w.lora.parameters())
        if hasattr(w, 'tables'):
            mx.eval(w.tables.parameters())
    
    # Brain
    brain = SharedBrain()
    
    # Swarm
    swarm = UnifiedSwarm(base, workers, brain)
    
    return swarm


if __name__ == "__main__":
    print("Ghost v20 - Unified Swarm")
    print("=" * 60)
    
    swarm = create_v20_swarm()
    
    stats = swarm.get_stats()
    print(f"\n📦 System Stats:")
    print(f"   Base: {stats['base_params']:,} params")
    print(f"   Workers: {stats['workers']}")
    
    # Quick test (untrained)
    print("\n🧪 Quick Test (untrained):")
    test_qs = ["3+5=", "France?", "Hello"]
    for q in test_qs:
        answer, conf, method = swarm.answer(q)
        print(f"   {q} → '{answer}' [{method}] (conf: {conf:.2f})")
    
    print("\n✅ v20 architecture ready!")
