"""
Ghost Model v9
==============
MoD + Memory + Binary Mamba for ultimate efficiency.

Features:
- 1-bit weights (20x compression)
- Per-token depth routing (50% compute savings)
- Persistent memory (infinite context)
"""
from ghost_model_v9.core import GhostWorkerV9, GhostWorker, MoDRouter, MemoryBank
