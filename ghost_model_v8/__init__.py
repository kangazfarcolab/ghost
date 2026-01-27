"""
Ghost Model v8
==============
Binary Mamba + Adaptive Depth for maximum efficiency.

Features:
- 1-bit weights (13x smaller)
- Early exit (30-50% less compute)
- SwarmMomentum training
"""
from ghost_model_v8.core import GhostWorkerV8, GhostWorker, BinaryMamba, AdaptiveDepthController
