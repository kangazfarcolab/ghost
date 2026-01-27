"""
Ghost Model v11 - Ultra Compression
====================================
Ternary + Codebook compression for extreme efficiency.

Target: 1B params in 250MB with 92-95% quality.
"""

__version__ = "11.0.0"
__author__ = "Ghost Team"

from ghost_model_v11.core import (
    GhostWorkerV11,
    GhostWorker,
    TernaryLinear,
    TernaryMamba,
    CodebookLinear,
    LearnedCodebook,
)

__all__ = [
    "GhostWorkerV11",
    "GhostWorker", 
    "TernaryLinear",
    "TernaryMamba",
    "CodebookLinear",
    "LearnedCodebook",
]
