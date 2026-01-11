"""
Speculative Decoding Implementation
====================================

This module implements various speculative decoding techniques for LLM inference optimization.

Techniques implemented:
- Draft-Target Speculative Decoding
- Medusa Multi-Head Decoding
- Tree Attention for parallel verification

Reference:
- NVIDIA TensorRT-LLM Speculative Decoding
- "Accelerating Large Language Model Decoding with Speculative Sampling" (Leviathan et al., 2023)
- "Medusa: Simple LLM Inference Acceleration Framework" (Cai et al., 2024)
"""

from .draft_target import DraftTargetDecoder
from .medusa_heads import MedusaDecoder
from .tree_attention import TreeAttention
from .benchmark import SpeculativeDecodingBenchmark

__all__ = [
    "DraftTargetDecoder",
    "MedusaDecoder",
    "TreeAttention",
    "SpeculativeDecodingBenchmark",
]
