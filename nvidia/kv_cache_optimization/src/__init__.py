"""
KV Cache Optimization Techniques
================================

This module implements various KV cache optimization techniques for
efficient LLM inference with long contexts.

Techniques implemented:
- KV Cache Quantization (FP16 → INT8 → INT4)
- Token Eviction Strategies (LRU, Attention-based)
- Sliding Window Attention
- KV Cache Compression

Reference:
- NVIDIA TensorRT-LLM NVFP4 KV Cache
- "Optimizing Inference for Long Context with NVFP4 KV Cache"
- "Introducing New KV Cache Reuse Optimizations in NVIDIA TensorRT-LLM"
"""

from .kv_cache_quantization import KVCacheQuantizer, QuantizedKVCache
from .kv_cache_eviction import KVCacheEvictionManager, AttentionBasedEviction
from .kv_cache_compression import KVCacheCompressor
from .memory_profiler import KVCacheMemoryProfiler

__all__ = [
    "KVCacheQuantizer",
    "QuantizedKVCache",
    "KVCacheEvictionManager",
    "AttentionBasedEviction",
    "KVCacheCompressor",
    "KVCacheMemoryProfiler",
]
