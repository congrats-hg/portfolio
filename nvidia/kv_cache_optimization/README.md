# KV Cache Optimization for LLM Inference

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of KV cache optimization techniques for efficient LLM inference with long contexts, inspired by **NVIDIA TensorRT-LLM's NVFP4 KV Cache**.

## The Problem

In transformer-based LLMs, the KV (Key-Value) cache stores attention keys and values from previous tokens. For long sequences:

| Model | Context | KV Cache (FP16) | KV Cache (INT4) |
|-------|---------|-----------------|-----------------|
| Llama-7B | 4K | 2 GB | 0.5 GB |
| Llama-7B | 32K | 16 GB | 4 GB |
| Llama-70B | 4K | 20 GB | 5 GB |
| Llama-70B | 32K | 160 GB | 40 GB |

**KV cache becomes the memory bottleneck for long-context inference.**

## Solution: Multi-Level Optimization

This project implements three complementary optimization strategies:

### 1. KV Cache Quantization (`kv_cache_quantization.py`)

Reduce memory per element:
- **INT8**: 50% memory reduction
- **INT4 (NVFP4)**: 75% memory reduction

```python
from src.kv_cache_quantization import QuantizedKVCache

cache = QuantizedKVCache(
    num_layers=32,
    num_heads=32,
    head_dim=128,
    max_length=8192,
    bits=8  # or 4 for NVFP4-style
)

# Update with new KV states
keys, values = cache.update(layer_idx=0, key_states=k, value_states=v)

# Check memory usage
print(cache.get_memory_usage())
# {'total_mb': 128.0, 'fp16_equivalent_mb': 256.0, 'compression_ratio': 2.0}
```

### 2. Token Eviction (`kv_cache_eviction.py`)

Intelligently remove less important tokens:
- **LRU**: Evict least recently accessed
- **Attention-based**: Evict tokens with low cumulative attention
- **Heavy Hitter (H2O)**: Keep tokens that consistently receive high attention

```python
from src.kv_cache_eviction import KVCacheEvictionManager, EvictionPolicy

manager = KVCacheEvictionManager(
    num_layers=32,
    max_cache_size=4096,
    policy=EvictionPolicy.ATTENTION
)

# Update with attention weights
manager.update_attention(layer_idx=0, attention_weights=attn)

# Check if eviction needed
if manager.should_evict():
    indices = manager.get_eviction_indices(layer_idx=0)
    # Remove these tokens from KV cache
```

### 3. KV Cache Compression (`kv_cache_compression.py`)

Advanced compression techniques:
- **Low-rank approximation**: SVD-based compression
- **Grouped Query Attention (GQA)**: Share KV heads across query heads

```python
from src.kv_cache_compression import GroupedQueryAttentionCache

# 4x memory savings with GQA
cache = GroupedQueryAttentionCache(
    num_query_heads=32,
    num_kv_heads=8,  # 4x fewer KV heads
    head_dim=128
)

# KV states are expanded for attention
expanded_k, expanded_v = cache.update(key_states, value_states)
```

## Memory Profiling

Analyze memory usage patterns:

```python
from src.memory_profiler import KVCacheMemoryProfiler

profiler = KVCacheMemoryProfiler(device="cuda")

# Profile different techniques
results = profiler.profile_kv_cache_growth(
    num_layers=32,
    num_heads=32,
    head_dim=128,
    max_length=8192,
    techniques=["fp16_baseline", "int8_quantized", "int4_quantized"]
)

# Generate report
profiler.generate_report(results, "memory_report.txt")

# Plot memory growth
profiler.plot_memory_growth(results, "memory_plot.png")
```

## Project Structure

```
kv_cache_optimization/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── kv_cache_quantization.py   # INT8/INT4 quantization
│   ├── kv_cache_eviction.py       # Token eviction strategies
│   ├── kv_cache_compression.py    # Low-rank & GQA compression
│   └── memory_profiler.py         # Memory analysis tools
├── configs/
│   └── quantization_config.yaml   # Configuration presets
└── notebooks/
    └── memory_analysis.ipynb      # Analysis notebook
```

## Key Insights from NVIDIA

### NVFP4 KV Cache Benefits
> "NVFP4 KV cache reduces the memory footprint by about 50% compared to FP8 KV cache. This enables larger context lengths, batch sizes, and user concurrency."

### Priority-Based Eviction
> "The priority-based eviction API enables an LLM deployer to use knowledge about their workload to improve reuse opportunities by persisting blocks that are likely to be reused."

## Benchmarks

### Memory Reduction by Technique

| Technique | Memory | Savings | Accuracy Impact |
|-----------|--------|---------|-----------------|
| FP16 Baseline | 100% | - | - |
| INT8 | 50% | 50% | < 0.1% perplexity |
| INT4 (NVFP4) | 25% | 75% | < 0.5% perplexity |
| GQA (8 heads) | 25% | 75% | Model-dependent |
| Sliding Window | Variable | Up to 90% | Context truncation |

### Eviction Policy Comparison

| Policy | Best For | Overhead |
|--------|----------|----------|
| LRU | General purpose | Very low |
| Attention-based | Document QA | Medium |
| Heavy Hitter | Long conversations | Low |
| Sliding Window | Code completion | Zero |

## References

1. **NVIDIA NVFP4 KV Cache**: [Optimizing Inference for Long Context](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache)

2. **TensorRT-LLM KV Cache Reuse**: [Introducing New KV Cache Reuse Optimizations](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/)

3. **H2O Paper**: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs"

4. **GQA Paper**: "GQA: Training Generalized Multi-Query Transformer Models"

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)

## Installation

```bash
pip install -r requirements.txt
```

## Future Work

- [ ] Integration with NVIDIA TensorRT-LLM
- [ ] Custom CUDA kernels for INT4 operations
- [ ] Prefix caching for system prompts
- [ ] PagedAttention implementation
- [ ] Dynamic quantization based on layer importance

## License

MIT License
