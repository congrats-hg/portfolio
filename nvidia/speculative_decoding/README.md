# Speculative Decoding for LLM Inference Acceleration

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive implementation of speculative decoding techniques for accelerating Large Language Model (LLM) inference, inspired by **NVIDIA TensorRT-LLM** optimizations.

## Overview

Speculative decoding is a key technique used in NVIDIA's TensorRT-LLM to achieve up to **3.55x throughput improvement** on Llama 3.3 70B. This project implements the core algorithms from scratch in PyTorch.

### Key Insight

> Standard autoregressive decoding generates one token per forward pass. Speculative decoding generates multiple tokens by having a fast "draft" model propose candidates, then verifying them in parallel with the target model. Since verification of K tokens costs approximately the same as generating 1 token (due to parallelism), we gain significant speedup when drafts are accepted.

## Implemented Techniques

### 1. Draft-Target Speculative Decoding (`draft_target.py`)

The classic approach using two models:
- **Draft Model**: Small, fast model (e.g., GPT-2) generates K candidate tokens
- **Target Model**: Large, accurate model (e.g., GPT-2-XL) verifies candidates in one forward pass

```python
from src.draft_target import DraftTargetDecoder, SpeculativeDecodingConfig

config = SpeculativeDecodingConfig(
    num_speculative_tokens=5,
    temperature=1.0,
    max_new_tokens=100
)

decoder = DraftTargetDecoder(
    draft_model=draft_model,
    target_model=target_model,
    tokenizer=tokenizer,
    config=config
)

output, metrics = decoder.generate("Once upon a time")
print(f"Acceptance rate: {metrics.acceptance_rate:.2%}")
print(f"Speedup: {metrics.speedup_factor:.2f}x")
```

### 2. Medusa Multi-Head Decoding (`medusa_heads.py`)

Single model with additional lightweight heads:
- No separate draft model needed
- Each head predicts token at different future position
- Candidates verified via tree attention

```python
from src.medusa_heads import MedusaDecoder, MedusaConfig

config = MedusaConfig(
    num_heads=4,
    num_candidates_per_head=5,
    tree_width=64
)

decoder = MedusaDecoder(
    base_model=model,
    tokenizer=tokenizer,
    config=config
)

output, metrics = decoder.generate("The future of AI is")
print(f"Tokens per iteration: {metrics['avg_tokens_per_iteration']:.2f}")
```

### 3. Tree Attention (`tree_attention.py`)

Efficient parallel verification using tree-structured attention:
- Shares computation for common prefixes
- Custom attention masks for tree structure
- Enables batch verification of all candidates

```python
from src.tree_attention import SpeculationTree, TreeVerifier

# Build speculation tree from candidates
tree = SpeculationTree()
tree.add_candidate([101, 202, 303])
tree.add_candidate([101, 202, 404])
tree.add_candidate([101, 505, 606])

# Visualize tree structure
#     101
#    /   \
#  202   505
#  / \     \
# 303 404  606

# Get attention mask for parallel verification
attention_mask = tree.get_attention_mask()
```

## Benchmarking

Run comprehensive benchmarks comparing autoregressive vs speculative decoding:

```bash
cd speculative_decoding
python -m src.benchmark
```

### Expected Results

| Method | Throughput (tok/s) | Acceptance Rate | Speedup |
|--------|-------------------|-----------------|---------|
| Autoregressive | 25-30 | - | 1.0x |
| Speculative (K=5) | 50-80 | 65-75% | 1.8-2.5x |
| Medusa (4 heads) | 60-90 | 70-80% | 2.0-3.0x |

*Results vary based on model size, hardware, and task type.*

## Project Structure

```
speculative_decoding/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── draft_target.py      # Draft-target speculative decoding
│   ├── medusa_heads.py      # Medusa multi-head approach
│   ├── tree_attention.py    # Tree attention implementation
│   └── benchmark.py         # Benchmarking suite
├── notebooks/
│   └── analysis.ipynb       # Results analysis & visualization
└── results/
    └── benchmark_results.json
```

## Key Algorithms

### Speculative Sampling

The core acceptance/rejection logic ensures outputs are **mathematically identical** to standard sampling:

```
For each draft token x_i with draft probability q(x_i) and target probability p(x_i):
  - Accept with probability: min(1, p(x_i) / q(x_i))
  - On rejection: sample from adjusted distribution max(0, p - q)
```

This guarantees that the final distribution matches the target model exactly.

### Tree Attention Mask

For candidate sequences sharing prefixes:
- Nodes only attend to ancestors (not siblings)
- Enables parallel verification of all paths
- Reduces O(K * N) verifications to O(N) with tree structure

## References

1. **NVIDIA TensorRT-LLM**: [Boost Llama 3.3 70B Inference 3x](https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/)

2. **Speculative Sampling Paper**: Leviathan et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023)

3. **Medusa Paper**: Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (2024)

4. **ReDrafter**: Apple ML Research, "Accelerating LLM Inference on NVIDIA GPUs with ReDrafter"

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- Transformers 4.35+

## Installation

```bash
pip install -r requirements.txt
```

## Future Work

- [ ] Integration with NVIDIA TensorRT-LLM for production deployment
- [ ] CUDA kernels for optimized tree attention
- [ ] Support for Eagle and Lookahead decoding
- [ ] Multi-GPU speculative decoding
- [ ] Dynamic speculation length based on acceptance rate

## License

MIT License
