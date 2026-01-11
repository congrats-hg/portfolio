"""
KV Cache Compression
====================

Advanced compression techniques for KV cache beyond simple quantization:
- Low-rank approximation
- Attention-aware compression
- Grouped key-value sharing

Reference:
- "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
- "MQA: Multi-Query Attention"
- "Efficient Memory Management for Large Language Model Serving with PagedAttention"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class CompressionConfig:
    """Configuration for KV cache compression."""
    compression_ratio: float = 0.5  # Target compression ratio
    rank: int = 64  # Rank for low-rank approximation
    num_kv_heads: int = 8  # For GQA-style compression (vs 32 query heads)
    use_svd: bool = True  # Use SVD for compression
    threshold: float = 0.01  # Singular value threshold


class LowRankKVCompressor:
    """
    Compresses KV cache using low-rank approximation.

    Uses truncated SVD to approximate KV matrices with lower rank,
    reducing memory while preserving most information.

    Memory savings: O(n*d) -> O(n*r + r*d) where r << min(n, d)
    """

    def __init__(
        self,
        rank: int = 64,
        adaptive_rank: bool = True,
        energy_threshold: float = 0.95,  # Keep 95% of energy
    ):
        self.rank = rank
        self.adaptive_rank = adaptive_rank
        self.energy_threshold = energy_threshold

    def compress(
        self,
        tensor: torch.Tensor,
        target_rank: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress tensor using truncated SVD.

        Args:
            tensor: [batch, heads, seq_len, head_dim] or [seq_len, hidden]
            target_rank: Override default rank

        Returns:
            Tuple of (U, S, V) where tensor ≈ U @ diag(S) @ V^T
        """
        rank = target_rank or self.rank
        original_shape = tensor.shape

        # Flatten to 2D for SVD
        if tensor.dim() == 4:
            batch, heads, seq_len, head_dim = tensor.shape
            tensor = tensor.reshape(batch * heads, seq_len, head_dim)
            tensor = tensor.reshape(-1, head_dim)

        # Perform SVD
        try:
            U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
        except RuntimeError:
            # Fall back to randomized SVD for large matrices
            U, S, Vh = self._randomized_svd(tensor, rank)

        if self.adaptive_rank:
            # Determine rank based on energy threshold
            total_energy = (S ** 2).sum()
            cumulative_energy = (S ** 2).cumsum(dim=0) / total_energy
            rank = (cumulative_energy < self.energy_threshold).sum().item() + 1
            rank = max(1, min(rank, self.rank))

        # Truncate
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        return U, S, Vh

    def _randomized_svd(
        self,
        tensor: torch.Tensor,
        rank: int,
        num_oversamples: int = 10,
        num_iterations: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomized SVD for large matrices.

        Uses power iteration with random projections for efficiency.
        """
        m, n = tensor.shape
        k = min(rank + num_oversamples, min(m, n))

        # Random projection
        Q = torch.randn(n, k, device=tensor.device, dtype=tensor.dtype)

        # Power iterations
        for _ in range(num_iterations):
            Q = tensor @ Q
            Q, _ = torch.linalg.qr(Q)
            Q = tensor.T @ Q
            Q, _ = torch.linalg.qr(Q)

        Q = tensor @ Q
        Q, _ = torch.linalg.qr(Q)

        # Project and compute SVD
        B = Q.T @ tensor
        Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
        U = Q @ Ub

        return U[:, :rank], S[:rank], Vh[:rank, :]

    def decompress(
        self,
        U: torch.Tensor,
        S: torch.Tensor,
        Vh: torch.Tensor,
        original_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Reconstruct tensor from SVD components."""
        # Reconstruct
        reconstructed = U @ torch.diag(S) @ Vh

        # Reshape to original
        return reconstructed.reshape(original_shape)

    def get_compression_ratio(
        self,
        original_shape: Tuple[int, ...],
        rank: int,
    ) -> float:
        """Calculate actual compression ratio."""
        if len(original_shape) == 4:
            batch, heads, seq_len, head_dim = original_shape
            original_size = batch * heads * seq_len * head_dim
        else:
            seq_len, head_dim = original_shape
            original_size = seq_len * head_dim
            batch = 1

        # Compressed size: U (n*r) + S (r) + V (r*d)
        compressed_size = seq_len * rank + rank + rank * head_dim

        return original_size / compressed_size


class GroupedQueryAttentionCache:
    """
    Implements GQA-style KV cache with fewer KV heads than query heads.

    In standard MHA: Q, K, V all have num_heads
    In GQA: Q has num_heads, K/V have num_kv_heads (< num_heads)
    Multiple query heads share the same KV head.

    Memory savings: (num_heads / num_kv_heads)x for KV cache
    """

    def __init__(
        self,
        num_query_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        max_length: int = 4096,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_length = max_length
        self.device = device

        assert num_query_heads % num_kv_heads == 0, \
            "num_query_heads must be divisible by num_kv_heads"
        self.num_groups = num_query_heads // num_kv_heads

        # Smaller KV cache
        self.key_cache = None
        self.value_cache = None
        self.current_length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with grouped KV states.

        Args:
            key_states: [batch, num_kv_heads, seq_len, head_dim]
            value_states: [batch, num_kv_heads, seq_len, head_dim]

        Returns:
            Expanded KV states for attention: [batch, num_query_heads, seq_len, head_dim]
        """
        batch_size = key_states.shape[0]
        seq_len = key_states.shape[2]

        if self.key_cache is None:
            self.key_cache = key_states
            self.value_cache = value_states
        else:
            self.key_cache = torch.cat([self.key_cache, key_states], dim=2)
            self.value_cache = torch.cat([self.value_cache, value_states], dim=2)

        self.current_length = self.key_cache.shape[2]

        # Expand for attention computation
        # [batch, kv_heads, seq, dim] -> [batch, q_heads, seq, dim]
        expanded_keys = self.key_cache.repeat_interleave(self.num_groups, dim=1)
        expanded_values = self.value_cache.repeat_interleave(self.num_groups, dim=1)

        return expanded_keys, expanded_values

    def get_memory_usage(self) -> Dict[str, float]:
        """Compare memory usage to standard MHA."""
        if self.key_cache is None:
            return {"current_mb": 0, "mha_equivalent_mb": 0, "savings_ratio": 0}

        current_bytes = self.key_cache.numel() * 2 * 2  # K + V, FP16
        mha_bytes = current_bytes * self.num_groups

        return {
            "current_mb": current_bytes / (1024 * 1024),
            "mha_equivalent_mb": mha_bytes / (1024 * 1024),
            "savings_ratio": self.num_groups,
            "num_kv_heads": self.num_kv_heads,
            "num_query_heads": self.num_query_heads,
        }

    def clear(self):
        self.key_cache = None
        self.value_cache = None
        self.current_length = 0


class KVCacheCompressor:
    """
    Unified compression interface for KV cache.

    Combines multiple compression techniques:
    1. Quantization (INT8/INT4)
    2. Low-rank approximation
    3. Token eviction

    Example usage:
        >>> compressor = KVCacheCompressor(
        ...     compression_ratio=0.5,
        ...     use_quantization=True,
        ...     use_low_rank=True
        ... )
        >>> compressed_k, compressed_v = compressor.compress(keys, values)
        >>> k, v = compressor.decompress(compressed_k, compressed_v)
    """

    def __init__(
        self,
        compression_ratio: float = 0.5,
        use_quantization: bool = True,
        quantization_bits: int = 8,
        use_low_rank: bool = False,
        low_rank: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.compression_ratio = compression_ratio
        self.use_quantization = use_quantization
        self.use_low_rank = use_low_rank
        self.device = device

        if use_quantization:
            from .kv_cache_quantization import KVCacheQuantizer, QuantizationConfig
            self.quantizer = KVCacheQuantizer(
                QuantizationConfig(bits=quantization_bits)
            )

        if use_low_rank:
            self.low_rank_compressor = LowRankKVCompressor(rank=low_rank)

    def compress(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[Any, Any]:
        """
        Compress KV states using configured techniques.

        Args:
            key_states: [batch, heads, seq_len, head_dim]
            value_states: [batch, heads, seq_len, head_dim]

        Returns:
            Tuple of compressed (keys, values)
        """
        compressed_keys = key_states
        compressed_values = value_states

        if self.use_low_rank:
            # Apply low-rank compression first
            k_u, k_s, k_v = self.low_rank_compressor.compress(key_states)
            v_u, v_s, v_v = self.low_rank_compressor.compress(value_states)
            compressed_keys = (k_u, k_s, k_v, key_states.shape)
            compressed_values = (v_u, v_s, v_v, value_states.shape)

        if self.use_quantization:
            if self.use_low_rank:
                # Quantize SVD components
                k_u, k_s, k_v, shape = compressed_keys
                v_u, v_s, v_v, shape = compressed_values
                compressed_keys = (
                    self.quantizer.quantize(k_u),
                    k_s,  # Keep singular values in FP
                    self.quantizer.quantize(k_v),
                    shape
                )
                compressed_values = (
                    self.quantizer.quantize(v_u),
                    v_s,
                    self.quantizer.quantize(v_v),
                    shape
                )
            else:
                compressed_keys = self.quantizer.quantize(key_states)
                compressed_values = self.quantizer.quantize(value_states)

        return compressed_keys, compressed_values

    def decompress(
        self,
        compressed_keys: Any,
        compressed_values: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress KV states.

        Args:
            compressed_keys: Compressed key representation
            compressed_values: Compressed value representation

        Returns:
            Tuple of (key_states, value_states)
        """
        if self.use_quantization and self.use_low_rank:
            # Dequantize SVD components
            k_u_q, k_s, k_v_q, k_shape = compressed_keys
            v_u_q, v_s, v_v_q, v_shape = compressed_values

            k_u = k_u_q.dequantize()
            k_v = k_v_q.dequantize()
            v_u = v_u_q.dequantize()
            v_v = v_v_q.dequantize()

            # Reconstruct from SVD
            keys = self.low_rank_compressor.decompress(k_u, k_s, k_v, k_shape)
            values = self.low_rank_compressor.decompress(v_u, v_s, v_v, v_shape)

        elif self.use_quantization:
            keys = compressed_keys.dequantize()
            values = compressed_values.dequantize()

        elif self.use_low_rank:
            k_u, k_s, k_v, k_shape = compressed_keys
            v_u, v_s, v_v, v_shape = compressed_values
            keys = self.low_rank_compressor.decompress(k_u, k_s, k_v, k_shape)
            values = self.low_rank_compressor.decompress(v_u, v_s, v_v, v_shape)

        else:
            keys, values = compressed_keys, compressed_values

        return keys, values

    def estimate_memory_savings(
        self,
        original_shape: Tuple[int, ...],
    ) -> Dict[str, float]:
        """Estimate memory savings for given tensor shape."""
        batch, heads, seq_len, head_dim = original_shape
        original_bytes = batch * heads * seq_len * head_dim * 2  # FP16

        # Quantization savings
        if self.use_quantization:
            quant_ratio = 2 / (self.quantizer.bits / 8)  # FP16 vs INT8/INT4
        else:
            quant_ratio = 1.0

        # Low-rank savings
        if self.use_low_rank:
            r = self.low_rank_compressor.rank
            low_rank_ratio = (seq_len * head_dim) / (seq_len * r + r + r * head_dim)
        else:
            low_rank_ratio = 1.0

        total_ratio = quant_ratio * low_rank_ratio
        compressed_bytes = original_bytes / total_ratio

        return {
            "original_mb": original_bytes / (1024 * 1024),
            "compressed_mb": compressed_bytes / (1024 * 1024),
            "compression_ratio": total_ratio,
            "memory_saved_mb": (original_bytes - compressed_bytes) / (1024 * 1024),
        }
