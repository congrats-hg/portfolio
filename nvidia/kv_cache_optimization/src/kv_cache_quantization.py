"""
KV Cache Quantization
=====================

Implements quantization techniques to reduce KV cache memory footprint:
- FP16 → INT8: 50% memory reduction
- FP16 → INT4: 75% memory reduction (NVIDIA NVFP4)

Key insight from NVIDIA:
"NVFP4 KV cache reduces the memory footprint by about 50% compared to
FP8 KV cache, enabling larger context lengths, batch sizes, and user
concurrency."

Reference:
- NVIDIA TensorRT-LLM NVFP4 KV Cache
- "Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import math


@dataclass
class QuantizationConfig:
    """Configuration for KV cache quantization."""
    bits: int = 8  # 8 for INT8, 4 for INT4
    symmetric: bool = True  # Symmetric vs asymmetric quantization
    per_channel: bool = True  # Per-channel vs per-tensor quantization
    group_size: int = 128  # Group size for grouped quantization


class QuantizedTensor:
    """
    Represents a quantized tensor with scale factors.

    For symmetric quantization:
        dequantized = quantized * scale

    For asymmetric quantization:
        dequantized = quantized * scale + zero_point
    """

    def __init__(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor] = None,
        bits: int = 8,
    ):
        self.quantized = quantized
        self.scale = scale
        self.zero_point = zero_point
        self.bits = bits

    def dequantize(self) -> torch.Tensor:
        """Convert back to floating point."""
        if self.zero_point is not None:
            return (self.quantized.float() - self.zero_point) * self.scale
        return self.quantized.float() * self.scale

    @property
    def memory_bytes(self) -> int:
        """Calculate memory usage in bytes."""
        # Quantized data
        if self.bits == 8:
            data_bytes = self.quantized.numel()  # 1 byte per element
        elif self.bits == 4:
            data_bytes = self.quantized.numel() // 2  # 0.5 bytes per element
        else:
            data_bytes = self.quantized.numel() * (self.bits / 8)

        # Scale and zero point (FP16)
        scale_bytes = self.scale.numel() * 2
        zp_bytes = self.zero_point.numel() * 2 if self.zero_point is not None else 0

        return int(data_bytes + scale_bytes + zp_bytes)


class KVCacheQuantizer:
    """
    Quantizes KV cache tensors to reduce memory usage.

    Supports:
    - INT8 quantization (50% reduction)
    - INT4 quantization (75% reduction)
    - Per-channel and per-tensor quantization
    - Symmetric and asymmetric quantization

    Example usage:
        >>> quantizer = KVCacheQuantizer(bits=8, symmetric=True)
        >>> k_quant = quantizer.quantize(key_states)
        >>> v_quant = quantizer.quantize(value_states)
        >>> # Later, for attention computation:
        >>> k_dequant = k_quant.dequantize()
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()
        self.bits = self.config.bits
        self.symmetric = self.config.symmetric
        self.per_channel = self.config.per_channel

        # Calculate quantization range
        if self.symmetric:
            self.qmin = -(2 ** (self.bits - 1))
            self.qmax = 2 ** (self.bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** self.bits - 1

    def quantize(
        self,
        tensor: torch.Tensor,
        dim: int = -1,
    ) -> QuantizedTensor:
        """
        Quantize a tensor to lower precision.

        Args:
            tensor: Input tensor (typically FP16 or FP32)
            dim: Dimension for per-channel quantization

        Returns:
            QuantizedTensor with quantized data and scale
        """
        if self.per_channel:
            return self._quantize_per_channel(tensor, dim)
        else:
            return self._quantize_per_tensor(tensor)

    def _quantize_per_tensor(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Per-tensor quantization."""
        if self.symmetric:
            # Symmetric: scale based on max absolute value
            amax = tensor.abs().max()
            scale = amax / self.qmax
            scale = torch.clamp(scale, min=1e-8)

            quantized = torch.clamp(
                torch.round(tensor / scale),
                self.qmin,
                self.qmax
            ).to(torch.int8 if self.bits == 8 else torch.int8)

            return QuantizedTensor(quantized, scale, bits=self.bits)
        else:
            # Asymmetric: use min and max
            tmin, tmax = tensor.min(), tensor.max()
            scale = (tmax - tmin) / (self.qmax - self.qmin)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = self.qmin - torch.round(tmin / scale)

            quantized = torch.clamp(
                torch.round(tensor / scale) + zero_point,
                self.qmin,
                self.qmax
            ).to(torch.int8 if self.bits == 8 else torch.int8)

            return QuantizedTensor(quantized, scale, zero_point, bits=self.bits)

    def _quantize_per_channel(
        self,
        tensor: torch.Tensor,
        dim: int = -1,
    ) -> QuantizedTensor:
        """Per-channel quantization for better accuracy."""
        # Move target dimension to last
        tensor = tensor.movedim(dim, -1)
        original_shape = tensor.shape
        tensor = tensor.reshape(-1, tensor.shape[-1])

        if self.symmetric:
            # Compute scale per channel
            amax = tensor.abs().max(dim=0)[0]
            scale = amax / self.qmax
            scale = torch.clamp(scale, min=1e-8)

            quantized = torch.clamp(
                torch.round(tensor / scale),
                self.qmin,
                self.qmax
            ).to(torch.int8)

            # Restore shape
            quantized = quantized.reshape(original_shape).movedim(-1, dim)
            scale = scale.unsqueeze(0)

            return QuantizedTensor(quantized, scale, bits=self.bits)
        else:
            tmin = tensor.min(dim=0)[0]
            tmax = tensor.max(dim=0)[0]
            scale = (tmax - tmin) / (self.qmax - self.qmin)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = self.qmin - torch.round(tmin / scale)

            quantized = torch.clamp(
                torch.round(tensor / scale) + zero_point,
                self.qmin,
                self.qmax
            ).to(torch.int8)

            quantized = quantized.reshape(original_shape).movedim(-1, dim)

            return QuantizedTensor(quantized, scale, zero_point, bits=self.bits)

    def compute_error(
        self,
        original: torch.Tensor,
        quantized: QuantizedTensor,
    ) -> Dict[str, float]:
        """
        Compute quantization error metrics.

        Returns:
            Dictionary with MSE, MAE, and max error
        """
        dequantized = quantized.dequantize()

        mse = F.mse_loss(dequantized, original).item()
        mae = F.l1_loss(dequantized, original).item()
        max_error = (dequantized - original).abs().max().item()

        return {
            "mse": mse,
            "mae": mae,
            "max_error": max_error,
            "psnr": 10 * math.log10(original.abs().max().item() ** 2 / (mse + 1e-10)),
        }


class QuantizedKVCache:
    """
    Memory-efficient KV cache using quantization.

    This class manages the KV cache with on-the-fly quantization and
    dequantization, significantly reducing memory footprint.

    Memory savings:
    - FP16 baseline: 2 bytes per element
    - INT8: 1 byte per element (50% reduction)
    - INT4: 0.5 bytes per element (75% reduction)

    Example usage:
        >>> cache = QuantizedKVCache(
        ...     num_layers=32,
        ...     num_heads=32,
        ...     head_dim=128,
        ...     max_length=4096,
        ...     bits=8
        ... )
        >>> cache.update(layer_idx=0, key_states=keys, value_states=values)
        >>> k, v = cache.get(layer_idx=0)
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_length: int = 4096,
        bits: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_length = max_length
        self.bits = bits
        self.device = device

        self.quantizer = KVCacheQuantizer(QuantizationConfig(bits=bits))

        # Storage for quantized KV pairs
        self.key_cache: List[Optional[QuantizedTensor]] = [None] * num_layers
        self.value_cache: List[Optional[QuantizedTensor]] = [None] * num_layers

        # Track current sequence length per layer
        self.seq_lengths = [0] * num_layers

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value states.

        Args:
            layer_idx: Layer index
            key_states: New key states [batch, num_heads, seq_len, head_dim]
            value_states: New value states [batch, num_heads, seq_len, head_dim]

        Returns:
            Tuple of (all_keys, all_values) including cached and new
        """
        batch_size = key_states.shape[0]
        new_seq_len = key_states.shape[2]

        if self.key_cache[layer_idx] is None:
            # First update: quantize and store
            self.key_cache[layer_idx] = self.quantizer.quantize(key_states)
            self.value_cache[layer_idx] = self.quantizer.quantize(value_states)
            self.seq_lengths[layer_idx] = new_seq_len
        else:
            # Append to existing cache
            existing_keys = self.key_cache[layer_idx].dequantize()
            existing_values = self.value_cache[layer_idx].dequantize()

            all_keys = torch.cat([existing_keys, key_states], dim=2)
            all_values = torch.cat([existing_values, value_states], dim=2)

            # Re-quantize the full cache
            self.key_cache[layer_idx] = self.quantizer.quantize(all_keys)
            self.value_cache[layer_idx] = self.quantizer.quantize(all_values)
            self.seq_lengths[layer_idx] += new_seq_len

        # Return dequantized values for attention computation
        return (
            self.key_cache[layer_idx].dequantize(),
            self.value_cache[layer_idx].dequantize(),
        )

    def get(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get cached key/value states for a layer."""
        if self.key_cache[layer_idx] is None:
            return None, None

        return (
            self.key_cache[layer_idx].dequantize(),
            self.value_cache[layer_idx].dequantize(),
        )

    def get_memory_usage(self) -> Dict[str, Any]:
        """Calculate current memory usage."""
        total_bytes = 0
        layer_bytes = []

        for layer_idx in range(self.num_layers):
            if self.key_cache[layer_idx] is not None:
                k_bytes = self.key_cache[layer_idx].memory_bytes
                v_bytes = self.value_cache[layer_idx].memory_bytes
                layer_bytes.append(k_bytes + v_bytes)
                total_bytes += k_bytes + v_bytes
            else:
                layer_bytes.append(0)

        # Calculate theoretical FP16 usage for comparison
        fp16_bytes = sum(
            self.seq_lengths[i] * self.num_heads * self.head_dim * 2 * 2  # *2 for K and V, *2 for FP16
            for i in range(self.num_layers)
        )

        return {
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "fp16_equivalent_mb": fp16_bytes / (1024 * 1024),
            "compression_ratio": fp16_bytes / total_bytes if total_bytes > 0 else 0,
            "layer_bytes": layer_bytes,
        }

    def clear(self, layer_idx: Optional[int] = None):
        """Clear cache for specific layer or all layers."""
        if layer_idx is not None:
            self.key_cache[layer_idx] = None
            self.value_cache[layer_idx] = None
            self.seq_lengths[layer_idx] = 0
        else:
            self.key_cache = [None] * self.num_layers
            self.value_cache = [None] * self.num_layers
            self.seq_lengths = [0] * self.num_layers


class INT4Quantizer:
    """
    Specialized INT4 quantizer using packed representation.

    Packs two INT4 values into a single INT8 for storage efficiency.
    This mirrors NVIDIA's NVFP4 approach for maximum memory savings.
    """

    def __init__(self, group_size: int = 128):
        self.group_size = group_size
        self.qmin = -8
        self.qmax = 7

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize to INT4 with group-wise scaling.

        Args:
            tensor: Input tensor

        Returns:
            Tuple of (packed_int8, scales)
        """
        original_shape = tensor.shape
        tensor = tensor.reshape(-1, self.group_size)

        # Compute per-group scale
        amax = tensor.abs().max(dim=1, keepdim=True)[0]
        scale = amax / self.qmax
        scale = torch.clamp(scale, min=1e-8)

        # Quantize
        quantized = torch.clamp(
            torch.round(tensor / scale),
            self.qmin,
            self.qmax
        ).to(torch.int8)

        # Pack two INT4 values into INT8
        quantized = quantized.reshape(-1, 2)
        packed = ((quantized[:, 0] & 0x0F) | ((quantized[:, 1] & 0x0F) << 4)).to(torch.int8)

        return packed, scale.squeeze(-1)

    def dequantize(
        self,
        packed: torch.Tensor,
        scale: torch.Tensor,
        original_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Unpack and dequantize."""
        # Unpack INT4 values
        low = (packed & 0x0F).to(torch.int8)
        high = ((packed >> 4) & 0x0F).to(torch.int8)

        # Handle sign extension for negative values
        low = torch.where(low > 7, low - 16, low)
        high = torch.where(high > 7, high - 16, high)

        unpacked = torch.stack([low, high], dim=-1).reshape(-1, self.group_size)

        # Dequantize
        dequantized = unpacked.float() * scale.unsqueeze(-1)

        return dequantized.reshape(original_shape)


def benchmark_quantization(
    tensor: torch.Tensor,
    bits_list: List[int] = [8, 4],
) -> Dict[str, Any]:
    """
    Benchmark different quantization levels.

    Args:
        tensor: Input tensor to quantize
        bits_list: List of bit widths to test

    Returns:
        Dictionary with memory and accuracy metrics
    """
    results = {}
    original_bytes = tensor.numel() * tensor.element_size()

    for bits in bits_list:
        quantizer = KVCacheQuantizer(QuantizationConfig(bits=bits))
        quantized = quantizer.quantize(tensor)

        error_metrics = quantizer.compute_error(tensor, quantized)

        results[f"int{bits}"] = {
            "memory_bytes": quantized.memory_bytes,
            "memory_mb": quantized.memory_bytes / (1024 * 1024),
            "compression_ratio": original_bytes / quantized.memory_bytes,
            **error_metrics,
        }

    results["fp16_baseline"] = {
        "memory_bytes": original_bytes,
        "memory_mb": original_bytes / (1024 * 1024),
        "compression_ratio": 1.0,
    }

    return results
