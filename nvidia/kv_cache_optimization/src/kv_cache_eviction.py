"""
KV Cache Eviction Strategies
============================

Implements various token eviction policies for managing KV cache size:
- LRU (Least Recently Used)
- Attention-based eviction
- Priority-based eviction

Key insight from NVIDIA:
"The priority-based eviction API enables an LLM deployer to use knowledge
about their workload to improve reuse opportunities by persisting blocks
that are likely to be reused."

Reference:
- NVIDIA TensorRT-LLM KV Cache Reuse
- "Efficient Memory Management for Large Language Model Serving with PagedAttention"
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict
import heapq


class EvictionPolicy(Enum):
    """Available eviction policies."""
    LRU = "lru"  # Least Recently Used
    ATTENTION = "attention"  # Based on attention scores
    PRIORITY = "priority"  # User-defined priorities
    SLIDING_WINDOW = "sliding_window"  # Fixed window
    HEAVY_HITTER = "heavy_hitter"  # Keep tokens with highest cumulative attention


@dataclass
class EvictionConfig:
    """Configuration for KV cache eviction."""
    max_cache_size: int = 4096  # Maximum tokens in cache
    eviction_policy: EvictionPolicy = EvictionPolicy.ATTENTION
    sliding_window_size: int = 2048  # For sliding window policy
    attention_decay: float = 0.99  # Decay factor for attention scores
    min_protected_tokens: int = 64  # Never evict first N tokens (system prompt)
    eviction_batch_size: int = 128  # Evict this many tokens at once


class LRUEvictionManager:
    """
    LRU-based KV cache eviction.

    Tracks access order and evicts least recently accessed tokens.
    Simple but effective for many workloads.
    """

    def __init__(self, max_size: int = 4096, min_protected: int = 64):
        self.max_size = max_size
        self.min_protected = min_protected
        self.access_order: OrderedDict[int, int] = OrderedDict()  # token_idx -> access_count
        self.current_size = 0

    def access(self, token_indices: List[int]):
        """Record access to tokens."""
        for idx in token_indices:
            if idx in self.access_order:
                self.access_order.move_to_end(idx)
            else:
                self.access_order[idx] = 1
                self.current_size += 1

    def get_eviction_candidates(self, num_to_evict: int) -> List[int]:
        """Get tokens to evict (oldest accessed, excluding protected)."""
        candidates = []
        for idx in self.access_order:
            if idx >= self.min_protected and len(candidates) < num_to_evict:
                candidates.append(idx)
            if len(candidates) >= num_to_evict:
                break
        return candidates

    def evict(self, indices: List[int]):
        """Remove tokens from tracking."""
        for idx in indices:
            if idx in self.access_order:
                del self.access_order[idx]
                self.current_size -= 1

    def should_evict(self) -> bool:
        """Check if eviction is needed."""
        return self.current_size > self.max_size


class AttentionBasedEviction:
    """
    Eviction based on cumulative attention scores.

    Tokens with consistently low attention are candidates for eviction.
    This is more intelligent than LRU as it considers semantic importance.

    Algorithm:
    1. Track cumulative attention score for each token
    2. Apply decay over time (recent attention matters more)
    3. Evict tokens with lowest cumulative scores
    """

    def __init__(
        self,
        max_size: int = 4096,
        min_protected: int = 64,
        decay: float = 0.99,
    ):
        self.max_size = max_size
        self.min_protected = min_protected
        self.decay = decay

        # Cumulative attention scores per token
        self.attention_scores: Dict[int, float] = {}
        self.current_size = 0

    def update_scores(
        self,
        attention_weights: torch.Tensor,
        token_indices: Optional[List[int]] = None,
    ):
        """
        Update attention scores based on new attention weights.

        Args:
            attention_weights: [batch, heads, query_len, key_len]
            token_indices: Optional mapping of cache positions to original indices
        """
        # Average attention across batch, heads, and queries
        avg_attention = attention_weights.mean(dim=(0, 1, 2))  # [key_len]

        for i, score in enumerate(avg_attention.tolist()):
            idx = token_indices[i] if token_indices else i

            if idx in self.attention_scores:
                # Decay existing score and add new
                self.attention_scores[idx] = self.attention_scores[idx] * self.decay + score
            else:
                self.attention_scores[idx] = score
                self.current_size += 1

    def get_eviction_candidates(self, num_to_evict: int) -> List[int]:
        """Get tokens with lowest attention scores."""
        # Sort by score (ascending)
        sorted_tokens = sorted(
            self.attention_scores.items(),
            key=lambda x: x[1]
        )

        candidates = []
        for idx, score in sorted_tokens:
            if idx >= self.min_protected and len(candidates) < num_to_evict:
                candidates.append(idx)
            if len(candidates) >= num_to_evict:
                break

        return candidates

    def evict(self, indices: List[int]):
        """Remove tokens from tracking."""
        for idx in indices:
            if idx in self.attention_scores:
                del self.attention_scores[idx]
                self.current_size -= 1

    def should_evict(self) -> bool:
        return self.current_size > self.max_size


class HeavyHitterEviction:
    """
    Heavy Hitter (H2O) based eviction.

    Based on the observation that a small subset of tokens ("heavy hitters")
    receive most of the attention across many queries.

    Reference: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs"
    """

    def __init__(
        self,
        max_size: int = 4096,
        min_protected: int = 64,
        heavy_hitter_ratio: float = 0.2,  # Keep top 20% as heavy hitters
    ):
        self.max_size = max_size
        self.min_protected = min_protected
        self.heavy_hitter_ratio = heavy_hitter_ratio

        self.cumulative_attention: Dict[int, float] = {}
        self.access_count: Dict[int, int] = {}
        self.current_size = 0

    def update(
        self,
        attention_weights: torch.Tensor,
        new_tokens: int = 1,
    ):
        """
        Update heavy hitter tracking with new attention.

        Args:
            attention_weights: [batch, heads, query_len, key_len]
            new_tokens: Number of new tokens added
        """
        # Sum attention received by each key position
        attention_sum = attention_weights.sum(dim=(0, 1, 2))  # [key_len]

        for i, attn in enumerate(attention_sum.tolist()):
            if i in self.cumulative_attention:
                self.cumulative_attention[i] += attn
                self.access_count[i] += 1
            else:
                self.cumulative_attention[i] = attn
                self.access_count[i] = 1
                self.current_size += 1

    def identify_heavy_hitters(self) -> List[int]:
        """Identify tokens that are heavy hitters."""
        if not self.cumulative_attention:
            return []

        # Calculate average attention per access
        avg_attention = {
            idx: self.cumulative_attention[idx] / self.access_count[idx]
            for idx in self.cumulative_attention
        }

        # Sort by average attention
        sorted_tokens = sorted(
            avg_attention.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Top heavy_hitter_ratio are heavy hitters
        num_heavy = max(1, int(len(sorted_tokens) * self.heavy_hitter_ratio))
        return [idx for idx, _ in sorted_tokens[:num_heavy]]

    def get_eviction_candidates(self, num_to_evict: int) -> List[int]:
        """Get non-heavy-hitter tokens for eviction."""
        heavy_hitters = set(self.identify_heavy_hitters())

        candidates = []
        for idx in sorted(self.cumulative_attention.keys()):
            if idx >= self.min_protected and idx not in heavy_hitters:
                candidates.append(idx)
            if len(candidates) >= num_to_evict:
                break

        return candidates

    def evict(self, indices: List[int]):
        """Remove tokens."""
        for idx in indices:
            if idx in self.cumulative_attention:
                del self.cumulative_attention[idx]
                del self.access_count[idx]
                self.current_size -= 1

    def should_evict(self) -> bool:
        return self.current_size > self.max_size


class KVCacheEvictionManager:
    """
    Unified manager for KV cache eviction across multiple layers.

    Coordinates eviction across all transformer layers while maintaining
    consistency and handling protected tokens (e.g., system prompts).

    Example usage:
        >>> manager = KVCacheEvictionManager(
        ...     num_layers=32,
        ...     max_cache_size=4096,
        ...     policy=EvictionPolicy.ATTENTION
        ... )
        >>> manager.update_attention(layer_idx=0, attention_weights=attn)
        >>> if manager.should_evict():
        ...     indices_to_evict = manager.get_eviction_indices()
        ...     kv_cache = apply_eviction(kv_cache, indices_to_evict)
    """

    def __init__(
        self,
        num_layers: int,
        max_cache_size: int = 4096,
        policy: EvictionPolicy = EvictionPolicy.ATTENTION,
        config: Optional[EvictionConfig] = None,
    ):
        self.num_layers = num_layers
        self.config = config or EvictionConfig(
            max_cache_size=max_cache_size,
            eviction_policy=policy,
        )

        # Create eviction handler based on policy
        self.eviction_handlers = []
        for _ in range(num_layers):
            if policy == EvictionPolicy.LRU:
                handler = LRUEvictionManager(
                    max_size=self.config.max_cache_size,
                    min_protected=self.config.min_protected_tokens,
                )
            elif policy == EvictionPolicy.ATTENTION:
                handler = AttentionBasedEviction(
                    max_size=self.config.max_cache_size,
                    min_protected=self.config.min_protected_tokens,
                    decay=self.config.attention_decay,
                )
            elif policy == EvictionPolicy.HEAVY_HITTER:
                handler = HeavyHitterEviction(
                    max_size=self.config.max_cache_size,
                    min_protected=self.config.min_protected_tokens,
                )
            else:
                handler = LRUEvictionManager(
                    max_size=self.config.max_cache_size,
                    min_protected=self.config.min_protected_tokens,
                )
            self.eviction_handlers.append(handler)

    def update_attention(
        self,
        layer_idx: int,
        attention_weights: torch.Tensor,
    ):
        """Update eviction tracking with attention weights."""
        handler = self.eviction_handlers[layer_idx]

        if isinstance(handler, AttentionBasedEviction):
            handler.update_scores(attention_weights)
        elif isinstance(handler, HeavyHitterEviction):
            handler.update(attention_weights)
        elif isinstance(handler, LRUEvictionManager):
            # For LRU, mark all attended tokens as accessed
            key_len = attention_weights.shape[-1]
            handler.access(list(range(key_len)))

    def should_evict(self, layer_idx: Optional[int] = None) -> bool:
        """Check if any layer needs eviction."""
        if layer_idx is not None:
            return self.eviction_handlers[layer_idx].should_evict()
        return any(h.should_evict() for h in self.eviction_handlers)

    def get_eviction_indices(
        self,
        layer_idx: int,
        num_to_evict: Optional[int] = None,
    ) -> List[int]:
        """Get indices to evict for a specific layer."""
        if num_to_evict is None:
            num_to_evict = self.config.eviction_batch_size

        handler = self.eviction_handlers[layer_idx]
        return handler.get_eviction_candidates(num_to_evict)

    def perform_eviction(
        self,
        layer_idx: int,
        indices: List[int],
    ):
        """Record that eviction was performed."""
        self.eviction_handlers[layer_idx].evict(indices)

    def get_statistics(self) -> Dict[str, Any]:
        """Get eviction statistics."""
        stats = {
            "policy": self.config.eviction_policy.value,
            "max_cache_size": self.config.max_cache_size,
            "layers": [],
        }

        for i, handler in enumerate(self.eviction_handlers):
            layer_stat = {
                "layer": i,
                "current_size": handler.current_size,
                "needs_eviction": handler.should_evict(),
            }
            stats["layers"].append(layer_stat)

        return stats


class SlidingWindowCache:
    """
    Simple sliding window KV cache.

    Keeps only the most recent N tokens, discarding older ones.
    Very memory efficient but loses long-range context.

    Good for:
    - Tasks with local context (e.g., code completion)
    - Memory-constrained environments
    - When combined with other techniques (e.g., prefix caching)
    """

    def __init__(
        self,
        window_size: int = 2048,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.window_size = window_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        # Pre-allocate cache buffers
        self.key_cache = torch.zeros(
            num_layers, 1, num_heads, window_size, head_dim,
            device=device, dtype=torch.float16
        )
        self.value_cache = torch.zeros(
            num_layers, 1, num_heads, window_size, head_dim,
            device=device, dtype=torch.float16
        )

        self.current_length = 0

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new states, sliding if necessary.

        Args:
            layer_idx: Layer index
            key_states: [batch, heads, seq_len, head_dim]
            value_states: [batch, heads, seq_len, head_dim]

        Returns:
            Tuple of (cached_keys, cached_values) up to window_size
        """
        new_len = key_states.shape[2]
        total_len = self.current_length + new_len

        if total_len <= self.window_size:
            # Cache not full, just append
            self.key_cache[layer_idx, :, :, self.current_length:total_len] = key_states
            self.value_cache[layer_idx, :, :, self.current_length:total_len] = value_states
        else:
            # Need to slide window
            if new_len >= self.window_size:
                # New tokens fill entire window
                self.key_cache[layer_idx] = key_states[:, :, -self.window_size:]
                self.value_cache[layer_idx] = value_states[:, :, -self.window_size:]
            else:
                # Shift and append
                shift = total_len - self.window_size
                self.key_cache[layer_idx, :, :, :-new_len] = self.key_cache[layer_idx, :, :, shift:self.current_length].clone()
                self.value_cache[layer_idx, :, :, :-new_len] = self.value_cache[layer_idx, :, :, shift:self.current_length].clone()
                self.key_cache[layer_idx, :, :, -new_len:] = key_states
                self.value_cache[layer_idx, :, :, -new_len:] = value_states

        # Update length (for first layer only to avoid counting multiple times)
        if layer_idx == 0:
            self.current_length = min(total_len, self.window_size)

        # Return valid portion of cache
        return (
            self.key_cache[layer_idx, :, :, :self.current_length],
            self.value_cache[layer_idx, :, :, :self.current_length],
        )

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage in MB."""
        total_bytes = self.key_cache.numel() * 2 + self.value_cache.numel() * 2  # FP16
        return {
            "total_mb": total_bytes / (1024 * 1024),
            "per_layer_mb": total_bytes / (1024 * 1024) / self.num_layers,
            "current_length": self.current_length,
            "window_size": self.window_size,
        }

    def clear(self):
        """Clear the cache."""
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.current_length = 0
