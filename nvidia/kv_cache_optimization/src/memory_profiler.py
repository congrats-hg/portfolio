"""
KV Cache Memory Profiler
========================

Comprehensive memory profiling tools for analyzing KV cache behavior:
- Memory usage over sequence length
- Comparison of different optimization techniques
- Visualization of memory patterns

Reference:
- NVIDIA Nsight profiling methodology
- PyTorch memory profiling tools
"""

import torch
import time
import gc
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict, field
import json
import numpy as np


@dataclass
class MemorySnapshot:
    """Single memory measurement."""
    timestamp_ms: float
    sequence_length: int
    allocated_mb: float
    reserved_mb: float
    kv_cache_mb: float
    model_mb: float
    other_mb: float


@dataclass
class ProfilingResult:
    """Results from a profiling session."""
    model_name: str
    technique: str
    config: Dict[str, Any]
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    final_memory_mb: float = 0.0
    tokens_per_second: float = 0.0
    total_time_ms: float = 0.0


class KVCacheMemoryProfiler:
    """
    Profiles memory usage of KV cache under different configurations.

    Tracks:
    - GPU memory allocation over time
    - Memory breakdown (model, KV cache, activations)
    - Impact of different optimization techniques

    Example usage:
        >>> profiler = KVCacheMemoryProfiler(device="cuda")
        >>> result = profiler.profile_sequence_growth(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     max_length=4096,
        ...     step_size=128
        ... )
        >>> profiler.visualize(result)
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self._baseline_memory = 0.0

    def _clear_memory(self):
        """Clear GPU memory for accurate measurement."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _get_memory_mb(self) -> Dict[str, float]:
        """Get current GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0}

        return {
            "allocated": torch.cuda.memory_allocated() / (1024 * 1024),
            "reserved": torch.cuda.memory_reserved() / (1024 * 1024),
        }

    def _record_snapshot(
        self,
        seq_length: int,
        kv_cache_mb: float,
        model_mb: float,
        start_time: float,
    ) -> MemorySnapshot:
        """Record a memory snapshot."""
        mem = self._get_memory_mb()
        return MemorySnapshot(
            timestamp_ms=(time.time() - start_time) * 1000,
            sequence_length=seq_length,
            allocated_mb=mem["allocated"],
            reserved_mb=mem["reserved"],
            kv_cache_mb=kv_cache_mb,
            model_mb=model_mb,
            other_mb=mem["allocated"] - kv_cache_mb - model_mb,
        )

    def profile_kv_cache_growth(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_length: int = 8192,
        step_size: int = 256,
        dtype: torch.dtype = torch.float16,
        techniques: Optional[List[str]] = None,
    ) -> Dict[str, ProfilingResult]:
        """
        Profile KV cache memory growth with different techniques.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
            max_length: Maximum sequence length to test
            step_size: Increment between measurements
            dtype: Data type for storage
            techniques: List of techniques to test

        Returns:
            Dictionary mapping technique name to ProfilingResult
        """
        if techniques is None:
            techniques = ["fp16_baseline", "int8_quantized", "int4_quantized", "sliding_window"]

        results = {}

        for technique in techniques:
            print(f"Profiling: {technique}")
            self._clear_memory()

            result = ProfilingResult(
                model_name=f"layers={num_layers}, heads={num_heads}, dim={head_dim}",
                technique=technique,
                config={
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "max_length": max_length,
                    "dtype": str(dtype),
                },
            )

            start_time = time.time()

            for seq_len in range(step_size, max_length + 1, step_size):
                # Calculate theoretical KV cache size
                if technique == "fp16_baseline":
                    bytes_per_element = 2
                    effective_seq_len = seq_len
                elif technique == "int8_quantized":
                    bytes_per_element = 1
                    effective_seq_len = seq_len
                elif technique == "int4_quantized":
                    bytes_per_element = 0.5
                    effective_seq_len = seq_len
                elif technique == "sliding_window":
                    bytes_per_element = 2
                    effective_seq_len = min(seq_len, 2048)  # Window of 2048
                else:
                    bytes_per_element = 2
                    effective_seq_len = seq_len

                # KV cache size: 2 (K+V) * layers * batch * heads * seq * dim * bytes
                kv_cache_bytes = (
                    2 * num_layers * 1 * num_heads * effective_seq_len * head_dim * bytes_per_element
                )
                kv_cache_mb = kv_cache_bytes / (1024 * 1024)

                # Simulate allocation to get real GPU memory numbers
                if torch.cuda.is_available():
                    # Allocate representative tensors
                    try:
                        if technique in ["fp16_baseline", "sliding_window"]:
                            k = torch.zeros(
                                num_layers, 1, num_heads, effective_seq_len, head_dim,
                                device=self.device, dtype=dtype
                            )
                            v = torch.zeros_like(k)
                        else:
                            # Quantized - use int8 tensor
                            k = torch.zeros(
                                num_layers, 1, num_heads, effective_seq_len, head_dim,
                                device=self.device, dtype=torch.int8
                            )
                            v = torch.zeros_like(k)

                        snapshot = self._record_snapshot(
                            seq_len, kv_cache_mb, 0, start_time
                        )
                        result.snapshots.append(snapshot)

                        del k, v
                        self._clear_memory()

                    except torch.cuda.OutOfMemoryError:
                        print(f"  OOM at seq_len={seq_len}")
                        break
                else:
                    # CPU simulation
                    snapshot = MemorySnapshot(
                        timestamp_ms=(time.time() - start_time) * 1000,
                        sequence_length=seq_len,
                        allocated_mb=kv_cache_mb,
                        reserved_mb=kv_cache_mb,
                        kv_cache_mb=kv_cache_mb,
                        model_mb=0,
                        other_mb=0,
                    )
                    result.snapshots.append(snapshot)

            if result.snapshots:
                result.peak_memory_mb = max(s.kv_cache_mb for s in result.snapshots)
                result.final_memory_mb = result.snapshots[-1].kv_cache_mb
                result.total_time_ms = (time.time() - start_time) * 1000

            results[technique] = result

        return results

    def compare_techniques(
        self,
        results: Dict[str, ProfilingResult],
    ) -> Dict[str, Any]:
        """
        Compare different techniques based on profiling results.

        Returns summary statistics and recommendations.
        """
        comparison = {
            "techniques": {},
            "recommendations": [],
        }

        baseline_peak = 0
        if "fp16_baseline" in results:
            baseline_peak = results["fp16_baseline"].peak_memory_mb

        for technique, result in results.items():
            peak = result.peak_memory_mb
            savings = (baseline_peak - peak) / baseline_peak * 100 if baseline_peak > 0 else 0

            comparison["techniques"][technique] = {
                "peak_memory_mb": peak,
                "savings_vs_baseline_pct": savings,
                "final_memory_mb": result.final_memory_mb,
                "num_measurements": len(result.snapshots),
            }

        # Generate recommendations
        if baseline_peak > 0:
            best_technique = min(
                results.keys(),
                key=lambda t: results[t].peak_memory_mb
            )
            comparison["recommendations"].append(
                f"Most memory efficient: {best_technique} "
                f"({comparison['techniques'][best_technique]['savings_vs_baseline_pct']:.1f}% savings)"
            )

        return comparison

    def generate_report(
        self,
        results: Dict[str, ProfilingResult],
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a text report of profiling results."""
        lines = [
            "=" * 60,
            "KV Cache Memory Profiling Report",
            "=" * 60,
            "",
        ]

        comparison = self.compare_techniques(results)

        for technique, result in results.items():
            lines.extend([
                f"\n{technique.upper()}",
                "-" * 40,
                f"  Peak Memory: {result.peak_memory_mb:.2f} MB",
                f"  Final Memory: {result.final_memory_mb:.2f} MB",
                f"  Total Time: {result.total_time_ms:.2f} ms",
                f"  Measurements: {len(result.snapshots)}",
            ])

            if technique in comparison["techniques"]:
                savings = comparison["techniques"][technique]["savings_vs_baseline_pct"]
                lines.append(f"  Savings vs Baseline: {savings:.1f}%")

        lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 40,
        ])
        for rec in comparison["recommendations"]:
            lines.append(f"  * {rec}")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report

    def export_to_json(
        self,
        results: Dict[str, ProfilingResult],
        output_path: str,
    ):
        """Export results to JSON for further analysis."""
        export_data = {}

        for technique, result in results.items():
            export_data[technique] = {
                "model_name": result.model_name,
                "technique": result.technique,
                "config": result.config,
                "peak_memory_mb": result.peak_memory_mb,
                "final_memory_mb": result.final_memory_mb,
                "total_time_ms": result.total_time_ms,
                "snapshots": [asdict(s) for s in result.snapshots],
            }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

    def plot_memory_growth(
        self,
        results: Dict[str, ProfilingResult],
        output_path: Optional[str] = None,
    ):
        """
        Plot memory growth curves for different techniques.

        Requires matplotlib (optional dependency).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting. Install with: pip install matplotlib")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, (technique, result) in enumerate(results.items()):
            if not result.snapshots:
                continue

            seq_lengths = [s.sequence_length for s in result.snapshots]
            memory_mb = [s.kv_cache_mb for s in result.snapshots]

            ax.plot(
                seq_lengths,
                memory_mb,
                label=technique,
                color=colors[i % len(colors)],
                linewidth=2,
            )

        ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
        ax.set_ylabel("KV Cache Memory (MB)", fontsize=12)
        ax.set_title("KV Cache Memory Growth by Technique", fontsize=14)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        else:
            plt.show()


def estimate_kv_cache_size(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    sequence_length: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,  # 2 for FP16, 1 for INT8, 0.5 for INT4
) -> Dict[str, float]:
    """
    Estimate KV cache memory requirements.

    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per head
        sequence_length: Maximum sequence length
        batch_size: Batch size
        dtype_bytes: Bytes per element

    Returns:
        Dictionary with memory estimates
    """
    # KV cache: 2 * layers * batch * heads * seq * dim * bytes
    total_bytes = 2 * num_layers * batch_size * num_heads * sequence_length * head_dim * dtype_bytes

    return {
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "total_gb": total_bytes / (1024 * 1024 * 1024),
        "per_layer_mb": total_bytes / num_layers / (1024 * 1024),
        "per_token_kb": (2 * num_layers * num_heads * head_dim * dtype_bytes) / 1024,
    }


def print_memory_estimates():
    """Print memory estimates for common model configurations."""
    configs = [
        ("Llama-7B", 32, 32, 128),
        ("Llama-13B", 40, 40, 128),
        ("Llama-70B", 80, 64, 128),
        ("Mistral-7B", 32, 8, 128),  # GQA with 8 KV heads
    ]

    print("\nKV Cache Memory Estimates (batch_size=1)")
    print("=" * 80)
    print(f"{'Model':<15} {'Seq Len':<10} {'FP16 (GB)':<12} {'INT8 (GB)':<12} {'INT4 (GB)':<12}")
    print("-" * 80)

    for model_name, layers, heads, dim in configs:
        for seq_len in [2048, 4096, 8192, 32768]:
            fp16 = estimate_kv_cache_size(layers, heads, dim, seq_len, dtype_bytes=2)
            int8 = estimate_kv_cache_size(layers, heads, dim, seq_len, dtype_bytes=1)
            int4 = estimate_kv_cache_size(layers, heads, dim, seq_len, dtype_bytes=0.5)

            print(f"{model_name:<15} {seq_len:<10} {fp16['total_gb']:<12.2f} {int8['total_gb']:<12.2f} {int4['total_gb']:<12.2f}")

        print()


if __name__ == "__main__":
    print_memory_estimates()
