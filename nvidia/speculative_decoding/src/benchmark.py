"""
Benchmarking Suite for Speculative Decoding
============================================

Comprehensive benchmarking tools to measure and compare:
- Throughput (tokens/second)
- Latency (time to first token, time per token)
- Acceptance rate
- Memory usage
- Speedup vs autoregressive baseline

Reference metrics from NVIDIA:
- TensorRT-LLM achieves up to 3.55x speedup with speculative decoding
- Acceptance rate typically 60-80% depending on task
"""

import torch
import time
import json
import gc
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method: str
    model_name: str
    prompt: str
    generated_text: str
    num_tokens: int
    total_time_ms: float
    tokens_per_second: float
    time_to_first_token_ms: float
    acceptance_rate: Optional[float] = None
    speedup_vs_baseline: Optional[float] = None
    memory_used_mb: Optional[float] = None
    extra_metrics: Optional[Dict[str, Any]] = None


@dataclass
class AggregatedResults:
    """Aggregated results from multiple benchmark runs."""
    method: str
    model_name: str
    num_runs: int
    avg_tokens_per_second: float
    std_tokens_per_second: float
    avg_acceptance_rate: Optional[float]
    avg_speedup: Optional[float]
    avg_memory_mb: Optional[float]
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float


class SpeculativeDecodingBenchmark:
    """
    Comprehensive benchmark suite for speculative decoding methods.

    Compares:
    1. Standard autoregressive decoding (baseline)
    2. Draft-target speculative decoding
    3. Medusa-style multi-head decoding

    Usage:
        >>> benchmark = SpeculativeDecodingBenchmark(
        ...     draft_model_name="gpt2",
        ...     target_model_name="gpt2-medium"
        ... )
        >>> results = benchmark.run_full_benchmark(prompts, num_runs=3)
        >>> benchmark.save_results(results, "benchmark_results.json")
    """

    # Standard benchmark prompts covering different domains
    DEFAULT_PROMPTS = [
        "Explain the concept of machine learning in simple terms:",
        "Write a Python function that implements binary search:",
        "The history of artificial intelligence began",
        "In a large language model, the attention mechanism",
        "The key differences between Python and JavaScript are",
        "To optimize neural network training, one should",
        "Climate change is affecting our planet in several ways:",
        "The process of photosynthesis involves",
    ]

    def __init__(
        self,
        draft_model_name: str = "gpt2",
        target_model_name: str = "gpt2-medium",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype
        self.draft_model_name = draft_model_name
        self.target_model_name = target_model_name

        print(f"Loading models on {device}...")
        self._load_models()

    def _load_models(self):
        """Load draft and target models."""
        # Load tokenizer (shared)
        self.tokenizer = AutoTokenizer.from_pretrained(self.target_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load draft model
        print(f"Loading draft model: {self.draft_model_name}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.draft_model_name,
            torch_dtype=self.dtype,
            device_map=self.device if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.draft_model = self.draft_model.to(self.device)
        self.draft_model.eval()

        # Load target model
        print(f"Loading target model: {self.target_model_name}")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_model_name,
            torch_dtype=self.dtype,
            device_map=self.device if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.target_model = self.target_model.to(self.device)
        self.target_model.eval()

    def _clear_memory(self):
        """Clear GPU memory between runs."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    @torch.no_grad()
    def benchmark_autoregressive(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> BenchmarkResult:
        """
        Benchmark standard autoregressive decoding (baseline).

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            BenchmarkResult with timing and throughput metrics
        """
        self._clear_memory()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Warm-up run
        _ = self.target_model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        self._clear_memory()
        memory_before = self._get_memory_usage()

        # Timed generation
        start_time = time.perf_counter()

        output_ids = self.target_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        memory_after = self._get_memory_usage()

        # Calculate metrics
        elapsed_ms = (end_time - start_time) * 1000
        num_new_tokens = output_ids.shape[1] - input_ids.shape[1]
        tokens_per_sec = (num_new_tokens * 1000) / elapsed_ms if elapsed_ms > 0 else 0

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return BenchmarkResult(
            method="autoregressive",
            model_name=self.target_model_name,
            prompt=prompt,
            generated_text=generated_text,
            num_tokens=num_new_tokens,
            total_time_ms=elapsed_ms,
            tokens_per_second=tokens_per_sec,
            time_to_first_token_ms=elapsed_ms / num_new_tokens if num_new_tokens > 0 else 0,
            memory_used_mb=memory_after - memory_before,
        )

    @torch.no_grad()
    def benchmark_speculative(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        num_speculative_tokens: int = 5,
        temperature: float = 1.0,
    ) -> BenchmarkResult:
        """
        Benchmark draft-target speculative decoding.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            num_speculative_tokens: Number of draft tokens per iteration
            temperature: Sampling temperature

        Returns:
            BenchmarkResult with timing, throughput, and acceptance metrics
        """
        from .draft_target import DraftTargetDecoder, SpeculativeDecodingConfig

        self._clear_memory()

        config = SpeculativeDecodingConfig(
            num_speculative_tokens=num_speculative_tokens,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        decoder = DraftTargetDecoder(
            draft_model=self.draft_model,
            target_model=self.target_model,
            tokenizer=self.tokenizer,
            config=config,
            device=self.device,
        )

        # Warm-up run
        _, _ = decoder.generate(prompt, max_new_tokens=5)

        self._clear_memory()
        memory_before = self._get_memory_usage()

        # Timed generation
        start_time = time.perf_counter()
        generated_text, metrics = decoder.generate(prompt, max_new_tokens=max_new_tokens)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        memory_after = self._get_memory_usage()

        elapsed_ms = (end_time - start_time) * 1000

        return BenchmarkResult(
            method="speculative_draft_target",
            model_name=f"{self.draft_model_name} + {self.target_model_name}",
            prompt=prompt,
            generated_text=generated_text,
            num_tokens=metrics.accepted_tokens,
            total_time_ms=elapsed_ms,
            tokens_per_second=metrics.tokens_per_second,
            time_to_first_token_ms=elapsed_ms / metrics.accepted_tokens if metrics.accepted_tokens > 0 else 0,
            acceptance_rate=metrics.acceptance_rate,
            memory_used_mb=memory_after - memory_before,
            extra_metrics={
                "draft_time_ms": metrics.draft_time_ms,
                "verify_time_ms": metrics.verify_time_ms,
                "rejected_tokens": metrics.rejected_tokens,
            },
        )

    def run_comparison(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        num_speculative_tokens: int = 5,
        temperature: float = 1.0,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run comparison between autoregressive and speculative decoding.

        Returns:
            Dictionary with results for each method
        """
        results = {}

        # Baseline: autoregressive
        print("  Running autoregressive baseline...")
        results["autoregressive"] = self.benchmark_autoregressive(
            prompt, max_new_tokens, temperature
        )

        # Speculative decoding
        print("  Running speculative decoding...")
        results["speculative"] = self.benchmark_speculative(
            prompt, max_new_tokens, num_speculative_tokens, temperature
        )

        # Calculate speedup
        baseline_tps = results["autoregressive"].tokens_per_second
        speculative_tps = results["speculative"].tokens_per_second

        if baseline_tps > 0:
            results["speculative"].speedup_vs_baseline = speculative_tps / baseline_tps

        return results

    def run_full_benchmark(
        self,
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 50,
        num_runs: int = 3,
        num_speculative_tokens: int = 5,
    ) -> Dict[str, AggregatedResults]:
        """
        Run full benchmark suite on multiple prompts.

        Args:
            prompts: List of prompts (uses default if None)
            max_new_tokens: Tokens to generate per prompt
            num_runs: Number of runs per prompt for averaging
            num_speculative_tokens: Draft tokens per iteration

        Returns:
            Dictionary of aggregated results per method
        """
        prompts = prompts or self.DEFAULT_PROMPTS

        all_results = {
            "autoregressive": [],
            "speculative": [],
        }

        print(f"\nRunning benchmark with {len(prompts)} prompts, {num_runs} runs each...")

        for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
            for run_idx in range(num_runs):
                comparison = self.run_comparison(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    num_speculative_tokens=num_speculative_tokens,
                )

                for method, result in comparison.items():
                    all_results[method].append(result)

        # Aggregate results
        aggregated = {}
        for method, results in all_results.items():
            tps_values = [r.tokens_per_second for r in results]
            latencies = [r.total_time_ms for r in results]
            acceptance_rates = [r.acceptance_rate for r in results if r.acceptance_rate is not None]
            speedups = [r.speedup_vs_baseline for r in results if r.speedup_vs_baseline is not None]
            memory_values = [r.memory_used_mb for r in results if r.memory_used_mb is not None]

            aggregated[method] = AggregatedResults(
                method=method,
                model_name=results[0].model_name if results else "",
                num_runs=len(results),
                avg_tokens_per_second=np.mean(tps_values),
                std_tokens_per_second=np.std(tps_values),
                avg_acceptance_rate=np.mean(acceptance_rates) if acceptance_rates else None,
                avg_speedup=np.mean(speedups) if speedups else None,
                avg_memory_mb=np.mean(memory_values) if memory_values else None,
                p50_latency_ms=np.percentile(latencies, 50),
                p90_latency_ms=np.percentile(latencies, 90),
                p99_latency_ms=np.percentile(latencies, 99),
            )

        return aggregated

    def print_results(self, results: Dict[str, AggregatedResults]):
        """Pretty print benchmark results."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)

        for method, agg in results.items():
            print(f"\n{method.upper()}")
            print("-" * 40)
            print(f"  Model: {agg.model_name}")
            print(f"  Runs: {agg.num_runs}")
            print(f"  Throughput: {agg.avg_tokens_per_second:.2f} ± {agg.std_tokens_per_second:.2f} tokens/sec")
            print(f"  Latency P50: {agg.p50_latency_ms:.2f} ms")
            print(f"  Latency P90: {agg.p90_latency_ms:.2f} ms")
            print(f"  Latency P99: {agg.p99_latency_ms:.2f} ms")

            if agg.avg_acceptance_rate is not None:
                print(f"  Acceptance Rate: {agg.avg_acceptance_rate * 100:.1f}%")
            if agg.avg_speedup is not None:
                print(f"  Speedup vs Baseline: {agg.avg_speedup:.2f}x")
            if agg.avg_memory_mb is not None:
                print(f"  Memory Usage: {agg.avg_memory_mb:.2f} MB")

        print("\n" + "=" * 70)

    def save_results(self, results: Dict[str, AggregatedResults], filepath: str):
        """Save results to JSON file."""
        serializable = {
            method: asdict(agg) for method, agg in results.items()
        }

        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"Results saved to {filepath}")

    @staticmethod
    def load_results(filepath: str) -> Dict[str, Dict]:
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)


def quick_benchmark(
    prompt: str = "Explain machine learning:",
    max_tokens: int = 30,
    draft_model: str = "gpt2",
    target_model: str = "gpt2-medium",
) -> None:
    """
    Quick benchmark for demonstration.

    Usage:
        >>> quick_benchmark()
    """
    print(f"Quick Benchmark: {draft_model} (draft) + {target_model} (target)")
    print(f"Prompt: {prompt[:50]}...")
    print(f"Max tokens: {max_tokens}")
    print()

    benchmark = SpeculativeDecodingBenchmark(
        draft_model_name=draft_model,
        target_model_name=target_model,
    )

    results = benchmark.run_comparison(prompt, max_new_tokens=max_tokens)
    benchmark.print_results({
        "autoregressive": AggregatedResults(
            method="autoregressive",
            model_name=target_model,
            num_runs=1,
            avg_tokens_per_second=results["autoregressive"].tokens_per_second,
            std_tokens_per_second=0,
            avg_acceptance_rate=None,
            avg_speedup=None,
            avg_memory_mb=results["autoregressive"].memory_used_mb,
            p50_latency_ms=results["autoregressive"].total_time_ms,
            p90_latency_ms=results["autoregressive"].total_time_ms,
            p99_latency_ms=results["autoregressive"].total_time_ms,
        ),
        "speculative": AggregatedResults(
            method="speculative",
            model_name=f"{draft_model} + {target_model}",
            num_runs=1,
            avg_tokens_per_second=results["speculative"].tokens_per_second,
            std_tokens_per_second=0,
            avg_acceptance_rate=results["speculative"].acceptance_rate,
            avg_speedup=results["speculative"].speedup_vs_baseline,
            avg_memory_mb=results["speculative"].memory_used_mb,
            p50_latency_ms=results["speculative"].total_time_ms,
            p90_latency_ms=results["speculative"].total_time_ms,
            p99_latency_ms=results["speculative"].total_time_ms,
        ),
    })


if __name__ == "__main__":
    quick_benchmark()
