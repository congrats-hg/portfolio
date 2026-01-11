"""
Synthetic Data Generation Pipeline
===================================

End-to-end pipeline combining all components for synthetic data generation.

Pipeline stages:
1. Instruction Generation (Self-Instruct / Evol-Instruct)
2. Response Generation
3. Quality Evaluation (Reward Model)
4. Filtering (Quality, Safety, Diversity)
5. Diversity Sampling

Reference:
- NVIDIA Nemotron-4 340B: 98% synthetic data
- NeMo Curator pipeline
"""

import torch
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from tqdm import tqdm

from .data_generator import SyntheticDataGenerator, GenerationConfig, SyntheticSample
from .reward_model import RewardModel, QualityEvaluator
from .data_filter import DataFilter, DiversityFilter, SafetyFilter, CombinedFilter, FilterConfig
from .diversity_sampler import DiversitySampler, DiversitySamplerConfig


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    # Generation settings
    num_samples: int = 1000
    evolve_instructions: bool = True
    evolution_depth: int = 2

    # Quality settings
    min_quality_score: float = 0.5
    use_reward_model: bool = True

    # Filtering settings
    enable_safety_filter: bool = True
    enable_diversity_filter: bool = True
    similarity_threshold: float = 0.85

    # Diversity sampling
    final_sample_size: Optional[int] = None  # If set, sample this many at the end
    num_clusters: int = 10

    # Output settings
    output_dir: str = "data/generated"
    save_intermediate: bool = True

    # Metadata
    run_name: str = "synthetic_run"


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""
    started_at: str = ""
    completed_at: str = ""
    total_time_seconds: float = 0.0

    instructions_generated: int = 0
    responses_generated: int = 0

    rejected_by_quality: int = 0
    rejected_by_safety: int = 0
    rejected_by_diversity: int = 0

    final_sample_count: int = 0
    diversity_score: Dict[str, Any] = field(default_factory=dict)


class SyntheticDataPipeline:
    """
    Complete synthetic data generation pipeline.

    Orchestrates all stages of synthetic data generation following
    NVIDIA Nemotron methodology.

    Example usage:
        >>> pipeline = SyntheticDataPipeline(model, tokenizer, config)
        >>> results = pipeline.run(seed_file="prompts/seeds.json")
        >>> pipeline.save(results)
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[PipelineConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or PipelineConfig()
        self.device = device

        # Initialize components
        self._init_components()

        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _init_components(self):
        """Initialize pipeline components."""
        # Generator
        gen_config = GenerationConfig(
            evolution_depth=self.config.evolution_depth
        )
        self.generator = SyntheticDataGenerator(
            self.model,
            self.tokenizer,
            config=gen_config,
            device=self.device
        )

        # Reward model / evaluator
        if self.config.use_reward_model:
            self.reward_model = RewardModel(
                self.model,
                self.tokenizer,
                mode="llm_judge",
                device=self.device
            )
            self.evaluator = QualityEvaluator(
                reward_model=self.reward_model
            )
        else:
            self.reward_model = None
            self.evaluator = QualityEvaluator()

        # Filters
        filter_config = FilterConfig(
            min_quality_score=self.config.min_quality_score,
            similarity_threshold=self.config.similarity_threshold
        )
        quality_filter = DataFilter(filter_config)
        safety_filter = SafetyFilter() if self.config.enable_safety_filter else None
        diversity_filter = DiversityFilter(
            similarity_threshold=self.config.similarity_threshold
        ) if self.config.enable_diversity_filter else None

        self.combined_filter = CombinedFilter(
            quality_filter=quality_filter,
            diversity_filter=diversity_filter or DiversityFilter(),
            safety_filter=safety_filter or SafetyFilter()
        )

        # Diversity sampler
        sampler_config = DiversitySamplerConfig(
            num_clusters=self.config.num_clusters
        )
        self.diversity_sampler = DiversitySampler(sampler_config)

    def run(
        self,
        seed_instructions: Optional[List[str]] = None,
        seed_file: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Args:
            seed_instructions: List of seed instructions
            seed_file: Path to seed file (alternative to seed_instructions)
            progress_callback: Optional callback(stage, current, total)

        Returns:
            Dictionary with results and statistics
        """
        stats = PipelineStats()
        stats.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        start_time = time.time()

        # Load seeds if file provided
        if seed_file and not seed_instructions:
            seed_instructions = self._load_seeds(seed_file)

        if not seed_instructions:
            raise ValueError("Either seed_instructions or seed_file required")

        print(f"Starting pipeline with {len(seed_instructions)} seed instructions")
        print(f"Target samples: {self.config.num_samples}")

        # Stage 1: Generate samples
        print("\n" + "=" * 50)
        print("Stage 1: Generating synthetic samples")
        print("=" * 50)

        samples = self.generator.generate(
            seed_instructions=seed_instructions,
            num_samples=self.config.num_samples,
            evolve=self.config.evolve_instructions,
        )

        stats.instructions_generated = len(samples)
        stats.responses_generated = len(samples)

        # Convert to dict format
        sample_dicts = [s.to_dict() for s in samples]

        if self.config.save_intermediate:
            self._save_intermediate(sample_dicts, "1_generated.jsonl")

        # Stage 2: Quality evaluation
        print("\n" + "=" * 50)
        print("Stage 2: Evaluating quality")
        print("=" * 50)

        for sample in tqdm(sample_dicts, desc="Evaluating"):
            evaluation = self.evaluator.evaluate(
                sample["instruction"],
                sample["response"]
            )
            sample["quality_scores"] = evaluation

        if self.config.save_intermediate:
            self._save_intermediate(sample_dicts, "2_evaluated.jsonl")

        # Stage 3: Filtering
        print("\n" + "=" * 50)
        print("Stage 3: Filtering samples")
        print("=" * 50)

        filtered, rejected = self.combined_filter.filter(sample_dicts)

        stats.rejected_by_quality = len(rejected.get("quality", []))
        stats.rejected_by_safety = len(rejected.get("safety", []))
        stats.rejected_by_diversity = len(rejected.get("diversity", []))

        print(f"Passed filtering: {len(filtered)}/{len(sample_dicts)}")
        print(f"  - Quality rejections: {stats.rejected_by_quality}")
        print(f"  - Safety rejections: {stats.rejected_by_safety}")
        print(f"  - Diversity rejections: {stats.rejected_by_diversity}")

        if self.config.save_intermediate:
            self._save_intermediate(filtered, "3_filtered.jsonl")

        # Stage 4: Diversity sampling (optional)
        print("\n" + "=" * 50)
        print("Stage 4: Diversity sampling")
        print("=" * 50)

        if self.config.final_sample_size and len(filtered) > self.config.final_sample_size:
            final_samples = self.diversity_sampler.sample(
                filtered,
                num_samples=self.config.final_sample_size,
                strategy="cluster_balanced"
            )
        else:
            final_samples = filtered

        stats.final_sample_count = len(final_samples)

        # Compute diversity metrics
        diversity_scores = self.diversity_sampler.compute_diversity_score(final_samples)
        stats.diversity_score = diversity_scores

        print(f"Final sample count: {len(final_samples)}")
        print(f"Topic entropy: {diversity_scores.get('topic_entropy', 0):.3f}")
        print(f"Vocabulary size: {diversity_scores.get('vocab_size', 0)}")

        # Finalize stats
        stats.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        stats.total_time_seconds = time.time() - start_time

        print("\n" + "=" * 50)
        print("Pipeline completed!")
        print("=" * 50)
        print(f"Total time: {stats.total_time_seconds:.1f} seconds")
        print(f"Final samples: {stats.final_sample_count}")

        return {
            "samples": final_samples,
            "stats": stats,
            "rejected": rejected,
        }

    def _load_seeds(self, seed_file: str) -> List[str]:
        """Load seed instructions from file."""
        path = Path(seed_file)

        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return data.get("instructions", data.get("seeds", []))
        elif path.suffix == ".txt":
            with open(path) as f:
                return [line.strip() for line in f if line.strip()]
        elif path.suffix == ".yaml" or path.suffix == ".yml":
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
                return data.get("instructions", data.get("seeds", []))
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _save_intermediate(self, samples: List[Dict], filename: str):
        """Save intermediate results."""
        path = self.output_dir / filename
        with open(path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

    def save(
        self,
        results: Dict[str, Any],
        format: str = "jsonl",
    ):
        """Save final results."""
        samples = results["samples"]
        stats = results["stats"]

        # Save samples
        samples_path = self.output_dir / f"{self.config.run_name}_samples.{format}"
        with open(samples_path, 'w') as f:
            if format == "jsonl":
                for sample in samples:
                    # Clean up for training format
                    clean = {
                        "instruction": sample["instruction"],
                        "response": sample["response"],
                    }
                    f.write(json.dumps(clean) + "\n")
            else:
                json.dump(samples, f, indent=2)

        # Save stats
        stats_path = self.output_dir / f"{self.config.run_name}_stats.json"
        with open(stats_path, 'w') as f:
            stats_dict = {
                k: v for k, v in stats.__dict__.items()
            }
            json.dump(stats_dict, f, indent=2)

        print(f"\nSaved {len(samples)} samples to {samples_path}")
        print(f"Saved stats to {stats_path}")


def run_simple_pipeline(
    model,
    tokenizer,
    seed_instructions: List[str],
    num_samples: int = 100,
    output_dir: str = "data/generated",
) -> List[Dict[str, str]]:
    """
    Simple function to run the pipeline with minimal configuration.

    Example:
        >>> seeds = ["Explain machine learning", "Write a poem", "Solve a math problem"]
        >>> samples = run_simple_pipeline(model, tokenizer, seeds, num_samples=50)
    """
    config = PipelineConfig(
        num_samples=num_samples,
        output_dir=output_dir,
        use_reward_model=True,
        enable_safety_filter=True,
        enable_diversity_filter=True,
    )

    pipeline = SyntheticDataPipeline(model, tokenizer, config)
    results = pipeline.run(seed_instructions=seed_instructions)
    pipeline.save(results)

    return results["samples"]
