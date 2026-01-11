"""
Synthetic Data Generation Pipeline
===================================

Implements NVIDIA Nemotron-style synthetic data generation for LLM training:
- Instruction generation from seed data
- Quality evaluation with reward model
- Diversity-based filtering
- Data curation pipeline

Reference:
- NVIDIA Nemotron-4 340B: 98% synthetic data for training
- NeMo Curator synthetic data pipelines
- "Nemotron-4 340B Technical Report"
"""

from .data_generator import SyntheticDataGenerator, InstructionGenerator
from .reward_model import RewardModel, QualityEvaluator
from .data_filter import DataFilter, DiversityFilter
from .diversity_sampler import DiversitySampler
from .pipeline import SyntheticDataPipeline

__all__ = [
    "SyntheticDataGenerator",
    "InstructionGenerator",
    "RewardModel",
    "QualityEvaluator",
    "DataFilter",
    "DiversityFilter",
    "DiversitySampler",
    "SyntheticDataPipeline",
]
