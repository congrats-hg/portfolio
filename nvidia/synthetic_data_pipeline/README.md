# Synthetic Data Generation Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of a synthetic data generation pipeline for LLM training, inspired by **NVIDIA Nemotron-4 340B** which used **98% synthetic data** for fine-tuning.

## Overview

High-quality training data is the key to powerful LLMs. This pipeline generates synthetic instruction-response pairs that can be used for:
- Supervised Fine-Tuning (SFT)
- Reinforcement Learning from Human Feedback (RLHF)
- Domain-specific model adaptation

### Key Features

- **Self-Instruct**: Generate diverse instructions from seed examples
- **Evol-Instruct**: Evolve simple instructions into complex ones
- **Quality Evaluation**: LLM-as-judge with 5-dimensional scoring
- **Multi-stage Filtering**: Quality, safety, and diversity filters
- **Diversity Sampling**: Ensure balanced coverage across topics

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Synthetic Data Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │    Seed      │ →  │  Instruction │ →  │   Response   │      │
│  │ Instructions │    │  Generation  │    │  Generation  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                              ↓                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Evol-      │ →  │   Quality    │ →  │   Reward     │      │
│  │   Instruct   │    │  Evaluation  │    │   Scoring    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                              ↓                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Safety     │ →  │  Diversity   │ →  │   Final      │      │
│  │   Filter     │    │   Filter     │    │   Dataset    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Generator (`data_generator.py`)

Generates synthetic instruction-response pairs:

```python
from src.data_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(model, tokenizer)

# Generate from seed instructions
samples = generator.generate(
    seed_instructions=["Write a function that...", "Explain how..."],
    num_samples=1000,
    evolve=True  # Apply Evol-Instruct
)

generator.save(samples, "data/synthetic.jsonl")
```

### 2. Reward Model (`reward_model.py`)

Evaluates quality using the Nemotron 5-dimensional framework:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Helpfulness | 30% | Does it address the instruction? |
| Correctness | 30% | Is the information accurate? |
| Coherence | 20% | Is it well-organized? |
| Complexity | 10% | Appropriate depth of thought? |
| Verbosity | 10% | Right level of detail? |

```python
from src.reward_model import RewardModel

reward_model = RewardModel(model, tokenizer, mode="llm_judge")

scores = reward_model.evaluate(
    instruction="Explain machine learning",
    response="Machine learning is..."
)

print(f"Overall score: {scores.overall:.2f}")
print(f"Helpfulness: {scores.helpfulness:.2f}")
```

### 3. Data Filter (`data_filter.py`)

Multi-stage filtering for quality assurance:

```python
from src.data_filter import CombinedFilter

filter = CombinedFilter()
passed, rejected = filter.filter(samples)

# Get filtering summary
print(f"Quality rejections: {len(rejected['quality'])}")
print(f"Safety rejections: {len(rejected['safety'])}")
print(f"Diversity rejections: {len(rejected['diversity'])}")
```

### 4. Diversity Sampler (`diversity_sampler.py`)

Ensures diverse coverage across topics:

```python
from src.diversity_sampler import DiversitySampler

sampler = DiversitySampler()

# Cluster-balanced sampling
diverse_samples = sampler.sample(
    all_samples,
    num_samples=1000,
    strategy="cluster_balanced"
)

# Check diversity metrics
metrics = sampler.compute_diversity_score(diverse_samples)
print(f"Topic entropy: {metrics['topic_entropy']:.3f}")
```

### 5. Complete Pipeline (`pipeline.py`)

End-to-end pipeline:

```python
from src.pipeline import SyntheticDataPipeline, PipelineConfig

config = PipelineConfig(
    num_samples=1000,
    evolve_instructions=True,
    min_quality_score=0.5,
    enable_safety_filter=True,
    output_dir="data/generated"
)

pipeline = SyntheticDataPipeline(model, tokenizer, config)
results = pipeline.run(seed_file="prompts/generation_prompts.yaml")
pipeline.save(results)
```

## Project Structure

```
synthetic_data_pipeline/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # Instruction & response generation
│   ├── reward_model.py         # Quality evaluation
│   ├── data_filter.py          # Filtering (quality, safety, diversity)
│   ├── diversity_sampler.py    # Diversity-aware sampling
│   └── pipeline.py             # Complete pipeline
├── prompts/
│   ├── generation_prompts.yaml # Generation templates
│   └── evaluation_criteria.yaml # Evaluation rubrics
├── data/
│   ├── seed_data/              # Seed instructions
│   └── generated/              # Output data
└── notebooks/
    └── data_analysis.ipynb     # Analysis notebook
```

## NVIDIA Nemotron Methodology

This pipeline follows the methodology from NVIDIA's Nemotron-4 340B:

> "Throughout the alignment process, they relied on only approximately 20K human-annotated data while their data generation pipeline synthesized over 98% of the data used for supervised fine-tuning and preference fine-tuning."

### Key Principles

1. **Quality over Quantity**: Rigorous filtering ensures only high-quality data
2. **Diversity**: Balanced coverage across topics and difficulty levels
3. **Iterative Refinement**: Evol-Instruct creates increasingly complex examples
4. **Multi-dimensional Evaluation**: 5-axis scoring for comprehensive quality assessment

## References

1. **NVIDIA Nemotron-4 340B**: [Synthetic Data Generation Pipeline](https://blogs.nvidia.com/blog/nemotron-4-synthetic-data-generation-llm-training/)

2. **NVIDIA Nemotron 3 Nano**: [Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)

3. **Self-Instruct**: Wang et al., "Self-Instruct: Aligning Language Model with Self Generated Instructions"

4. **Evol-Instruct**: Xu et al., "WizardLM: Empowering Large Language Models to Follow Complex Instructions"

5. **NeMo Curator**: [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/24.12/datacuration/syntheticdata.html)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- sentence-transformers (for diversity)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.pipeline import run_simple_pipeline

# Load model
model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Simple generation
seeds = [
    "Explain machine learning",
    "Write a sorting algorithm",
    "Describe photosynthesis"
]

samples = run_simple_pipeline(
    model, tokenizer, seeds,
    num_samples=100
)

print(f"Generated {len(samples)} high-quality samples")
```

## License

MIT License
