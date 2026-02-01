# NVIDIA LLM Research Portfolio

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**Large Language Model 추론 최적화 및 학습 데이터 생성 기법 연구**

NVIDIA의 최신 LLM 기술 스택(TensorRT-LLM, NeMo, Nemotron)을 기반으로 구현한 토이 프로젝트 모음입니다.

---

## 프로젝트 개요

| 프로젝트 | 설명 | 핵심 기술 |
|---------|------|----------|
| [Speculative Decoding](./speculative_decoding/) | LLM 추론 처리량 최대 3.5x 향상 | Draft-Target, Medusa, Tree Attention |
| [KV Cache Optimization](./kv_cache_optimization/) | 메모리 사용량 최대 75% 절감 | INT8/INT4 양자화, Token Eviction, GQA |
| [Synthetic Data Pipeline](./synthetic_data_pipeline/) | 98% 합성 데이터 학습 파이프라인 | Self-Instruct, Reward Model, Diversity Sampling |

---

## 1. Speculative Decoding

**NVIDIA TensorRT-LLM Speculative Decoding 구현**

LLM의 autoregressive 특성으로 인한 추론 병목을 해결하는 최적화 기법입니다.

### 핵심 아이디어

```
기존 방식: 1 forward pass → 1 token (느림)
Speculative: 1 forward pass → K tokens 검증 (빠름)

핵심 통찰: K개 토큰 검증 비용 ≈ 1개 토큰 생성 비용 (병렬화 덕분)
```

### 구현 기법

1. **Draft-Target Speculative Decoding**
   - 작은 draft 모델로 K개 토큰 후보 생성
   - 큰 target 모델로 한 번에 검증

2. **Medusa Multi-Head Decoding**
   - 단일 모델에 여러 예측 헤드 추가
   - 별도 draft 모델 없이 병렬 예측

3. **Tree Attention**
   - 여러 후보 시퀀스를 트리 구조로 구성
   - 공유 prefix 계산 재사용

### 성능 목표

| 메트릭 | 목표치 | NVIDIA 참고치 |
|--------|--------|--------------|
| Throughput 향상 | 1.8-2.5x | 3.55x (H200) |
| Acceptance Rate | 65-75% | 70-80% |

📁 **[상세 문서 보기](./speculative_decoding/README.md)**

---

## 2. KV Cache Optimization

**NVIDIA NVFP4 KV Cache 최적화 기법 구현**

Long context LLM에서 메모리 병목이 되는 KV Cache를 최적화합니다.

### 문제 상황

| 모델 | Context | KV Cache (FP16) | KV Cache (INT4) |
|------|---------|-----------------|-----------------|
| Llama-7B | 4K | 2 GB | 0.5 GB |
| Llama-70B | 32K | 160 GB | 40 GB |

### 구현 기법

1. **KV Cache 양자화**
   - INT8: 50% 메모리 절감
   - INT4 (NVFP4): 75% 메모리 절감

2. **Token Eviction**
   - LRU: 최근 사용 기반
   - Attention-based: 중요도 기반
   - Heavy Hitter (H2O): 누적 attention 기반

3. **GQA (Grouped Query Attention)**
   - Query head 수 > KV head 수
   - Mistral-7B 스타일 4x 메모리 절감

### 성능 목표

| 기법 | 메모리 절감 | Perplexity 영향 |
|------|------------|----------------|
| INT8 | 50% | < 0.1% |
| INT4 | 75% | < 0.5% |
| GQA | 75% | 모델 의존 |

📁 **[상세 문서 보기](./kv_cache_optimization/README.md)**

---

## 3. Synthetic Data Pipeline

**NVIDIA Nemotron-4 340B 스타일 합성 데이터 생성**

학습 데이터의 98%를 합성 데이터로 생성하는 파이프라인입니다.

### 파이프라인 구조

```
Seed Instructions → Instruction Generation → Evolution
        ↓
Response Generation → Quality Evaluation → Filtering
        ↓
Diversity Sampling → Final Dataset (98% synthetic)
```

### 구현 기법

1. **Self-Instruct**
   - 소수의 seed에서 다양한 instruction 생성

2. **Evol-Instruct**
   - 간단한 instruction을 복잡하게 진화

3. **Reward Model 평가** (Nemotron 5축)
   - Helpfulness (30%)
   - Correctness (30%)
   - Coherence (20%)
   - Complexity (10%)
   - Verbosity (10%)

4. **Diversity Sampling**
   - Embedding 클러스터링
   - Topic 균형 샘플링

📁 **[상세 문서 보기](./synthetic_data_pipeline/README.md)**

---

## 기술 스택

- **언어**: Python 3.8+, CUDA
- **프레임워크**: PyTorch 2.0+, Transformers, TensorRT-LLM
- **모델**: Llama, Mistral, GPT-2 (오픈소스)
- **도구**: Jupyter, Weights & Biases, NVIDIA Nsight

---

## 설치 및 실행

```bash
# 환경 설정
python -m venv venv
source venv/bin/activate

# 각 프로젝트별 의존성 설치
pip install -r speculative_decoding/requirements.txt
pip install -r kv_cache_optimization/requirements.txt
pip install -r synthetic_data_pipeline/requirements.txt
```

---

## Quick Start 가이드

### 1. Speculative Decoding 사용법

**Draft-Target 방식으로 추론 가속화:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from speculative_decoding.src.draft_target import (
    DraftTargetDecoder,
    SpeculativeDecodingConfig
)

# 모델 로드 (작은 draft 모델 + 큰 target 모델)
draft_model = AutoModelForCausalLM.from_pretrained("gpt2")
target_model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 설정: 한 번에 5개 토큰 추측, 최대 100개 생성
config = SpeculativeDecodingConfig(
    num_speculative_tokens=5,
    max_new_tokens=100
)

# 디코더 생성 및 실행
decoder = DraftTargetDecoder(draft_model, target_model, tokenizer, config)
output, metrics = decoder.generate("Once upon a time")

print(f"생성 결과: {output}")
print(f"속도 향상: {metrics.speedup_factor:.2f}x")
print(f"수락률: {metrics.acceptance_rate:.1%}")
```

**Medusa 멀티헤드 방식:**

```python
from speculative_decoding.src.medusa_heads import MedusaDecoder, MedusaConfig

# Medusa 설정 (4개 예측 헤드)
config = MedusaConfig(num_heads=4, num_candidates=64)
decoder = MedusaDecoder(base_model, tokenizer, config)

output, metrics = decoder.generate("Explain quantum computing")
print(f"Medusa 속도 향상: {metrics.speedup_factor:.2f}x")
```

**벤치마크 실행:**

```python
from speculative_decoding.src.benchmark import SpeculativeDecodingBenchmark

benchmark = SpeculativeDecodingBenchmark(
    draft_model, target_model, tokenizer
)
results = benchmark.run_comparison(
    prompts=["Once upon a time", "The future of AI"],
    methods=["autoregressive", "speculative", "medusa"]
)
benchmark.print_results(results)
```

---

### 2. KV Cache Optimization 사용법

**INT8/INT4 양자화로 메모리 절감:**

```python
from kv_cache_optimization.src.kv_cache_quantization import (
    QuantizedKVCache,
    QuantizationConfig
)

# INT8 양자화 KV Cache (50% 메모리 절감)
config = QuantizationConfig(bits=8, per_channel=True)
cache = QuantizedKVCache(
    num_layers=32,
    num_heads=32,
    head_dim=128,
    max_length=8192,
    config=config
)

# 메모리 사용량 확인
memory_info = cache.get_memory_usage()
print(f"압축률: {memory_info['compression_ratio']:.2f}x")
print(f"메모리 절감: {memory_info['memory_saved_mb']:.1f} MB")

# INT4 양자화 (75% 메모리 절감)
config_int4 = QuantizationConfig(bits=4, per_channel=True)
cache_int4 = QuantizedKVCache(
    num_layers=32, num_heads=32, head_dim=128,
    max_length=8192, config=config_int4
)
```

**Token Eviction으로 캐시 관리:**

```python
from kv_cache_optimization.src.kv_cache_eviction import (
    KVCacheEvictionManager,
    EvictionPolicy
)

# Heavy Hitter (H2O) 정책: 중요한 토큰 유지
eviction_manager = KVCacheEvictionManager(
    max_cache_size=2048,
    policy=EvictionPolicy.HEAVY_HITTER
)

# Attention 점수 업데이트 및 eviction
eviction_manager.update_attention_scores(attention_weights)
tokens_to_evict = eviction_manager.get_eviction_candidates(num_to_evict=100)
```

**메모리 프로파일링:**

```python
from kv_cache_optimization.src.memory_profiler import KVCacheProfiler

profiler = KVCacheProfiler(model)
report = profiler.profile_inference(
    input_ids,
    max_length=4096,
    techniques=["baseline", "int8", "int4", "eviction"]
)
profiler.plot_memory_comparison(report)
profiler.save_report(report, "memory_analysis.json")
```

---

### 3. Synthetic Data Pipeline 사용법

**전체 파이프라인 실행:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from synthetic_data_pipeline.src.pipeline import (
    SyntheticDataPipeline,
    PipelineConfig
)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 파이프라인 설정
config = PipelineConfig(
    num_samples=1000,           # 생성할 샘플 수
    min_quality_score=0.5,      # 최소 품질 점수
    enable_evolution=True,      # Evol-Instruct 사용
    diversity_clusters=50       # 다양성 클러스터 수
)

# 파이프라인 실행
pipeline = SyntheticDataPipeline(model, tokenizer, config)
results = pipeline.run(
    seed_file="synthetic_data_pipeline/prompts/generation_prompts.yaml"
)

# 결과 저장
pipeline.save(results, "synthetic_data_pipeline/output/generated_data.jsonl")
print(f"생성된 샘플: {len(results.samples)}")
print(f"평균 품질 점수: {results.stats.avg_quality_score:.2f}")
```

**개별 컴포넌트 사용:**

```python
# 1. Instruction 생성
from synthetic_data_pipeline.src.data_generator import InstructionGenerator

generator = InstructionGenerator(model, tokenizer)
instructions = generator.generate_from_seeds(
    seeds=["Write a Python function", "Explain machine learning"],
    num_per_seed=10
)

# 2. 품질 평가
from synthetic_data_pipeline.src.reward_model import QualityEvaluator

evaluator = QualityEvaluator(model, tokenizer)
scores = evaluator.evaluate(instruction, response)
print(f"Helpfulness: {scores.helpfulness:.2f}")
print(f"Correctness: {scores.correctness:.2f}")
print(f"총점: {scores.weighted_average:.2f}")

# 3. 필터링
from synthetic_data_pipeline.src.data_filter import CombinedFilter, FilterConfig

filter_config = FilterConfig(
    min_instruction_length=10,
    min_response_length=50,
    min_quality_score=0.6
)
data_filter = CombinedFilter(filter_config)
filtered_samples = data_filter.filter(samples)

# 4. 다양성 샘플링
from synthetic_data_pipeline.src.diversity_sampler import DiversitySampler

sampler = DiversitySampler(num_clusters=50)
diverse_samples = sampler.sample(filtered_samples, num_samples=500)
```

---

## 참고 자료

### NVIDIA 공식 자료

- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [TensorRT-LLM Speculative Decoding](https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/)
- [NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)
- [KV Cache Reuse](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/)
- [Nemotron-4 340B](https://blogs.nvidia.com/blog/nemotron-4-synthetic-data-generation-llm-training/)
- [Nemotron 3 Nano Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)

### 학술 논문

- "Accelerating Large Language Model Decoding with Speculative Sampling" (Leviathan et al., 2023)
- "Medusa: Simple LLM Inference Acceleration Framework" (Cai et al., 2024)
- "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs"
- "Self-Instruct: Aligning Language Model with Self Generated Instructions"

---

## 디렉토리 구조

```
nvidia/
├── README.md                      # 이 파일
├── speculative_decoding/          # 프로젝트 1
│   ├── src/
│   │   ├── draft_target.py       # Draft-Target 구현
│   │   ├── medusa_heads.py       # Medusa 구현
│   │   ├── tree_attention.py     # Tree Attention 구현
│   │   └── benchmark.py          # 벤치마크 도구
│   └── README.md
├── kv_cache_optimization/         # 프로젝트 2
│   ├── src/
│   │   ├── kv_cache_quantization.py  # INT8/INT4 양자화
│   │   ├── kv_cache_eviction.py      # Token Eviction
│   │   ├── kv_cache_compression.py   # 압축 기법
│   │   └── memory_profiler.py        # 메모리 프로파일링
│   └── README.md
└── synthetic_data_pipeline/       # 프로젝트 3
    ├── src/
    │   ├── data_generator.py     # 데이터 생성
    │   ├── reward_model.py       # 품질 평가
    │   ├── data_filter.py        # 필터링
    │   ├── diversity_sampler.py  # 다양성 샘플링
    │   └── pipeline.py           # 전체 파이프라인
    ├── prompts/                   # 프롬프트 템플릿
    └── README.md
```

---

## 라이선스

MIT License

---

*이 프로젝트는 NVIDIA의 공개된 기술 블로그, 논문, 오픈소스 코드를 참고하여 학습 목적으로 구현되었습니다.*
