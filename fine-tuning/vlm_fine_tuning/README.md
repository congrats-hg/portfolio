# Document-Specialized Vision-Language Model

> Qwen2-VL-2B를 문서 이해에 특화시킨 LoRA fine-tuning 프로젝트

<!-- Badges -->
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-Apache_2.0-green)

---

## Demo

<!-- TODO: 실험 완료 후 결과 이미지/GIF 추가 -->
```
[Invoice 이미지]
Q: "What is the total amount?"
A: "$1,234.56" ✓
```

---

## Highlights

| | |
|:--|:--|
| **+💛💛💛%** | DocVQA 성능 향상 (vs zero-shot baseline) |
| **💛💛💛M** | 학습 파라미터 (전체의 💛💛💛%) |
| **💛💛💛ms** | 추론 속도 (single image) |

---

## Why This Project?

대형 VLM(GPT-4V, Claude)은 문서 이해에 강력하지만, **비용과 지연시간** 문제로 실무 도입이 어렵습니다.

이 프로젝트는 세 가지 질문에 답합니다:

1. **2B 모델로도 실용적인 문서 이해가 가능한가?**
2. **어떤 문서 유형에서 fine-tuning 효과가 큰가?**
3. **실제 실패 케이스는 무엇이고, 어떻게 개선할 수 있는가?**

---

## Quick Start

```bash
pip install torch transformers peft qwen-vl-utils
```

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# Load
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
model = PeftModel.from_pretrained(model, "💛💛💛/document-vlm-lora")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# 전체 추론 코드는 scripts/inference.py 참조
```

---

## Results

### Main Benchmark

| Benchmark | Baseline | Ours | Δ |
|-----------|----------|------|---|
| DocVQA (ANLS) | 💛💛💛 | 💛💛💛 | **+💛💛💛** |
| ChartQA (Acc) | 💛💛💛 | 💛💛💛 | **+💛💛💛** |
| InfoVQA (ANLS) | 💛💛💛 | 💛💛💛 | **+💛💛💛** |

> Baseline = Qwen2-VL-2B-Instruct zero-shot

### vs Other Models

| Model | Size | DocVQA |
|-------|------|--------|
| GPT-4V | - | 💛💛💛 |
| Qwen2-VL-7B | 7B | 💛💛💛 |
| **Ours** | **2B** | **💛💛💛** |

---

## Tech Stack

![Qwen2-VL](https://img.shields.io/badge/Base-Qwen2--VL--2B-9B59B6)
![LoRA](https://img.shields.io/badge/Method-LoRA-E67E22)
![PEFT](https://img.shields.io/badge/Library-PEFT-3498DB)
![Wandb](https://img.shields.io/badge/Tracking-Wandb-FFCC00)

---

## Project Structure

```
├── configs/          # 학습 설정
├── scripts/          # 학습/평가/추론 스크립트
├── src/              # 데이터셋, 모델, 유틸리티
├── notebooks/        # 분석 노트북
└── results/          # 실험 결과
```

---

## What I Learned

<!-- TODO: 실험 후 작성 -->
- 💛💛💛 (e.g., "LoRA rank와 성능의 trade-off 관계 이해")
- 💛💛💛 (e.g., "문서 유형별 데이터 불균형이 성능에 미치는 영향")
- 💛💛💛 (e.g., "VLM 평가 메트릭(ANLS) 특성과 한계")

---

## Limitations & Future Work

**Limitations**
- 영어 문서만 평가 (다국어 미지원)
- 단일 페이지 문서만 처리
- Single seed (42) 결과

**Future Work**
- [ ] 다국어 문서 지원 (한국어, 중국어)
- [ ] Multi-page 문서 처리
- [ ] 양자화를 통한 Edge 배포

---

<details>
<summary><strong>Technical Details</strong> (클릭하여 펼치기)</summary>

### Model Configuration

| | |
|---|---|
| Base Model | Qwen2-VL-2B-Instruct |
| Fine-tuning | LoRA (r=💛💛💛, α=💛💛💛) |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
| Vision Encoder | Frozen |
| Trainable Params | 💛💛💛M (💛💛💛%) |

### Training Setup

```yaml
training:
  epochs: 💛💛💛
  batch_size: 💛💛💛
  learning_rate: 💛💛💛
  scheduler: cosine

data:
  min_pixels: 💛💛💛
  max_pixels: 💛💛💛
```

### Dataset

| Source | Samples | Type |
|--------|---------|------|
| 💛💛💛 | 💛💛💛 | 💛💛💛 |
| 💛💛💛 | 💛💛💛 | 💛💛💛 |
| **Total** | **💛💛💛** | - |

### Evaluation Protocol

- **Metrics**: ANLS (DocVQA, InfoVQA), Relaxed Accuracy (ChartQA)
- **Baseline**: Qwen2-VL-2B-Instruct zero-shot
- **Data Leakage**: Train/eval 데이터 분리 검증 완료

### Performance by Document Type

| Type | Baseline | Ours | Δ |
|------|----------|------|---|
| Forms | 💛💛💛 | 💛💛💛 | +💛💛💛 |
| Tables | 💛💛💛 | 💛💛💛 | +💛💛💛 |
| Invoices | 💛💛💛 | 💛💛💛 | +💛💛💛 |

### Ablation: LoRA Rank

| Rank | DocVQA | Params |
|------|--------|--------|
| 16 | 💛💛💛 | 💛💛💛M |
| 32 | 💛💛💛 | 💛💛💛M |
| 64 | 💛💛💛 | 💛💛💛M |

### Failure Analysis

| Error Type | Frequency | Example |
|------------|-----------|---------|
| 💛💛💛 | 💛💛💛% | 💛💛💛 |
| 💛💛💛 | 💛💛💛% | 💛💛💛 |

### Inference Performance

| GPU | Latency | Cost/1K images |
|-----|---------|----------------|
| A100 | 💛💛💛 ms | $💛💛💛 |
| T4 | 💛💛💛 ms | $💛💛💛 |
| GPT-4V API | 💛💛💛 ms | $💛💛💛 |

</details>

---

<details>
<summary><strong>Reproducibility</strong> (클릭하여 펼치기)</summary>

### Environment

```
Python: 3.10.12
CUDA: 12.1
OS: Ubuntu 22.04 LTS
```

### Requirements

```
torch==2.1.2
transformers==4.37.2
peft==0.7.1
accelerate==0.25.0
datasets==2.16.1
wandb==0.16.2
qwen-vl-utils==0.0.2
```

### Training

```bash
python scripts/train.py --config configs/document_lora.yaml
```

### Evaluation

```bash
python scripts/evaluate.py \
    --model_path outputs/checkpoint-final \
    --benchmarks docvqa chartqa infovqa
```

### Full Inference Code

```python
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info

# Load model
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "💛💛💛/document-vlm-lora")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Prepare input
image = Image.open("invoice.png")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is the invoice total?"},
        ],
    }
]

# Process
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)

# Generate
output = model.generate(**inputs, max_new_tokens=256)
generated_ids = output[:, inputs.input_ids.shape[1]:]
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

</details>

---

## License

Apache 2.0

---

## Links

**Model**: [huggingface.co/💛💛💛/document-vlm-lora](https://huggingface.co/💛💛💛/document-vlm-lora)
**Wandb**: [wandb.ai/💛💛💛/document-vlm](https://wandb.ai/💛💛💛/document-vlm)
**Contact**: 💛💛💛
