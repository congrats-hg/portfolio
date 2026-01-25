# Document-Specialized Vision-Language Model

> LoRA fine-tuning Qwen2-VL-2B for enterprise document understanding

<!-- Badges -->
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-Apache_2.0-green)

---

## Demo

<!-- TODO: Add result image/GIF after experiments -->
```
[Invoice Image]
Q: "What is the total amount?"
A: "$1,234.56" ✓
```

---

## Highlights

| | |
|:--|:--|
| **+💛💛💛%** | DocVQA improvement (vs zero-shot baseline) |
| **💛💛💛M** | Trainable parameters (💛💛💛% of total) |
| **💛💛💛ms** | Inference latency (single image) |

---

## Motivation

Large VLMs (GPT-4V, Claude) excel at document understanding, but **cost and latency** make production deployment challenging.

This project answers three questions:

1. **Can a 2B model achieve practical document understanding?**
2. **Which document types benefit most from fine-tuning?**
3. **What are the real-world failure modes, and how can we address them?**

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

# See scripts/inference.py for full inference code
```

---

## Results

### Main Benchmarks

| Benchmark | Baseline | Ours | Δ |
|-----------|----------|------|---|
| DocVQA (ANLS) | 💛💛💛 | 💛💛💛 | **+💛💛💛** |
| ChartQA (Acc) | 💛💛💛 | 💛💛💛 | **+💛💛💛** |
| InfoVQA (ANLS) | 💛💛💛 | 💛💛💛 | **+💛💛💛** |

> Baseline = Qwen2-VL-2B-Instruct zero-shot

### Comparison with Other Models

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
├── configs/          # Training configurations
├── scripts/          # Train/eval/inference scripts
├── src/              # Dataset, model, utilities
├── notebooks/        # Analysis notebooks
└── results/          # Experiment results
```

---

## Key Takeaways

<!-- TODO: Fill after experiments -->
- 💛💛💛 (e.g., "Understanding the trade-off between LoRA rank and performance")
- 💛💛💛 (e.g., "Impact of document type imbalance on model performance")
- 💛💛💛 (e.g., "Characteristics and limitations of ANLS metric for VLM evaluation")

---

## Limitations & Future Work

**Limitations**
- English documents only (multilingual not evaluated)
- Single-page documents only
- Single seed (42) results

**Future Work**
- [ ] Multilingual support (Korean, Chinese)
- [ ] Multi-page document understanding
- [ ] Quantization for edge deployment

---

<details>
<summary><strong>Technical Details</strong> (Click to expand)</summary>

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
- **Data Leakage**: Verified no overlap between train/eval splits

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
<summary><strong>Reproducibility</strong> (Click to expand)</summary>

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
