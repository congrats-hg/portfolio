"""
ChartQA LoRA Fine-tuning Script
dataset: HuggingFaceM4/ChartQA
model: Qwen/Qwen2.5-VL-3B-Instruct
"""

from datasets import load_dataset
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, PeftModel

import transformers
transformers.logging.set_verbosity_info()

import wandb
wandb.init(project="chartqa-finetuning")

# ============================================================
# Configuration
# ============================================================
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
NUM_SAMPLES = 1000  # evaluation samples

system_message = "You are a chart analysis model that extracts precise answers from charts and graphs."

prompt = """Based on the chart image, answer the question.
Question: {question}

Provide only the answer (number or short text), no explanation.
Answer:"""

# ============================================================
# Data Formatting
# ============================================================
def format_data(sample, prompt_template):
    """ChartQA 데이터를 학습 포맷으로 변환"""
    image = sample["image"]

    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_template.format(question=sample["query"])},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["label"][0]}],
            },
        ],
    }

# ============================================================
# Evaluation Functions
# ============================================================
def relaxed_match(output, expected):
    """ChartQA relaxed accuracy: 숫자는 ±5% 허용"""
    output, expected = output.strip(), expected.strip()
    if output.lower() == expected.lower():
        return True
    try:
        out_num = float(output.replace(",", "").replace("%", ""))
        exp_num = float(expected.replace(",", "").replace("%", ""))
        if exp_num == 0:
            return out_num == 0
        return abs(out_num - exp_num) <= abs(exp_num) * 0.05
    except:
        return False


def evaluate_model(model, processor, test_samples, num_samples=100):
    """테스트 샘플에 대해 정확도 측정"""
    model.eval()
    correct = 0
    total = min(num_samples, len(test_samples))

    for i, sample in enumerate(test_samples[:total]):
        # 추론 수행 (system + user 메시지만 사용)
        messages = sample["messages"][:2]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 이미지 추출
        image = sample["messages"][1]["content"][0]["image"]

        # 입력 준비
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

        # 생성
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)

        # 디코딩
        output = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()

        # 정답 비교 (relaxed match)
        expected = sample["messages"][2]["content"][0]["text"].strip()
        if relaxed_match(output, expected):
            correct += 1

        if (i + 1) % 20 == 0:
            print(f"Evaluated {i + 1}/{total} samples, current accuracy: {correct/(i+1):.2%}")

    return correct / total

# ============================================================
# Collate Function
# ============================================================
def collate_fn(examples):
    """
    텍스트와 이미지가 포함된 대화 데이터를 모델 학습에 적합한 형태로 변환
    """
    # 1. 텍스트 전처리 - 채팅 템플릿 적용
    texts = [processor.apply_chat_template(ex["messages"], tokenize=False) for ex in examples]

    # 2. 이미지 데이터 추출 및 전처리
    image_inputs = [process_vision_info(ex["messages"])[0] for ex in examples]

    # 3. 텍스트 토크나이징 + 이미지 인코딩
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    # 4. 라벨 생성 (손실 계산용)
    labels = batch["input_ids"].clone()

    # 5. 패딩 토큰 손실 계산에서 제외
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # 6. 이미지 토큰 손실 계산에서 제외
    image_tokens = [151652, 151653, 151655]
    for token_id in image_tokens:
        labels[labels == token_id] = -100

    # 7. assistant 응답 이전 토큰들 마스킹
    assistant_start_tokens = processor.tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )

    for i, input_ids in enumerate(batch["input_ids"]):
        input_ids_list = input_ids.tolist()

        # assistant 시작 위치 찾기
        for j in range(len(input_ids_list) - len(assistant_start_tokens) + 1):
            if input_ids_list[j:j + len(assistant_start_tokens)] == assistant_start_tokens:
                response_start = j + len(assistant_start_tokens)
                labels[i, :response_start] = -100
                break

    batch["labels"] = labels
    return batch


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # 데이터셋 로드
    print("Loading dataset...")
    dataset = load_dataset("HuggingFaceM4/ChartQA")
    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    test_dataset = dataset["test"]

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")

    # 데이터 포맷팅
    print("Formatting data...")
    train_formatted = [format_data(row, prompt) for row in train_dataset]
    val_formatted = [format_data(row, prompt) for row in val_dataset]
    test_formatted = [format_data(row, prompt) for row in test_dataset]

    # 모델 및 프로세서 로드
    print("Loading model and processor...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=512 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    # LoRA 설정
    peft_config = LoraConfig(
        lora_alpha=64,
        lora_dropout=0.05,
        r=32,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
    )

    # LoRA Fine-tuning 설정
    lora_args = SFTConfig(
        output_dir="qwen25vl-chartqa-lora",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        bf16=True,
        learning_rate=1e-4,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="wandb"
    )

    # LoRA Trainer 생성 및 학습
    print("Starting LoRA fine-tuning...")
    lora_trainer = SFTTrainer(
        model=model,
        args=lora_args,
        train_dataset=train_formatted,
        eval_dataset=val_formatted,
        data_collator=collate_fn,
        peft_config=peft_config,
        processing_class=processor.tokenizer
    )

    lora_trainer.train()
    lora_trainer.save_model()
    print("LoRA fine-tuning completed!")

    # ============================================================
    # Evaluation
    # ============================================================
    print("\n" + "="*60)
    print("Starting Evaluation...")
    print("="*60 + "\n")

    # Base 모델 평가
    print("Loading Base model...")
    base_model_eval = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    print("Evaluating Base model...")
    base_accuracy = evaluate_model(base_model_eval, processor, test_formatted, num_samples=NUM_SAMPLES)
    print(f"\nBase model accuracy: {base_accuracy:.2%}")
    del base_model_eval
    torch.cuda.empty_cache()

    # LoRA 모델 평가
    print("\nLoading LoRA model...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    lora_model = PeftModel.from_pretrained(base_model, "qwen25vl-chartqa-lora")
    print("Evaluating LoRA model...")
    lora_accuracy = evaluate_model(lora_model, processor, test_formatted, num_samples=NUM_SAMPLES)
    print(f"\nLoRA model accuracy: {lora_accuracy:.2%}")

    # 결과 요약
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"| Model | Accuracy |")
    print(f"|-------|----------|")
    print(f"| Base  | {base_accuracy:.2%}   |")
    print(f"| LoRA  | {lora_accuracy:.2%}   |")

    wandb.finish()
