"""
Base vs Full Fine-tuned 모델 평가 스크립트
Usage: python evaluate.py
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

# 설정
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FULL_FT_PATH = os.path.join(SCRIPT_DIR, "qwen25vl-chartqa-full-ft")
NUM_SAMPLES = 1000

# 프롬프트 설정
system_message = "You are a chart analysis model that extracts precise answers from charts and graphs."
prompt = """Based on the chart image, answer the question.
Question: {question}

Provide only the answer (number or short text), no explanation.
Answer:"""


def format_data(sample, prompt_template):
    """ChartQA 데이터를 평가 포맷으로 변환"""
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": prompt_template.format(question=sample["query"])},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["label"][0]}],
            },
        ],
    }


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
        messages = sample["messages"][:2]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = sample["messages"][1]["content"][0]["image"]
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)

        output = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()

        expected = sample["messages"][2]["content"][0]["text"].strip()
        if relaxed_match(output, expected):
            correct += 1

        if (i + 1) % 20 == 0:
            print(f"  Evaluated {i + 1}/{total} samples, current accuracy: {correct/(i+1):.2%}")

    return correct / total


def main():
    print("=" * 50)
    print("ChartQA Model Evaluation")
    print("=" * 50)

    # 데이터셋 로드
    print("\n[1/4] Loading dataset...")
    dataset = load_dataset("HuggingFaceM4/ChartQA")
    test_dataset = dataset["test"]
    test_formatted = [format_data(row, prompt) for row in test_dataset]
    print(f"  Test samples: {len(test_formatted)}")

    # 프로세서 로드
    print("\n[2/4] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=512 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    # # Base 모델 평가
    # print("\n[3/4] Evaluating Base model...")
    # base_model = AutoModelForImageTextToText.from_pretrained(
    #     MODEL_ID,
    #     device_map="auto",
    #     dtype=torch.bfloat16
    # )
    # base_accuracy = evaluate_model(base_model, processor, test_formatted, num_samples=NUM_SAMPLES)
    # print(f"  Base model accuracy: {base_accuracy:.2%}")
    # del base_model
    # torch.cuda.empty_cache()

    # Full FT 모델 평가
    print("\n[4/4] Evaluating Full Fine-tuned model...")
    full_ft_model = AutoModelForImageTextToText.from_pretrained(
        FULL_FT_PATH,
        device_map="auto",
        dtype=torch.bfloat16
    )
    full_ft_accuracy = evaluate_model(full_ft_model, processor, test_formatted, num_samples=NUM_SAMPLES)
    print(f"  Full FT model accuracy: {full_ft_accuracy:.2%}")
    del full_ft_model
    torch.cuda.empty_cache()

    base_accuracy = 0.7170

    # 결과 출력
    print("\n" + "=" * 50)
    print("Results Summary")
    print("=" * 50)
    print(f"\n| Model   | Accuracy |")
    print(f"|---------|----------|")
    print(f"| Base    | {base_accuracy:.2%}   |")
    print(f"| Full FT | {full_ft_accuracy:.2%}   |")
    print(f"\nImprovement: {(full_ft_accuracy - base_accuracy) * 100:+.2f}%p")


if __name__ == "__main__":
    main()
