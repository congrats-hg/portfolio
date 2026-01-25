"""
프롬프트별 Base 모델 평가 및 JSON 저장 스크립트
노트북과 별개로 실행하여 결과를 저장합니다.
한 번 실행으로 시스템 메시지 있는/없는 버전 모두 평가합니다.
"""
import json
import os
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor


# 설정
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
NUM_SAMPLES = 500
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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

# 시스템 메시지
SYSTEM_MESSAGE = "You are a chart analysis model that extracts precise answers from charts and graphs."

# 프롬프트 정의 (단순 문자열 출력)
PROMPTS = {
    "prompt_1": """Answer the following question based on the chart.
Question: {question}

Provide only the answer (number or short text), no explanation.
Answer:""",

    "prompt_2": """Based on the chart image, answer the question.
Question: {question}

Provide only the answer (number or short text), no explanation.
Answer:""",

    "prompt_3": """Analyze the chart image and answer the question.
Question: {question}

Provide only the answer (number or short text), no explanation.
Answer:""",

    "prompt_4": """Analyze the chart and answer.
Question: {question}

Provide only the answer (number or short text), no explanation.
Answer: """
}


def format_data(sample, prompt_template, system_message=None):
    """데이터를 모델 입력 형식으로 변환"""
    messages = []

    # 시스템 메시지가 있으면 추가
    if system_message:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        })

    messages.extend([
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
    ])

    return {"messages": messages}


def evaluate_model(model, processor, test_samples, num_samples=100):
    """모델 평가 - 각 샘플의 출력도 반환"""
    model.eval()
    correct = 0
    total = min(num_samples, len(test_samples))
    outputs = []

    for i, sample in enumerate(test_samples[:total]):
        # 시스템 메시지 유무에 따라 user 메시지 인덱스가 다름
        messages = sample["messages"][:-1]  # assistant 제외
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # user 메시지에서 이미지 찾기
        for msg in messages:
            if msg["role"] == "user":
                image = msg["content"][0]["image"]
                break

        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)

        output = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()

        expected = sample["messages"][-1]["content"][0]["text"].strip()
        is_correct = relaxed_match(output, expected)
        if is_correct:
            correct += 1

        outputs.append({
            "output": output,
            "correct": is_correct
        })

        if (i + 1) % 20 == 0:
            print(f"  Evaluated {i + 1}/{total}, accuracy: {correct/(i+1):.2%}")

    return correct / total, outputs


def run_evaluation(model, processor, test_subset, system_message, output_path):
    """시스템 메시지 유무에 따른 평가 실행"""
    mode = "with" if system_message else "without"
    print(f"\n{'=' * 60}")
    print(f"Evaluating {mode} system message...")
    print("=" * 60)

    # 샘플 데이터 준비
    samples_data = [
        {
            "index": i,
            "query": row["query"],
            "expected": row["label"][0],
            "outputs": {}
        }
        for i, row in enumerate(test_subset)
    ]

    summary = {}

    for prompt_name, prompt_template in PROMPTS.items():
        print(f"\n[{prompt_name}]")
        test_formatted = [format_data(row, prompt_template, system_message) for row in test_subset]
        accuracy, outputs = evaluate_model(model, processor, test_formatted, num_samples=NUM_SAMPLES)

        # 각 샘플에 프롬프트별 출력 추가
        for i, out in enumerate(outputs):
            samples_data[i]["outputs"][prompt_name] = out["output"]

        summary[prompt_name] = {
            "accuracy": accuracy,
            "accuracy_pct": f"{accuracy:.2%}"
        }

    # 평균 정확도 계산
    mean_accuracy = sum(data["accuracy"] for data in summary.values()) / len(summary)
    print(f"\nMean accuracy across all prompts: {mean_accuracy:.2%}")

    # 최적 프롬프트 찾기
    best_prompt = max(summary.items(), key=lambda x: x[1]["accuracy"])

    results = {
        "metadata": {
            "model": MODEL_ID,
            "dataset": "HuggingFaceM4/ChartQA",
            "num_samples": NUM_SAMPLES,
            "system_message": system_message if system_message else None,
            "timestamp": datetime.now().isoformat(),
            "mean_accuracy": mean_accuracy,
            "best_prompt": best_prompt[0],
            "best_accuracy": best_prompt[1]["accuracy"]
        },
        "summary": summary,
        "samples": samples_data
    }

    # JSON 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nBest prompt: {best_prompt[0]} ({best_prompt[1]['accuracy']:.2%})")
    print(f"Results saved to: {output_path}")

    # 표 출력
    print("\n| Prompt | Accuracy |")
    print("|--------|----------|")
    for name, data in summary.items():
        marker = " *" if name == best_prompt[0] else ""
        print(f"| {name}{marker} | {data['accuracy_pct']} |")

    return summary


def main():
    print("=" * 60)
    print("ChartQA Prompt Evaluation Script")
    print("=" * 60)

    # 데이터셋 로드
    print("\nLoading dataset...")
    dataset = load_dataset("HuggingFaceM4/ChartQA")
    test_dataset = dataset["test"]
    print(f"Test samples: {len(test_dataset)}")

    # 모델 로드
    print(f"\nLoading model: {MODEL_ID}")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # 테스트 데이터 준비
    test_subset = list(test_dataset)[:NUM_SAMPLES]

    # 1. 시스템 메시지 있는 버전 평가
    run_evaluation(
        model, processor, test_subset,
        system_message=SYSTEM_MESSAGE,
        output_path=os.path.join(SCRIPT_DIR, "prompt_results_w_system_message.json")
    )

    # 2. 시스템 메시지 없는 버전 평가
    run_evaluation(
        model, processor, test_subset,
        system_message=None,
        output_path=os.path.join(SCRIPT_DIR, "prompt_results_wo_system_message.json")
    )

    print("\n" + "=" * 60)
    print("All evaluations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
