"""
Multi-News Prompt Evaluation Script
Dataset: Awesome075/multi_news_parquet
Model: Qwen/Qwen2.5-7B-Instruct
Task: Multi-document summarization (English)

Evaluates multiple prompt templates with/without system message.
Results saved to JSON for analysis.
"""
import json
import os
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer


# Settings
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
NUM_SAMPLES = 200
MAX_DOC_LENGTH = 6000  # Character limit for document truncation
MAX_NEW_TOKENS = 300   # For summary generation
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_rouge_single(prediction, reference):
    """Compute ROUGE scores for a single prediction-reference pair"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }


# System message
SYSTEM_MESSAGE = """You are a professional summarization model.
You create concise, accurate summaries that capture key information from multiple news articles."""

# Prompt templates
PROMPTS = {
    "prompt_1": """Below are multiple news articles about the same topic, separated by |||||.
Please summarize the key points from all articles.

Articles:
{document}

Summary:""",

    "prompt_2": """The following news articles cover the same event. Read all articles and provide a comprehensive summary.

{document}

Provide a summary that captures the main points:""",

    "prompt_3": """Multiple news sources are provided below (separated by |||||).
Synthesize the information and write a concise summary.

Sources:
{document}

Summary:""",

    "prompt_4": """Read the following news articles and write a summary.

Articles:
{document}

Write a summary (200-300 words):""",

    "prompt_5": """You are given multiple news articles about the same topic.
Extract the key information and create a unified summary.

{document}

Summary:""",
}


def format_data(sample, prompt_template, system_message=None, max_doc_length=MAX_DOC_LENGTH):
    """Format sample for model input"""
    document = sample["document"]

    # Truncate if too long
    if len(document) > max_doc_length:
        document = document[:max_doc_length]
        last_period = document.rfind('.')
        if last_period > max_doc_length * 0.8:
            document = document[:last_period + 1]

    messages = []

    if system_message:
        messages.append({
            "role": "system",
            "content": system_message,
        })

    messages.extend([
        {
            "role": "user",
            "content": prompt_template.format(document=document),
        },
        {
            "role": "assistant",
            "content": sample["summary"].strip(),
        },
    ])

    return {"messages": messages}


def evaluate_model(model, tokenizer, test_samples, num_samples=100, max_new_tokens=MAX_NEW_TOKENS):
    """Evaluate model and return ROUGE scores with outputs"""
    model.eval()
    total = min(num_samples, len(test_samples))
    outputs = []

    all_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for i, sample in enumerate(test_samples[:total]):
        # Prepare input (exclude assistant message)
        messages = sample["messages"][:-1]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        output = tokenizer.decode(
            generated_ids[0, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        expected = sample["messages"][-1]["content"].strip()

        # Compute ROUGE
        scores = compute_rouge_single(output, expected)
        for key in all_scores:
            all_scores[key].append(scores[key])

        outputs.append({
            "output": output,
            "rouge1": scores['rouge1'],
            "rouge2": scores['rouge2'],
            "rougeL": scores['rougeL'],
        })

        if (i + 1) % 20 == 0:
            avg_rouge1 = sum(all_scores['rouge1']) / len(all_scores['rouge1'])
            print(f"  Evaluated {i + 1}/{total}, avg ROUGE-1: {avg_rouge1:.4f}")

    # Compute averages
    avg_scores = {key: sum(vals) / len(vals) for key, vals in all_scores.items()}

    return avg_scores, outputs


def run_evaluation(model, tokenizer, test_subset, system_message, output_path):
    """Run evaluation for all prompts with/without system message"""
    mode = "with" if system_message else "without"
    print(f"\n{'=' * 60}")
    print(f"Evaluating {mode} system message...")
    print("=" * 60)

    # Prepare sample metadata
    samples_data = [
        {
            "index": i,
            "document_preview": row["document"][:200] + "...",
            "expected": row["summary"],
            "outputs": {}
        }
        for i, row in enumerate(test_subset)
    ]

    summary = {}

    for prompt_name, prompt_template in PROMPTS.items():
        print(f"\n[{prompt_name}]")
        test_formatted = [
            format_data(row, prompt_template, system_message)
            for row in test_subset
        ]

        scores, outputs = evaluate_model(
            model, tokenizer, test_formatted, num_samples=NUM_SAMPLES
        )

        # Add outputs to sample data
        for i, out in enumerate(outputs):
            samples_data[i]["outputs"][prompt_name] = {
                "generated": out["output"],
                "rouge1": out["rouge1"],
                "rouge2": out["rouge2"],
                "rougeL": out["rougeL"],
            }

        summary[prompt_name] = {
            "rouge1": scores['rouge1'],
            "rouge2": scores['rouge2'],
            "rougeL": scores['rougeL'],
        }

    # Compute mean across prompts
    mean_rouge1 = sum(d["rouge1"] for d in summary.values()) / len(summary)
    mean_rouge2 = sum(d["rouge2"] for d in summary.values()) / len(summary)
    mean_rougeL = sum(d["rougeL"] for d in summary.values()) / len(summary)

    # Find best prompt (by ROUGE-1)
    best_prompt = max(summary.items(), key=lambda x: x[1]["rouge1"])

    results = {
        "metadata": {
            "model": MODEL_ID,
            "dataset": "Awesome075/multi_news_parquet",
            "num_samples": NUM_SAMPLES,
            "system_message": system_message if system_message else None,
            "timestamp": datetime.now().isoformat(),
            "mean_rouge1": mean_rouge1,
            "mean_rouge2": mean_rouge2,
            "mean_rougeL": mean_rougeL,
            "best_prompt": best_prompt[0],
            "best_rouge1": best_prompt[1]["rouge1"],
        },
        "summary": summary,
        "samples": samples_data
    }

    # Save JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nBest prompt: {best_prompt[0]} (ROUGE-1: {best_prompt[1]['rouge1']:.4f})")
    print(f"Results saved to: {output_path}")

    # Print table
    print("\n| Prompt | ROUGE-1 | ROUGE-2 | ROUGE-L |")
    print("|--------|---------|---------|---------|")
    for name, data in summary.items():
        marker = " *" if name == best_prompt[0] else ""
        print(f"| {name}{marker} | {data['rouge1']:.4f} | {data['rouge2']:.4f} | {data['rougeL']:.4f} |")

    return summary


def main():
    print("=" * 60)
    print("Multi-News Prompt Evaluation Script")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("Awesome075/multi_news_parquet")
    test_dataset = dataset["test"]
    print(f"Test samples: {len(test_dataset)}")

    # Load model
    print(f"\nLoading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Prepare test subset
    test_subset = list(test_dataset)[:NUM_SAMPLES]

    # 1. Evaluate with system message
    run_evaluation(
        model, tokenizer, test_subset,
        system_message=SYSTEM_MESSAGE,
        output_path=os.path.join(SCRIPT_DIR, "prompt_results_w_system_message.json")
    )

    # 2. Evaluate without system message
    run_evaluation(
        model, tokenizer, test_subset,
        system_message=None,
        output_path=os.path.join(SCRIPT_DIR, "prompt_results_wo_system_message.json")
    )

    print("\n" + "=" * 60)
    print("All evaluations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
