import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random
from datetime import datetime, timedelta

def random_yeardate():
    # 시작 날짜와 종료 날짜 설정 (원하는 범위 내에서 날짜를 생성할 수 있도록)
    start_date = datetime(2025, 10, 27)
    end_date = datetime(2025, 12, 31)

    # 시작 날짜와 종료 날짜 사이에서 랜덤한 날짜 생성
    random_days = random.randint(0, (end_date - start_date).days)
    random_date = start_date + timedelta(days=random_days)

    return random_date.strftime("%Y년 %m월 %d일")

with open("/home/ryu5090/project/kyeongin/ft/briefing/summary/2_eval_880.json", "r", encoding='utf-8') as f:
    eval_880 = json.load(f)

def format_data(dataset):
    user_prompt = dataset["input"]["prompt"]
    
    return {
        "messages": [
            {
                "role": "user",
                "content": user_prompt
            },
            {
                "role": "assistant",
                "content": dataset["output"]
            }
        ]
    }

test_dataset = [format_data(dataset) for dataset in eval_880]

# print(test_dataset[:2])

model_id = "kakaocorp/kanana-1.5-2.1b-instruct-2505"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda:1",
    dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

text = tokenizer.apply_chat_template(
    test_dataset[0]["messages"], tokenize=False, add_generation_prompt=False
)

prompt_lst = []
label_lst = []

for dataset_item in test_dataset:
    # 전체 대화를 템플릿에 적용
    text = tokenizer.apply_chat_template(
        dataset_item["messages"], tokenize=False, add_generation_prompt=False
    )
    
    # prompt는 assistant 응답 직전까지
    prompt = text.split('<|start_header_id|>assistant<|end_header_id|>')[0] + '<|start_header_id|>assistant<|end_header_id|>\n\n'
    
    # label은 실제 assistant 응답
    label = dataset_item["messages"][1]["content"]  # assistant의 응답
    
    prompt_lst.append(prompt)
    label_lst.append(label)

print(f"Total prompts: {len(prompt_lst)}")
print("Sample prompt:")
print(prompt_lst[0])
print("Sample label:")
print(label_lst[0])


# 파인튜닝 모델 테스트

import torch
from peft import AutoPeftModelForCausalLM
from transformers import  AutoTokenizer, pipeline

peft_model_id = "/home/ryu5090/project/kyeongin/ft/briefing/summary_ft_model/checkpoint-1850"
fine_tuned_model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id, device_map="cuda:1", torch_dtype=torch.float16)
pipe = pipeline("text-generation", model=fine_tuned_model, tokenizer=tokenizer)

eos_token = tokenizer("<|eot_id|>",add_special_tokens=False)["input_ids"][0]

def test_inference(pipe, prompt):
    outputs = pipe(prompt, max_new_tokens=1024, eos_token_id=eos_token, do_sample=False)
    return outputs[0]['generated_text'][len(prompt):].strip()

for prompt, label in zip(prompt_lst[300:305], label_lst[300:305]):
    # print(f"    prompt:\n{prompt}")
    print(f"    response:\n{test_inference(pipe, prompt)}")
    print(f"    label:\n{label}")
    print("-"*50)


# 기본 모델 테스트

base_model_id = "kakaocorp/kanana-1.5-2.1b-instruct-2505"
model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="cuda:1", torch_dtype=torch.float16)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

for prompt, label in zip(prompt_lst[300:305], label_lst[300:305]):
    # print(f"    prompt:\n{prompt}")
    print(f"    response:\n{test_inference(pipe, prompt)}")
    print(f"    label:\n{label}")
    print("-"*50)