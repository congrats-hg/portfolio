from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

import json
import random

# import trl, transformers
# print(trl.__version__, transformers.__version__) # 0.23.1 4.56.2

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

train_dataset = [format_data(dataset) for dataset in train_3300]

# 데이터 확인
# print(train_dataset[100]["messages"])

model_id = "kakaocorp/kanana-1.5-2.1b-instruct-2505"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda:1",
    dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

text = tokenizer.apply_chat_template(
    train_dataset[0]["messages"], tokenize=False, add_generation_prompt=False
)

# 템플릿 적용 확인
print(text)

# 리스트 형태에서 다시 Dataset 객체로 변경
# print(type(train_dataset)) # <class 'list'>
train_dataset = Dataset.from_list(train_dataset)
# print(type(train_dataset)) # <class 'datasets.arrow_dataset.Dataset'>

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

# 최대 길이
max_seq_length=2048

args = SFTConfig(
    output_dir="/home/ryu5090/project/kyeongin/ft/briefing/summary_ft_model",           # 저장될 디렉토리와 저장소 ID
    max_seq_length=max_seq_length,
    num_train_epochs=3,                      # 학습할 총 에포크 수 
    per_device_train_batch_size=2,           # GPU당 배치 크기
    gradient_accumulation_steps=2,           # 그래디언트 누적 스텝 수
    gradient_checkpointing=True,             # 메모리 절약을 위한 체크포인팅
    optim="adamw_torch_fused",               # 최적화기
    logging_steps=10,                        # 로그 기록 주기
    save_strategy="steps",                   # 저장 전략
    save_steps=50,                           # 저장 주기
    bf16=True,                              # bfloat16 사용
    learning_rate=1e-4,                     # 학습률
    max_grad_norm=0.3,                      # 그래디언트 클리핑
    warmup_ratio=0.03,                      # 워밍업 비율
    lr_scheduler_type="constant",           # 고정 학습률
    push_to_hub=False,                      # 허브 업로드 안 함
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to=None
)

def collate_fn(batch):
    new_batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    
    for example in batch:
        # messages의 각 내용에서 개행문자 제거
        clean_messages = []
        for message in example["messages"]:
            clean_message = {
                "role": message["role"],
                "content": message["content"]
            }
            clean_messages.append(clean_message)
        
        # 깨끗해진 메시지로 템플릿 적용
        text = tokenizer.apply_chat_template(
            clean_messages,
            tokenize=False,
            add_generation_prompt=False
        ).strip()
        
        # 텍스트를 토큰화
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # 레이블 초기화
        labels = [-100] * len(input_ids)
        
        # assistant 응답 부분 찾기
        im_start = "<|start_header_id|>"
        im_end = "<|eot_id|>" # <|end_header_id|>
        assistant = "assistant"
        
        # 토큰 ID 가져오기
        im_start_tokens = tokenizer.encode(im_start, add_special_tokens=False)
        im_end_tokens = tokenizer.encode(im_end, add_special_tokens=False)
        assistant_tokens = tokenizer.encode(assistant, add_special_tokens=False)
        
        i = 0
        while i < len(input_ids):
            # <|start_header_id|>assistant 찾기
            if (i + len(im_start_tokens) <= len(input_ids) and 
                input_ids[i:i+len(im_start_tokens)] == im_start_tokens):
                
                # assistant 토큰 찾기
                assistant_pos = i + len(im_start_tokens)
                if (assistant_pos + len(assistant_tokens) <= len(input_ids) and 
                    input_ids[assistant_pos:assistant_pos+len(assistant_tokens)] == assistant_tokens):
                    
                    # assistant 응답의 시작 위치로 이동
                    current_pos = assistant_pos + len(assistant_tokens)
                    
                    # <|eot_id|>를 찾을 때까지 레이블 설정
                    while current_pos < len(input_ids):
                        if (current_pos + len(im_end_tokens) <= len(input_ids) and 
                            input_ids[current_pos:current_pos+len(im_end_tokens)] == im_end_tokens):
                            # <|eot_id|> 토큰도 레이블에 포함
                            for j in range(len(im_end_tokens)):
                                labels[current_pos + j] = input_ids[current_pos + j]
                            break
                        labels[current_pos] = input_ids[current_pos]
                        current_pos += 1
                    
                    i = current_pos
                
            i += 1
        
        new_batch["input_ids"].append(input_ids)
        new_batch["attention_mask"].append(attention_mask)
        new_batch["labels"].append(labels)
    
    # 패딩 적용
    max_length = max(len(ids) for ids in new_batch["input_ids"])
    
    for i in range(len(new_batch["input_ids"])):
        padding_length = max_length - len(new_batch["input_ids"][i])
        
        new_batch["input_ids"][i].extend([tokenizer.pad_token_id] * padding_length)
        new_batch["attention_mask"][i].extend([0] * padding_length)
        new_batch["labels"][i].extend([-100] * padding_length)
    
    # 텐서로 변환
    for k, v in new_batch.items():
        new_batch[k] = torch.tensor(v)
    
    return new_batch

# collate_fn 테스트 (배치 크기 1로)
example = train_dataset[0]
batch = collate_fn([example])

# print("\n처리된 배치 데이터:")
# print("입력 ID 형태:", batch["input_ids"].shape)
# print("어텐션 마스크 형태:", batch["attention_mask"].shape)
# print("레이블 형태:", batch["labels"].shape)


# print('입력에 대한 정수 인코딩 결과:')
# print(batch["input_ids"][0].tolist())

# print('레이블에 대한 정수 인코딩 결과:')
# print(batch["labels"][0].tolist())

# 학습
trainer = SFTTrainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    # max_seq_length=max_seq_length,  # 최대 시퀀스 길이 설정
    train_dataset=train_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
)

# 학습 시작
trainer.train() # 모델이 자동으로 허브와 output_dir에 저장됨

# 모델 저장
trainer.save_model()   # 최종 모델을 저장 