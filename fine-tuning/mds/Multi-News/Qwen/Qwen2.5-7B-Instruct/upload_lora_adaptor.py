# # merge

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel

# base_model_path = "kakaocorp/kanana-1.5-2.1b-instruct-2505"
# adapter_path = "/home/seoulsc/backend/data/prompting/briefing/summary/ft_kanana/checkpoint-1875"
# merged_model_path = "/home/seoulsc/backend/data/model"

# device_arg = {"device_map": "auto"}

# # 베이스 모델 로드
# print(f"Loading base model from: {base_model_path}")
# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_path,
#     return_dict=True,
#     dtype=torch.float16,
#     **device_arg
# )

# # LoRA 어댑터 로드 및 병합
# print(f"Loading and merging PEFT from: {adapter_path}")
# model = PeftModel.from_pretrained(base_model, adapter_path, **device_arg)
# model = model.merge_and_unload()
# # merged_model.save_pretrained('MERGED_model') # 토크나이저도 같이
# # MERGED_model

# # 토크나이저 로드
# tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# # 저장
# print(f"Saving merged model to: {merged_model_path}")
# model.save_pretrained(merged_model_path)
# tokenizer.save_pretrained(merged_model_path)
# print("✅ 모델과 토크나이저 저장 완료")

# 허깅페이스에 모델 올리기
from huggingface_hub import HfApi

api = HfApi()

username = "congrats-hg"
MODEL_NAME = "kanana-summarizer-2-1850"
api.create_repo(
    token="",
    repo_id=f"{username}/{MODEL_NAME}",
    repo_type="model"
)

api.upload_folder(
    token="",
    repo_id=f"{username}/{MODEL_NAME}",
    folder_path="/home/seoulsc/backend/_backup/fine-tune/prompting/ft/briefing/summary_ft_model/checkpoint-1850"
)