import os
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import HfApi

load_dotenv(find_dotenv())

api = HfApi()

USER_NAME = "congrats-hg"
MODEL_NAME = "qwen25vl-chartqa-full-ft"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(SCRIPT_DIR, MODEL_NAME, "checkpoint-1769")

api = HfApi()
api.upload_folder(
    folder_path=FOLDER_PATH,
    repo_id=f"{USER_NAME}/qwen25vl-chartqa-full-ft",
    repo_type="model",
    token=os.getenv("HF_TOKEN"),
)