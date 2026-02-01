import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API")

# Paths
BASE_DIR = Path(__file__).parent
JSON_SOURCE_DIR = BASE_DIR / "kyeongin_recv"
STORE_STATE_FILE = BASE_DIR / "store_state.json"

# Upload Configuration
BATCH_SIZE = 50
RATE_LIMIT_DELAY = 2.0  # seconds between API calls
MAX_RETRIES = 5
OPERATION_TIMEOUT = 300  # seconds

# Gemini Configuration
MODEL_NAME = "gemini-2.5-flash"
STORE_DISPLAY_NAME = "kyeongin-news-store"
