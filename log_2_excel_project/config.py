from pathlib import Path

# Google Sheets 설정
SPREADSHEET_ID = "1M2nBBBSTIylCbECM3YRSWRZ5z9Y45pTKXAjE-ScHnHE"
CREDENTIALS_PATH = Path(__file__).parent / "credentials.json"

# 로그 파일 경로
CURRENT_LOG_PATH = "/home/seoulsc/backend/logs/prod-backend.log"  # 운영용
ARCHIVED_LOG_DIR = "/home/ryu5090/dev/portfolio/log_2_excel_project"  # 테스트용
ARCHIVED_LOG_PATTERN = "backend.log.{year}-{month:02d}"

# 테스트용 샘플 로그 패턴 추가
SAMPLE_LOG_PATTERN = "sample_*.log"  # 추가

# 셀 색상 설정 (RGB 0-1 범위)
COLOR_DATETIME = {"red": 0.85, "green": 0.92, "blue": 1.0}      # 연한 파랑
COLOR_SEARCH_QUERY = {"red": 0.9, "green": 1.0, "blue": 0.9}    # 연한 초록
COLOR_ERROR = {"red": 0.91, "green": 0.45, "blue": 0.32}        # #e87352

# 주차 범위 정의
WEEK_RANGES = [
    (1, 7),
    (8, 14),
    (15, 21),
    (22, 28),
    (29, 31),
]