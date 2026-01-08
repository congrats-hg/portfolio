#!/usr/bin/env python3
"""
로그 파일을 파싱하여 Google Sheets에 업데이트하는 메인 스크립트.
매일 자정에 cron으로 실행.
"""
import sys
from datetime import datetime
from pathlib import Path

from config import CURRENT_LOG_PATH, ARCHIVED_LOG_DIR, ARCHIVED_LOG_PATTERN, SAMPLE_LOG_PATTERN
from log_parser import LogParser
from sheets_handler import SheetsHandler


def get_log_files_to_process() -> list[Path]:
    """처리할 로그 파일 목록 반환"""
    files = []
    
    # 1. 테스트용 샘플 로그 파일들 (sample_*.log)
    log_dir = Path(ARCHIVED_LOG_DIR)
    sample_files = sorted(log_dir.glob(SAMPLE_LOG_PATTERN))
    files.extend(sample_files)
    
    # 2. 운영용 현재 로그 파일
    if not sample_files:  # 샘플 파일이 없을 때만 운영 파일 사용
        current_log = Path(CURRENT_LOG_PATH)
        if current_log.exists():
            files.append(current_log)
    
    # 3. 아카이브 파일 (매달 1일에만)
    now = datetime.now()
    if now.day == 1:
        if now.month == 1:
            prev_year, prev_month = now.year - 1, 12
        else:
            prev_year, prev_month = now.year, now.month - 1
        
        archive_name = ARCHIVED_LOG_PATTERN.format(year=prev_year, month=prev_month)
        archive_path = Path(ARCHIVED_LOG_DIR) / archive_name
        if archive_path.exists():
            files.append(archive_path)
    
    return files


def main():
    print(f"[{datetime.now()}] 로그 처리 시작")
    
    # 1. 처리할 로그 파일 확인
    log_files = get_log_files_to_process()
    if not log_files:
        print("처리할 로그 파일이 없습니다.")
        sys.exit(0)
    
    print(f"처리할 파일: {[str(f) for f in log_files]}")
    
    # 2. 로그 파싱
    parser = LogParser()
    entries = parser.parse_multiple_files(log_files)
    print(f"파싱된 검색어 수: {len(entries)}")
    
    if not entries:
        print("새로운 검색어가 없습니다.")
        sys.exit(0)
    
    # 3. Google Sheets 업데이트
    try:
        handler = SheetsHandler()
        handler.update_sheet(entries)
        print(f"[{datetime.now()}] Google Sheets 업데이트 완료")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()