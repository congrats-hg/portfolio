import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class LogEntry:
    datetime_str: str  # "2025-11-26 08:56:31"
    search_query: str
    session_id: str
    has_error: bool = False

    @property
    def date(self) -> datetime:
        return datetime.strptime(self.datetime_str, "%Y-%m-%d %H:%M:%S")

    @property
    def year_month(self) -> str:
        """시트 이름용: '2025-11' 형식"""
        return self.date.strftime("%Y-%m")

    @property
    def day(self) -> int:
        return self.date.day

    def unique_key(self) -> str:
        """중복 판단용 키"""
        return f"{self.datetime_str}|{self.search_query}"


class LogParser:
    # 세션 ID 패턴: session_xxxxx_xxxxx
    SESSION_PATTERN = re.compile(r'session_[a-z0-9]+_\d+')
    
    # 검색어 패턴: 💬 이모지 뒤의 텍스트
    SEARCH_QUERY_PATTERN = re.compile(
        r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{2,3}.*?💬\s*(.+)$'
    )
    
    # 오류 패턴
    ERROR_PATTERNS = [
        re.compile(r'⛔'),
        re.compile(r'질문에 대한 답변을 생성하지 못했습니다'),
    ]

    def __init__(self):
        self.entries: list[LogEntry] = []
        self._session_queries: dict[str, LogEntry] = {}  # session_id -> 마지막 LogEntry

    def parse_file(self, file_path: str | Path) -> list[LogEntry]:
        """로그 파일을 파싱하여 LogEntry 리스트 반환"""
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"파일이 존재하지 않습니다: {file_path}")
            return []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_session: Optional[str] = None

        for line in lines:
            # 세션 ID 추출
            session_match = self.SESSION_PATTERN.search(line)
            if session_match:
                current_session = session_match.group()

            # 검색어 추출 (💬 이모지)
            query_match = self.SEARCH_QUERY_PATTERN.match(line)
            if query_match and current_session:
                datetime_str = query_match.group(1)
                search_query = query_match.group(2).strip()
                
                entry = LogEntry(
                    datetime_str=datetime_str,
                    search_query=search_query,
                    session_id=current_session,
                    has_error=False
                )
                self.entries.append(entry)
                self._session_queries[current_session] = entry

            # 오류 확인
            for error_pattern in self.ERROR_PATTERNS:
                if error_pattern.search(line):
                    # 현재 세션의 마지막 검색어에 오류 표시
                    if current_session and current_session in self._session_queries:
                        self._session_queries[current_session].has_error = True
                    break

        return self.entries

    def parse_multiple_files(self, file_paths: list[str | Path]) -> list[LogEntry]:
        """여러 로그 파일을 파싱 (중복 제거)"""
        all_entries: dict[str, LogEntry] = {}
        
        for file_path in file_paths:
            self.entries = []
            self._session_queries = {}
            entries = self.parse_file(file_path)
            
            for entry in entries:
                key = entry.unique_key()
                if key not in all_entries:
                    all_entries[key] = entry
                elif entry.has_error:
                    # 이미 있는 엔트리에 오류 정보 업데이트
                    all_entries[key].has_error = True

        return list(all_entries.values())