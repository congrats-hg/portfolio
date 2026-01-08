```
/home/ryu5090/dev/portfolio/log_2_excel_project/
├── config.py           # 설정값 (스프레드시트 ID, 경로 등)
├── log_parser.py       # 로그 파일 파싱 로직
├── sheets_handler.py   # Google Sheets API 처리
├── main.py             # 메인 실행 스크립트
├── requirements.txt    # 의존성
└── credentials.json    # 서비스 계정 키 (수동 배치)
```

- requirements.txt
```
google-api-python-client==2.111.0
google-auth-httplib2==0.2.0
google-auth-oauthlib==1.2.0
gspread==6.0.0
gspread-formatting==1.1.2
```

- Cron 설정 (매일 자정 실행)
1. `crontab -e`
2. 스크립트 열리면 다음 추가하고 저장: `0 0 * * * /usr/bin/python3 /home/ryu5090/dev/portfolio/log_2_excel_project/main.py >> /home/ryu5090/dev/portfolio/log_2_excel_project/cron.log 2>&1`
3. 현재 등록된 cron 작업 확인: `crontab -l`
4. cron 실행 로그 확인: `tail -f /home/ryu5090/dev/portfolio/log_2_excel_project/cron.log`

- 당장 테스트하고 싶으면 main.py 실행

# 로그 파일 → Google Sheets 자동화 프로젝트

로그 파일에서 사용자 검색어를 추출하여 Google Sheets에 자동으로 기록하는 Python 프로젝트입니다.

---

## 📁 프로젝트 구조

```
/home/ryu5090/dev/portfolio/log_2_excel_project/
├── config.py           # 설정값 (스프레드시트 ID, 경로, 색상 등)
├── log_parser.py       # 로그 파일 파싱 로직
├── sheets_handler.py   # Google Sheets API 처리
├── main.py             # 메인 실행 스크립트
├── requirements.txt    # 필요한 패키지 목록
├── credentials.json    # Google 서비스 계정 키 (직접 배치)
└── README.md           # 이 문서
```

---

## 🔧 각 파일 설명

### 1. `config.py` - 설정 파일

모든 설정값을 한 곳에서 관리합니다. **스타일을 바꾸고 싶다면 이 파일을 수정하세요.**

```python
from pathlib import Path

# Google Sheets 설정
SPREADSHEET_ID = "1M2nBBBSTIylCbECM3YRSWRZ5z9Y45pTKXAjE-ScHnHE"  # 시트 URL의 /d/ 뒤 부분
CREDENTIALS_PATH = Path(__file__).parent / "credentials.json"

# 로그 파일 경로
CURRENT_LOG_PATH = "/home/seoulsc/backend/logs/prod-backend.log"
ARCHIVED_LOG_DIR = "/home/seoulsc/backend/logs"
ARCHIVED_LOG_PATTERN = "backend.log.{year}-{month:02d}"

# ⭐ 셀 색상 설정 (RGB 0~1 범위)
# 예: 빨간색 = {"red": 1.0, "green": 0.0, "blue": 0.0}
COLOR_DATETIME = {"red": 0.85, "green": 0.92, "blue": 1.0}      # 연한 파랑
COLOR_SEARCH_QUERY = {"red": 0.9, "green": 1.0, "blue": 0.9}    # 연한 초록
COLOR_ERROR = {"red": 0.91, "green": 0.45, "blue": 0.32}        # 오류 표시 (주황빨강)

# 주차 범위 정의 (1~7일, 8~14일, ...)
WEEK_RANGES = [
    (1, 7),
    (8, 14),
    (15, 21),
    (22, 28),
    (29, 31),
]
```

#### 🎨 색상 변경 방법

색상은 **RGB 값을 0~1 사이의 소수**로 표현합니다.

| 일반 RGB (0~255) | config.py 형식 (0~1) | 계산 방법 |
|------------------|----------------------|-----------|
| `rgb(255, 0, 0)` (빨강) | `{"red": 1.0, "green": 0.0, "blue": 0.0}` | 255÷255=1.0 |
| `rgb(100, 200, 150)` | `{"red": 0.39, "green": 0.78, "blue": 0.59}` | 100÷255≈0.39 |
| `#e87352` | `{"red": 0.91, "green": 0.45, "blue": 0.32}` | 232÷255≈0.91 |

**HEX → RGB 변환:**
- `#e87352` = `rgb(232, 115, 82)`
- red: 232 ÷ 255 = 0.91
- green: 115 ÷ 255 = 0.45
- blue: 82 ÷ 255 = 0.32

---

### 2. `log_parser.py` - 로그 파싱

로그 파일에서 검색어를 추출합니다.

#### 핵심 로직

```python
# 검색어 패턴: 💬 이모지 뒤의 텍스트 추출
SEARCH_QUERY_PATTERN = re.compile(
    r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{2,3}.*?💬\s*(.+)$'
)

# 오류 패턴: ⛔ 이모지 또는 특정 문구
ERROR_PATTERNS = [
    re.compile(r'⛔'),
    re.compile(r'질문에 대한 답변을 생성하지 못했습니다'),
]
```

#### 정규표현식 설명

```
^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{2,3}.*?💬\s*(.+)$

^                           → 줄의 시작
(\d{4}-\d{2}-\d{2})         → 날짜 캡처 (2025-11-26)
 (\d{2}:\d{2}:\d{2})        → 시간 캡처 (08:56:31)
,\d{2,3}                    → 밀리초 (,678 또는 ,67)
.*?                         → 중간 내용 (최소 매칭)
💬\s*                       → 💬 이모지 + 공백
(.+)                        → 검색어 캡처
$                           → 줄의 끝
```

---

### 3. `sheets_handler.py` - Google Sheets 처리

Google Sheets API를 사용하여 데이터를 업데이트합니다.

#### 주요 메서드

| 메서드 | 역할 |
|--------|------|
| `get_or_create_sheet()` | 월별 시트 가져오기/생성 |
| `_initialize_sheet_headers()` | 헤더 초기화 |
| `_append_week_entries()` | 주차별 데이터 추가 |
| `_apply_formatting()` | 셀 색상 적용 |
| `_update_week_counts()` | 주차별 검색어 수 업데이트 |

---

## 🎨 스타일 커스터마이징 가이드

### 1. 배경색 변경

`config.py`에서 색상 값을 수정합니다:

```python
# 날짜/시간 셀 - 연한 노란색으로 변경
COLOR_DATETIME = {"red": 1.0, "green": 0.95, "blue": 0.8}

# 검색어 셀 - 연한 보라색으로 변경
COLOR_SEARCH_QUERY = {"red": 0.9, "green": 0.85, "blue": 1.0}

# 오류 셀 - 진한 빨간색으로 변경
COLOR_ERROR = {"red": 1.0, "green": 0.3, "blue": 0.3}
```

### 2. 테두리 추가

`sheets_handler.py`의 `_apply_formatting()` 메서드를 수정합니다:

```python
from gspread_formatting import CellFormat, Color, Border, Borders

# 테두리 스타일 정의
thin_border = Border(style='SOLID', color=Color(0, 0, 0))  # 검은색 실선
borders = Borders(
    top=thin_border,
    bottom=thin_border,
    left=thin_border,
    right=thin_border
)

# CellFormat에 borders 추가
datetime_fmt = CellFormat(
    backgroundColor=Color(
        COLOR_DATETIME['red'],
        COLOR_DATETIME['green'],
        COLOR_DATETIME['blue']
    ),
    borders=borders  # ⭐ 테두리 추가
)
```

#### 테두리 스타일 종류

| style 값 | 설명 |
|----------|------|
| `'SOLID'` | 실선 |
| `'DASHED'` | 점선 |
| `'DOTTED'` | 도트 |
| `'SOLID_MEDIUM'` | 중간 굵기 실선 |
| `'SOLID_THICK'` | 굵은 실선 |
| `'DOUBLE'` | 이중선 |

### 3. 셀 크기(열 너비) 조정

`sheets_handler.py`의 `_initialize_sheet_headers()` 메서드에 추가:

```python
def _initialize_sheet_headers(self, worksheet: Worksheet):
    """시트 헤더 초기화"""
    # 기존 헤더 설정 코드...
    
    # ⭐ 열 너비 설정 추가
    requests = []
    for week_idx in range(len(WEEK_RANGES)):
        col_datetime = week_idx * 3  # 0-based index
        col_query = week_idx * 3 + 1
        
        # 날짜/시간 열 너비: 180픽셀
        requests.append({
            'updateDimensionProperties': {
                'range': {
                    'sheetId': worksheet.id,
                    'dimension': 'COLUMNS',
                    'startIndex': col_datetime,
                    'endIndex': col_datetime + 1
                },
                'properties': {'pixelSize': 180},
                'fields': 'pixelSize'
            }
        })
        
        # 검색어 열 너비: 300픽셀
        requests.append({
            'updateDimensionProperties': {
                'range': {
                    'sheetId': worksheet.id,
                    'dimension': 'COLUMNS',
                    'startIndex': col_query,
                    'endIndex': col_query + 1
                },
                'properties': {'pixelSize': 300},
                'fields': 'pixelSize'
            }
        })
    
    if requests:
        self.spreadsheet.batch_update({'requests': requests})
```

### 4. 행 높이 조정

```python
# 행 높이 설정 (예: 1~3행 헤더 높이 40픽셀)
requests = [{
    'updateDimensionProperties': {
        'range': {
            'sheetId': worksheet.id,
            'dimension': 'ROWS',
            'startIndex': 0,  # 0-based (1행)
            'endIndex': 3     # 3행까지
        },
        'properties': {'pixelSize': 40},
        'fields': 'pixelSize'
    }
}]
self.spreadsheet.batch_update({'requests': requests})
```

### 5. 텍스트 정렬 및 글꼴 설정

```python
from gspread_formatting import CellFormat, Color, TextFormat, HorizontalAlignment

datetime_fmt = CellFormat(
    backgroundColor=Color(
        COLOR_DATETIME['red'],
        COLOR_DATETIME['green'],
        COLOR_DATETIME['blue']
    ),
    # ⭐ 텍스트 가운데 정렬
    horizontalAlignment=HorizontalAlignment.CENTER,
    # ⭐ 글꼴 설정
    textFormat=TextFormat(
        bold=True,           # 굵게
        fontSize=10,         # 글꼴 크기
        foregroundColor=Color(0.2, 0.2, 0.2)  # 글꼴 색상 (진한 회색)
    )
)
```

### 6. 헤더 스타일 별도 적용

```python
def _style_headers(self, worksheet: Worksheet):
    """헤더 행에 특별한 스타일 적용"""
    from gspread_formatting import CellFormat, Color, TextFormat, format_cell_range
    
    header_fmt = CellFormat(
        backgroundColor=Color(0.2, 0.4, 0.6),  # 진한 파랑
        textFormat=TextFormat(
            bold=True,
            fontSize=11,
            foregroundColor=Color(1, 1, 1)  # 흰색 글씨
        ),
        horizontalAlignment='CENTER'
    )
    
    # 1~3행에 헤더 스타일 적용
    format_cell_range(worksheet, '1:3', header_fmt, self.spreadsheet)
```

---

## 🚀 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. Google 서비스 계정 설정

1. [Google Cloud Console](https://console.cloud.google.com/) 접속
2. 프로젝트 생성 또는 선택
3. **APIs & Services > Library** → "Google Sheets API" 활성화
4. **APIs & Services > Credentials** → **Create Credentials > Service Account**
5. 서비스 계정 생성 후 **Keys** 탭에서 JSON 키 다운로드
6. 다운로드한 파일을 `credentials.json`으로 이름 변경 후 프로젝트 폴더에 배치
7. 서비스 계정 이메일(예: `xxx@project.iam.gserviceaccount.com`)을 Google Sheets에 **편집자**로 공유

### 3. 수동 실행

```bash
cd /home/ryu5090/dev/portfolio/log_2_excel_project
python3 main.py
```

### 4. 자동 실행 (Cron 설정)

```bash
# crontab 편집
crontab -e

# 매일 자정에 실행 (아래 줄 추가)
0 0 * * * /usr/bin/python3 /home/ryu5090/dev/portfolio/log_2_excel_project/main.py >> /home/ryu5090/dev/portfolio/log_2_excel_project/cron.log 2>&1
```

### 5. Cron 실행 확인

```bash
# 등록된 cron 작업 확인
crontab -l

# 실행 로그 확인
tail -f /home/ryu5090/dev/portfolio/log_2_excel_project/cron.log
```

---

## 📊 시트 구조

### 월별 시트 (예: 2025-11)

| A열 (01~07) | B열 | C열 (08~14) | D열 | ... |
|-------------|-----|-------------|-----|-----|
| 주차별 검색어 수: 15 | | 주차별 검색어 수: 23 | | |
| 날짜 및 시간 | 검색어 | 날짜 및 시간 | 검색어 | |
| 2025-11-01 09:30:15 | 검색어1 | 2025-11-08 10:20:45 | 검색어5 | |
| 2025-11-01 10:15:22 | 검색어2 | ... | ... | |

- **날짜/시간 셀**: 연한 파랑 배경
- **검색어 셀**: 연한 초록 배경
- **오류 발생 검색어**: 주황빨강 배경 (`#e87352`)

---

## 🔍 문제 해결

### "처리할 로그 파일이 없습니다"

`config.py`의 `CURRENT_LOG_PATH`가 실제 파일 경로와 일치하는지 확인하세요.

### "credentials.json 오류"

- 서비스 계정 키 파일이 올바른 형식인지 확인
- 파일에 `"type": "service_account"` 가 있어야 함

### "권한 오류"

서비스 계정 이메일을 Google Sheets에 편집자로 공유했는지 확인하세요.

### API 할당량 초과

Google Sheets API는 분당 요청 수 제한이 있습니다. 대량 데이터 처리 시 `time.sleep(1)`을 추가하세요.

---

## 📚 참고 자료

- [gspread 문서](https://docs.gspread.org/)
- [gspread-formatting 문서](https://gspread-formatting.readthedocs.io/)
- [Google Sheets API 문서](https://developers.google.com/sheets/api)
- [정규표현식 테스트](https://regex101.com/)