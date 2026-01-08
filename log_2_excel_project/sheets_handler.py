import gspread
from google.oauth2.service_account import Credentials
from gspread.worksheet import Worksheet
from gspread_formatting import (
    CellFormat, Color, format_cell_range
)

from config import (
    SPREADSHEET_ID, CREDENTIALS_PATH,
    COLOR_DATETIME, COLOR_SEARCH_QUERY, COLOR_ERROR, WEEK_RANGES
)
from log_parser import LogEntry


class SheetsHandler:
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    def __init__(self):
        creds = Credentials.from_service_account_file(
            CREDENTIALS_PATH, scopes=self.SCOPES
        )
        self.client = gspread.authorize(creds)
        self.spreadsheet = self.client.open_by_key(SPREADSHEET_ID)
        self._existing_keys: set[str] = set()

    def get_or_create_sheet(self, sheet_name: str) -> Worksheet:
        """시트 가져오기 또는 생성"""
        try:
            worksheet = self.spreadsheet.worksheet(sheet_name)
        except gspread.WorksheetNotFound:
            worksheet = self.spreadsheet.add_worksheet(
                title=sheet_name, rows=1000, cols=20
            )
            self._initialize_sheet_headers(worksheet)
        return worksheet

    def _get_or_create_summary_sheet(self) -> Worksheet:
        """'종합' 시트 가져오기 또는 생성"""
        try:
            worksheet = self.spreadsheet.worksheet("종합")
        except gspread.WorksheetNotFound:
            worksheet = self.spreadsheet.add_worksheet(
                title="종합", rows=100, cols=20
            )
            # 종합 시트를 맨 앞으로 이동
            self.spreadsheet.reorder_worksheets([worksheet] + [
                ws for ws in self.spreadsheet.worksheets() if ws.title != "종합"
            ])
        return worksheet

    def _initialize_sheet_headers(self, worksheet: Worksheet):
        """시트 헤더 초기화"""
        headers = []
        for i, (start, end) in enumerate(WEEK_RANGES):
            col_idx = i * 3
            headers.extend([
                (1, col_idx + 1, f"{start:02d}~{end:02d}"),
                (2, col_idx + 1, "주차별 검색어 수"),
                (3, col_idx + 1, "날짜 및 시간"),
                (3, col_idx + 2, "검색어"),
                (3, col_idx + 3, ""),  # 빈 구분 열
            ])
        
        # 배치 업데이트
        cells = []
        for row, col, value in headers:
            cells.append(gspread.Cell(row, col, value))
        
        if cells:
            worksheet.update_cells(cells)

    def _get_week_index(self, day: int) -> int:
        """날짜로 주차 인덱스 반환 (0-based)"""
        for i, (start, end) in enumerate(WEEK_RANGES):
            if start <= day <= end:
                return i
        return 0

    def _load_existing_keys(self, worksheet: Worksheet, year_month: str):
        """기존 데이터의 키 로드 (중복 방지용)"""
        self._existing_keys.clear()
        
        try:
            all_values = worksheet.get_all_values()
        except Exception:
            return

        for row_idx, row in enumerate(all_values):
            if row_idx < 3:  # 헤더 스킵
                continue
            for week_idx in range(len(WEEK_RANGES)):
                col_idx = week_idx * 3
                if col_idx + 1 < len(row):
                    datetime_str = row[col_idx] if col_idx < len(row) else ""
                    query = row[col_idx + 1] if col_idx + 1 < len(row) else ""
                    if datetime_str and query:
                        key = f"{datetime_str}|{query}"
                        self._existing_keys.add(key)

    def update_sheet(self, entries: list[LogEntry]):
        """엔트리들을 해당 시트에 업데이트"""
        # 월별로 그룹화
        by_month: dict[str, list[LogEntry]] = {}
        for entry in entries:
            year_month = entry.year_month
            if year_month not in by_month:
                by_month[year_month] = []
            by_month[year_month].append(entry)

        for year_month, month_entries in by_month.items():
            self._update_month_sheet(year_month, month_entries)
        
        # 종합 시트 업데이트
        self._update_summary_sheet()

    def _update_month_sheet(self, year_month: str, entries: list[LogEntry]):
        """월별 시트 업데이트"""
        worksheet = self.get_or_create_sheet(year_month)
        self._load_existing_keys(worksheet, year_month)

        # 주차별로 그룹화
        by_week: dict[int, list[LogEntry]] = {i: [] for i in range(len(WEEK_RANGES))}
        for entry in entries:
            if entry.unique_key() in self._existing_keys:
                continue  # 중복 스킵
            week_idx = self._get_week_index(entry.day)
            by_week[week_idx].append(entry)

        # 각 주차별로 데이터 추가
        for week_idx, week_entries in by_week.items():
            if not week_entries:
                continue
            self._append_week_entries(worksheet, week_idx, week_entries)

        # 주차별 검색어 수 업데이트
        self._update_week_counts(worksheet)

    def _append_week_entries(
        self, worksheet: Worksheet, week_idx: int, entries: list[LogEntry]
    ):
        """주차 열에 엔트리 추가"""
        col_datetime = week_idx * 3 + 1  # 1-based
        col_query = week_idx * 3 + 2

        # 현재 해당 열의 마지막 행 찾기
        try:
            col_values = worksheet.col_values(col_datetime)
            next_row = len(col_values) + 1
            if next_row <= 3:
                next_row = 4  # 헤더 다음 행부터 시작
        except Exception:
            next_row = 4

        # 데이터 및 포맷 준비
        cells = []
        format_requests = []

        for i, entry in enumerate(entries):
            row = next_row + i
            cells.append(gspread.Cell(row, col_datetime, entry.datetime_str))
            cells.append(gspread.Cell(row, col_query, entry.search_query))

            # 포맷 정보 저장
            format_requests.append({
                'row': row,
                'col_datetime': col_datetime,
                'col_query': col_query,
                'has_error': entry.has_error
            })

        # 데이터 업데이트
        if cells:
            worksheet.update_cells(cells)

        # 색상 포맷 적용
        self._apply_formatting(worksheet, format_requests)

    def _apply_formatting(self, worksheet: Worksheet, format_requests: list[dict]):
        """셀 색상 포맷 적용"""
        if not format_requests:
            return

        try:
            from gspread_formatting import CellFormat, Color, format_cell_range
            
            for req in format_requests:
                row = req['row']
                
                # 날짜/시간 셀 색상
                datetime_fmt = CellFormat(
                    backgroundColor=Color(
                        COLOR_DATETIME['red'],
                        COLOR_DATETIME['green'],
                        COLOR_DATETIME['blue']
                    )
                )
                datetime_range = f"{gspread.utils.rowcol_to_a1(row, req['col_datetime'])}"
                format_cell_range(worksheet, datetime_range, datetime_fmt, self.spreadsheet)
                
                # 검색어 셀 색상
                if req['has_error']:
                    query_color = COLOR_ERROR
                else:
                    query_color = COLOR_SEARCH_QUERY
                
                query_fmt = CellFormat(
                    backgroundColor=Color(
                        query_color['red'],
                        query_color['green'],
                        query_color['blue']
                    )
                )
                query_range = f"{gspread.utils.rowcol_to_a1(row, req['col_query'])}"
                format_cell_range(worksheet, query_range, query_fmt, self.spreadsheet)
                    
        except ImportError:
            print("gspread_formatting 패키지가 필요합니다. pip install gspread-formatting")
        except Exception as e:
            print(f"셀 포맷 적용 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

    def _update_week_counts(self, worksheet: Worksheet):
        """주차별 검색어 수 업데이트 (2행)"""
        cells = []
        
        for week_idx in range(len(WEEK_RANGES)):
            col_datetime = week_idx * 3 + 1
            
            try:
                col_values = worksheet.col_values(col_datetime)
                # 4행부터 데이터이므로 헤더 3행 제외
                count = len([v for v in col_values[3:] if v.strip()])
            except Exception:
                count = 0
            
            cells.append(gspread.Cell(2, col_datetime, count))  # 숫자만 저장
        
        if cells:
            worksheet.update_cells(cells)

    def _update_summary_sheet(self):
        """종합 시트 업데이트 - 월별 검색어 수 테이블 + 차트"""
        summary_sheet = self._get_or_create_summary_sheet()
        
        # 모든 월별 시트에서 데이터 수집
        month_data = self._collect_month_data()
        
        if not month_data:
            print("월별 데이터가 없습니다.")
            return
        
        # 종합 시트에 데이터 테이블 작성
        self._write_summary_table(summary_sheet, month_data)
        
        # 차트 업데이트
        self._update_summary_chart(summary_sheet, month_data)

    def _collect_month_data(self) -> dict[str, list[int]]:
        """모든 월별 시트에서 주차별 검색어 수 수집"""
        month_data = {}
        
        for worksheet in self.spreadsheet.worksheets():
            # YYYY-MM 형식의 시트만 처리
            title = worksheet.title
            if len(title) == 7 and title[4] == '-':
                try:
                    # 2행에서 주차별 검색어 수 읽기
                    row2 = worksheet.row_values(2)
                    week_counts = []
                    
                    for week_idx in range(len(WEEK_RANGES)):
                        col_idx = week_idx * 3  # 0-based
                        if col_idx < len(row2):
                            val = row2[col_idx]
                            # 숫자 추출 (숫자만 또는 "주차별 검색어 수: 15" 형식)
                            if isinstance(val, (int, float)):
                                week_counts.append(int(val))
                            elif val.isdigit():
                                week_counts.append(int(val))
                            elif ":" in str(val):
                                try:
                                    num = int(str(val).split(":")[-1].strip())
                                    week_counts.append(num)
                                except:
                                    week_counts.append(0)
                            else:
                                week_counts.append(0)
                        else:
                            week_counts.append(0)
                    
                    month_data[title] = week_counts
                except Exception as e:
                    print(f"시트 {title} 데이터 수집 오류: {e}")
        
        # 날짜순 정렬
        return dict(sorted(month_data.items()))

    def _write_summary_table(self, summary_sheet: Worksheet, month_data: dict[str, list[int]]):
        """종합 시트에 월별 검색어 수 테이블 작성"""
        # 헤더 작성
        headers = ["월"] + [f"{s:02d}~{e:02d}" for s, e in WEEK_RANGES] + ["월 합계"]
        
        cells = []
        # 헤더 (1행)
        for col, header in enumerate(headers, 1):
            cells.append(gspread.Cell(1, col, header))
        
        # 데이터 (2행부터)
        row = 2
        for month, week_counts in month_data.items():
            cells.append(gspread.Cell(row, 1, month))
            for col, count in enumerate(week_counts, 2):
                cells.append(gspread.Cell(row, col, count))
            # 월 합계
            cells.append(gspread.Cell(row, len(headers), sum(week_counts)))
            row += 1
        
        # 삭제: 전체 합계 행 관련 코드 전부 제거
        
        if cells:
            summary_sheet.update_cells(cells)
        
        # 헤더 스타일 적용
        self._style_summary_headers(summary_sheet, len(headers), len(month_data) + 1)  # +2 → +1로 변경

    def _style_summary_headers(self, worksheet: Worksheet, num_cols: int, num_rows: int):
        """종합 시트 헤더 스타일 적용"""
        try:
            from gspread_formatting import CellFormat, Color, TextFormat, format_cell_range
            
            # 헤더 스타일 (1행)
            header_fmt = CellFormat(
                backgroundColor=Color(0.2, 0.4, 0.7),
                textFormat=TextFormat(bold=True, foregroundColor=Color(1, 1, 1)),
                horizontalAlignment='CENTER'
            )
            header_range = f"A1:{gspread.utils.rowcol_to_a1(1, num_cols)}"
            format_cell_range(worksheet, header_range, header_fmt, self.spreadsheet)
            
            # 월 컬럼 스타일 (A열)
            month_fmt = CellFormat(
                backgroundColor=Color(0.9, 0.9, 0.95),
                textFormat=TextFormat(bold=True),
                horizontalAlignment='CENTER'
            )
            month_range = f"A2:A{num_rows}"
            format_cell_range(worksheet, month_range, month_fmt, self.spreadsheet)
            
            # 합계 행 스타일
            total_fmt = CellFormat(
                backgroundColor=Color(0.95, 0.95, 0.8),
                textFormat=TextFormat(bold=True),
                horizontalAlignment='CENTER'
            )
            total_range = f"A{num_rows}:{gspread.utils.rowcol_to_a1(num_rows, num_cols)}"
            format_cell_range(worksheet, total_range, total_fmt, self.spreadsheet)
            
            # 데이터 영역 스타일
            data_fmt = CellFormat(
                horizontalAlignment='CENTER'
            )
            data_range = f"B2:{gspread.utils.rowcol_to_a1(num_rows - 1, num_cols)}"
            format_cell_range(worksheet, data_range, data_fmt, self.spreadsheet)
            
        except Exception as e:
            print(f"종합 시트 스타일 적용 오류: {e}")

    def _update_summary_chart(self, summary_sheet: Worksheet, month_data: dict[str, list[int]]):
        """종합 시트에 차트 생성/업데이트 - 월별 총합 꺾은선 그래프"""
        if not month_data:
            return
        
        num_months = len(month_data)
        num_cols = len(WEEK_RANGES) + 2  # 월 + 주차들 + 합계
        
        # 기존 차트 삭제
        try:
            sheet_id = summary_sheet.id
            existing_charts = self.spreadsheet.fetch_sheet_metadata().get('sheets', [])
            for sheet in existing_charts:
                if sheet.get('properties', {}).get('sheetId') == sheet_id:
                    charts = sheet.get('charts', [])
                    for chart in charts:
                        chart_id = chart.get('chartId')
                        if chart_id:
                            self.spreadsheet.batch_update({
                                'requests': [{
                                    'deleteEmbeddedObject': {
                                        'objectId': chart_id
                                    }
                                }]
                            })
        except Exception as e:
            print(f"기존 차트 삭제 중 오류 (무시): {e}")
        
        # 새 차트 생성
        sheet_id = summary_sheet.id
        
        # 월 합계 열 인덱스 (G열 = 6, 0-based)
        total_col_index = len(WEEK_RANGES) + 1  # 월 + 주차들 다음 = 월 합계 열
        
        # 차트 요청 생성 - 월별 총합 꺾은선 그래프
        chart_request = {
            'addChart': {
                'chart': {
                    'spec': {
                        'title': '월별 검색어 수',
                        'basicChart': {
                            'chartType': 'LINE',
                            'legendPosition': 'BOTTOM_LEGEND',
                            'axis': [
                                {
                                    'position': 'BOTTOM_AXIS',
                                    'title': '월'
                                },
                                {
                                    'position': 'LEFT_AXIS',
                                    'title': '검색어 수'
                                }
                            ],
                            'domains': [
                                {
                                    'domain': {
                                        'sourceRange': {
                                            'sources': [{
                                                'sheetId': sheet_id,
                                                'startRowIndex': 0,
                                                'endRowIndex': num_months + 1,  # 헤더 + 데이터 (합계 행 제외)
                                                'startColumnIndex': 0,
                                                'endColumnIndex': 1  # A열 (월)
                                            }]
                                        }
                                    }
                                }
                            ],
                            'series': [
                                {
                                    'series': {
                                        'sourceRange': {
                                            'sources': [{
                                                'sheetId': sheet_id,
                                                'startRowIndex': 0,
                                                'endRowIndex': num_months + 1,  # 헤더 + 데이터
                                                'startColumnIndex': total_col_index,
                                                'endColumnIndex': total_col_index + 1  # 월 합계 열
                                            }]
                                        }
                                    },
                                    'targetAxis': 'LEFT_AXIS',
                                    'dataLabel': {
                                        'type': 'DATA',
                                        'placement': 'ABOVE'
                                    }
                                }
                            ],
                            'headerCount': 1,
                            'lineSmoothing': False  # 직선 연결
                        }
                    },
                    'position': {
                        'overlayPosition': {
                            'anchorCell': {
                                'sheetId': sheet_id,
                                'rowIndex': num_months + 4,
                                'columnIndex': 0
                            },
                            'offsetXPixels': 0,
                            'offsetYPixels': 0,
                            'widthPixels': 600,
                            'heightPixels': 350
                        }
                    }
                }
            }
        }
        
        try:
            self.spreadsheet.batch_update({'requests': [chart_request]})
            print("종합 시트 차트 생성 완료")
        except Exception as e:
            print(f"차트 생성 오류: {e}")
            import traceback
            traceback.print_exc()