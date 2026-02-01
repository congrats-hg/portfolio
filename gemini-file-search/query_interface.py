import logging
from typing import Optional

from file_search_manager import FileSearchManager

logger = logging.getLogger(__name__)


class QueryInterface:
    def __init__(self, manager: FileSearchManager, store_name: str):
        self.manager = manager
        self.store_name = store_name

    def search(self, query: str) -> str:
        """Basic search without filters"""
        return self.manager.query(self.store_name, query)

    def search_by_author(self, query: str, author: str) -> str:
        """Search with author filter"""
        metadata_filter = f"author={author}"
        return self.manager.query(self.store_name, query, metadata_filter)

    def search_by_category(self, query: str, category: str) -> str:
        """Search with category filter"""
        metadata_filter = f"category={category}"
        return self.manager.query(self.store_name, query, metadata_filter)

    def search_by_year_month(self, query: str, year_month: str) -> str:
        """Search with year_month filter (format: YYYY-MM)"""
        metadata_filter = f"year_month={year_month}"
        return self.manager.query(self.store_name, query, metadata_filter)

    def search_with_filter(
        self,
        query: str,
        author: Optional[str] = None,
        category: Optional[str] = None,
        year_month: Optional[str] = None,
    ) -> str:
        """Search with multiple optional filters"""
        filters = []
        if author:
            filters.append(f"author={author}")
        if category:
            filters.append(f"category={category}")
        if year_month:
            filters.append(f"year_month={year_month}")

        metadata_filter = " AND ".join(filters) if filters else None
        return self.manager.query(self.store_name, query, metadata_filter)

    def search_with_sources(
        self,
        query: str,
        author: Optional[str] = None,
        category: Optional[str] = None,
        year_month: Optional[str] = None,
    ) -> dict:
        """Search and return response with source citations"""
        filters = []
        if author:
            filters.append(f"author={author}")
        if category:
            filters.append(f"category={category}")
        if year_month:
            filters.append(f"year_month={year_month}")

        metadata_filter = " AND ".join(filters) if filters else None
        response = self.manager.get_grounding_metadata(
            self.store_name, query, metadata_filter
        )

        result = {"text": response.text, "sources": []}

        # Extract grounding metadata if available
        if response.candidates and response.candidates[0].grounding_metadata:
            grounding = response.candidates[0].grounding_metadata
            result["grounding_metadata"] = grounding

        return result

    def interactive_search(self):
        """Interactive CLI search mode"""
        print("\n=== Gemini File Search Interactive Mode ===")
        print("Commands: /quit, /author <name>, /category <name>, /date <YYYY-MM>")
        print("Example: 반도체 클러스터")
        print("Example with filter: /author 이영지 전력 공급")
        print()

        current_author = None
        current_category = None
        current_year_month = None

        while True:
            try:
                user_input = input("검색> ").strip()

                if not user_input:
                    continue

                if user_input.lower() == "/quit":
                    print("종료합니다.")
                    break

                # Parse commands
                if user_input.startswith("/author "):
                    parts = user_input[8:].split(" ", 1)
                    current_author = parts[0]
                    query = parts[1] if len(parts) > 1 else ""
                    if not query:
                        print(f"저자 필터 설정: {current_author}")
                        continue
                elif user_input.startswith("/category "):
                    parts = user_input[10:].split(" ", 1)
                    current_category = parts[0]
                    query = parts[1] if len(parts) > 1 else ""
                    if not query:
                        print(f"카테고리 필터 설정: {current_category}")
                        continue
                elif user_input.startswith("/date "):
                    parts = user_input[6:].split(" ", 1)
                    current_year_month = parts[0]
                    query = parts[1] if len(parts) > 1 else ""
                    if not query:
                        print(f"날짜 필터 설정: {current_year_month}")
                        continue
                elif user_input.startswith("/clear"):
                    current_author = None
                    current_category = None
                    current_year_month = None
                    print("필터 초기화됨")
                    continue
                else:
                    query = user_input

                # Show active filters
                active_filters = []
                if current_author:
                    active_filters.append(f"저자={current_author}")
                if current_category:
                    active_filters.append(f"카테고리={current_category}")
                if current_year_month:
                    active_filters.append(f"날짜={current_year_month}")
                if active_filters:
                    print(f"[필터: {', '.join(active_filters)}]")

                # Execute search
                print("\n검색 중...")
                result = self.search_with_filter(
                    query,
                    author=current_author,
                    category=current_category,
                    year_month=current_year_month,
                )
                print(f"\n{result}\n")
                print("-" * 50)

            except KeyboardInterrupt:
                print("\n종료합니다.")
                break
            except Exception as e:
                print(f"오류 발생: {e}")
