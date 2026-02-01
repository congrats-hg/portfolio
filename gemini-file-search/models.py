from dataclasses import dataclass
from typing import Optional


@dataclass
class NewsArticle:
    nid: str
    art_id: str
    title: str
    content: str
    author: str
    category: str
    service_daytime: str
    pcUrl: str
    source_file: str
    subTitle: Optional[str] = ""
    keyword: Optional[str] = ""

    @property
    def date(self) -> str:
        """Extract date part: YYYY-MM-DD"""
        if self.service_daytime:
            return self.service_daytime.split(" ")[0]
        return ""

    @property
    def year_month(self) -> str:
        """Extract year-month: YYYY-MM"""
        date = self.date
        if date and len(date) >= 7:
            return date[:7]
        return ""

    def to_searchable_text(self) -> str:
        """Format article for optimal search"""
        return f"""---
제목: {self.title}
저자: {self.author}
카테고리: {self.category}
날짜: {self.service_daytime}
URL: {self.pcUrl}
기사ID: {self.art_id}
---

{self.content}
"""

    def get_custom_metadata(self) -> list[dict]:
        """Return Gemini-compatible metadata list"""
        metadata = []

        if self.author:
            metadata.append({"key": "author", "string_value": self.author})
        if self.category:
            metadata.append({"key": "category", "string_value": self.category})
        if self.year_month:
            metadata.append({"key": "year_month", "string_value": self.year_month})
        if self.art_id:
            metadata.append({"key": "art_id", "string_value": self.art_id})

        return metadata
