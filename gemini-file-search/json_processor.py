import json
import logging
from pathlib import Path
from typing import Generator, Optional

from models import NewsArticle

logger = logging.getLogger(__name__)


def load_json_file(file_path: Path) -> Optional[NewsArticle]:
    """Load a single JSON file and return NewsArticle object"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        item = data.get("item", {})

        return NewsArticle(
            nid=item.get("nid", ""),
            art_id=item.get("art_id", ""),
            title=item.get("title", ""),
            content=item.get("content", ""),
            author=item.get("author", ""),
            category=item.get("category", ""),
            service_daytime=item.get("service_daytime", ""),
            pcUrl=item.get("pcUrl", ""),
            subTitle=item.get("subTitle", ""),
            keyword=item.get("keyword", ""),
            source_file=file_path.name,
        )
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def load_all_files(directory: Path) -> Generator[NewsArticle, None, None]:
    """Load all JSON files from directory as a generator"""
    json_files = sorted(directory.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {directory}")

    for file_path in json_files:
        article = load_json_file(file_path)
        if article:
            yield article


def validate_article(article: NewsArticle) -> bool:
    """Validate that article has required fields"""
    if not article.title:
        return False
    if not article.content:
        return False
    if not article.art_id:
        return False
    return True


def get_file_count(directory: Path) -> int:
    """Get total count of JSON files in directory"""
    return len(list(directory.glob("*.json")))
