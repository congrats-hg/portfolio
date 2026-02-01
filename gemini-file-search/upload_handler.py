import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

import config
from file_search_manager import FileSearchManager
from json_processor import load_all_files, validate_article, get_file_count
from models import NewsArticle

logger = logging.getLogger(__name__)


class UploadHandler:
    def __init__(
        self,
        manager: FileSearchManager,
        state_file: Optional[Path] = None,
    ):
        self.manager = manager
        self.state_file = state_file or config.STORE_STATE_FILE

    def load_state(self) -> dict:
        """Load upload state from file"""
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "store_name": None,
            "uploaded_files": [],
            "failed_files": [],
            "last_updated": None,
            "total_count": 0,
            "uploaded_count": 0,
            "failed_count": 0,
        }

    def save_state(self, state: dict):
        """Save upload state to file"""
        state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def get_pending_files(
        self, all_files: list[str], state: dict
    ) -> list[str]:
        """Get list of files not yet uploaded"""
        uploaded = set(state.get("uploaded_files", []))
        failed = set(state.get("failed_files", []))
        processed = uploaded | failed
        return [f for f in all_files if f not in processed]

    def handle_rate_limit(self, retry_count: int) -> float:
        """Calculate exponential backoff delay"""
        delay = config.RATE_LIMIT_DELAY * (2**retry_count)
        return min(delay, 60)  # Cap at 60 seconds

    def upload_article(
        self,
        article: NewsArticle,
        store_name: str,
        retry_count: int = 0,
    ) -> bool:
        """Upload a single article with retry logic"""
        if not validate_article(article):
            logger.warning(f"Invalid article: {article.source_file}")
            return False

        try:
            success = self.manager.upload_file(
                store_name=store_name,
                content=article.to_searchable_text(),
                display_name=f"article-{article.art_id}",
                metadata=article.get_custom_metadata(),
            )
            return success
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str:
                if retry_count < config.MAX_RETRIES:
                    delay = self.handle_rate_limit(retry_count)
                    logger.warning(
                        f"Rate limit hit, waiting {delay}s (retry {retry_count + 1})"
                    )
                    time.sleep(delay)
                    return self.upload_article(
                        article, store_name, retry_count + 1
                    )
            logger.error(f"Upload failed: {article.source_file} - {e}")
            return False

    def upload_all(
        self,
        directory: Optional[Path] = None,
        resume: bool = True,
        limit: Optional[int] = None,
    ) -> dict:
        """Upload all files from directory with progress tracking"""
        directory = directory or config.JSON_SOURCE_DIR
        state = self.load_state() if resume else self.load_state()

        # Get or create store
        if state["store_name"] and resume:
            store_name = state["store_name"]
            logger.info(f"Resuming with existing store: {store_name}")
        else:
            store_name = self.manager.get_or_create_store(
                config.STORE_DISPLAY_NAME
            )
            state["store_name"] = store_name

        # Get file list
        total_files = get_file_count(directory)
        state["total_count"] = total_files

        # Determine pending files
        all_file_names = [f.name for f in sorted(directory.glob("*.json"))]
        pending = self.get_pending_files(all_file_names, state)

        if limit:
            pending = pending[:limit]

        logger.info(
            f"Total: {total_files}, "
            f"Uploaded: {len(state['uploaded_files'])}, "
            f"Failed: {len(state['failed_files'])}, "
            f"Pending: {len(pending)}"
        )

        if not pending:
            logger.info("No files to upload")
            return state

        # Process files
        for file_name in tqdm(pending, desc="Uploading"):
            file_path = directory / file_name

            # Load article
            from json_processor import load_json_file

            article = load_json_file(file_path)

            if article is None:
                state["failed_files"].append(file_name)
                state["failed_count"] = len(state["failed_files"])
                continue

            # Upload
            success = self.upload_article(article, store_name)

            if success:
                state["uploaded_files"].append(file_name)
                state["uploaded_count"] = len(state["uploaded_files"])
            else:
                state["failed_files"].append(file_name)
                state["failed_count"] = len(state["failed_files"])

            # Save state periodically (every 10 files)
            if (
                len(state["uploaded_files"]) + len(state["failed_files"])
            ) % 10 == 0:
                self.save_state(state)

            # Rate limit delay
            time.sleep(config.RATE_LIMIT_DELAY)

        # Final save
        self.save_state(state)
        return state

    def get_status(self) -> dict:
        """Get current upload status"""
        state = self.load_state()
        return {
            "store_name": state.get("store_name"),
            "total": state.get("total_count", 0),
            "uploaded": len(state.get("uploaded_files", [])),
            "failed": len(state.get("failed_files", [])),
            "last_updated": state.get("last_updated"),
        }
