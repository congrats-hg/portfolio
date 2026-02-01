import logging
import time
import tempfile
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

import config

logger = logging.getLogger(__name__)


class FileSearchManager:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.client = genai.Client(api_key=self.api_key)

    def create_store(self, display_name: str) -> str:
        """Create a new file search store and return its name"""
        file_search_store = self.client.file_search_stores.create(
            config={"display_name": display_name}
        )
        logger.info(f"Created file search store: {file_search_store.name}")
        return file_search_store.name

    def get_or_create_store(self, display_name: str) -> str:
        """Get existing store by display name or create new one"""
        for store in self.client.file_search_stores.list():
            if store.display_name == display_name:
                logger.info(f"Found existing store: {store.name}")
                return store.name

        return self.create_store(display_name)

    def list_stores(self) -> list:
        """List all file search stores"""
        stores = list(self.client.file_search_stores.list())
        return stores

    def delete_store(self, store_name: str, force: bool = False):
        """Delete a file search store"""
        self.client.file_search_stores.delete(
            name=store_name, config={"force": force}
        )
        logger.info(f"Deleted store: {store_name}")

    def wait_for_operation(
        self, operation, timeout: int = config.OPERATION_TIMEOUT
    ) -> bool:
        """Wait for an async operation to complete"""
        start_time = time.time()
        while not operation.done:
            if time.time() - start_time > timeout:
                logger.error("Operation timed out")
                return False
            time.sleep(5)
            operation = self.client.operations.get(operation)
        return True

    def upload_file(
        self,
        store_name: str,
        content: str,
        display_name: str,
        metadata: Optional[list[dict]] = None,
    ) -> bool:
        """Upload content to file search store"""
        try:
            # Create a temporary file with the content
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write(content)
                temp_path = f.name

            try:
                # Direct upload to file search store
                operation = self.client.file_search_stores.upload_to_file_search_store(
                    file=temp_path,
                    file_search_store_name=store_name,
                    config={"display_name": display_name},
                )

                success = self.wait_for_operation(operation)
                if success:
                    logger.debug(f"Uploaded: {display_name}")
                return success
            finally:
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Upload failed for {display_name}: {e}")
            return False

    def list_documents(self, store_name: str) -> list:
        """List all documents in a file search store"""
        documents = list(
            self.client.file_search_stores.documents.list(parent=store_name)
        )
        return documents

    def query(
        self,
        store_name: str,
        question: str,
        metadata_filter: Optional[str] = None,
    ) -> str:
        """Query the file search store"""
        file_search_config = types.FileSearch(
            file_search_store_names=[store_name],
        )
        if metadata_filter:
            file_search_config = types.FileSearch(
                file_search_store_names=[store_name],
                metadata_filter=metadata_filter,
            )

        response = self.client.models.generate_content(
            model=config.MODEL_NAME,
            contents=question,
            config=types.GenerateContentConfig(
                tools=[types.Tool(file_search=file_search_config)]
            ),
        )

        return response.text

    def get_grounding_metadata(
        self,
        store_name: str,
        question: str,
        metadata_filter: Optional[str] = None,
    ):
        """Query and return full response with grounding metadata"""
        file_search_config = types.FileSearch(
            file_search_store_names=[store_name],
        )
        if metadata_filter:
            file_search_config = types.FileSearch(
                file_search_store_names=[store_name],
                metadata_filter=metadata_filter,
            )

        response = self.client.models.generate_content(
            model=config.MODEL_NAME,
            contents=question,
            config=types.GenerateContentConfig(
                tools=[types.Tool(file_search=file_search_config)]
            ),
        )

        return response
