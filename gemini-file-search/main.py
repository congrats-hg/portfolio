#!/usr/bin/env python3
"""
Gemini File Search CLI for Kyeongin News Articles

Usage:
    python main.py upload [--resume] [--limit N]
    python main.py search "검색어" [--author NAME] [--category NAME] [--date YYYY-MM]
    python main.py interactive
    python main.py status
    python main.py list-docs
    python main.py delete-store [--force]
"""

import argparse
import logging
import sys

import config
from file_search_manager import FileSearchManager
from upload_handler import UploadHandler
from query_interface import QueryInterface

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gemini_file_search.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def cmd_upload(args):
    """Handle upload command"""
    manager = FileSearchManager()
    handler = UploadHandler(manager)

    logger.info("Starting upload process...")

    state = handler.upload_all(
        resume=args.resume,
        limit=args.limit,
    )

    print(f"\n=== Upload Complete ===")
    print(f"Store: {state.get('store_name')}")
    print(f"Uploaded: {len(state.get('uploaded_files', []))}")
    print(f"Failed: {len(state.get('failed_files', []))}")
    print(f"Total: {state.get('total_count', 0)}")


def cmd_search(args):
    """Handle search command"""
    manager = FileSearchManager()
    handler = UploadHandler(manager)

    state = handler.load_state()
    store_name = state.get("store_name")

    if not store_name:
        print("Error: No file search store found. Run 'upload' first.")
        sys.exit(1)

    query_interface = QueryInterface(manager, store_name)

    result = query_interface.search_with_filter(
        args.query,
        author=args.author,
        category=args.category,
        year_month=args.date,
    )

    print(f"\n=== Search Results ===")
    print(f"Query: {args.query}")
    if args.author:
        print(f"Author filter: {args.author}")
    if args.category:
        print(f"Category filter: {args.category}")
    if args.date:
        print(f"Date filter: {args.date}")
    print()
    print(result)


def cmd_interactive(args):
    """Handle interactive search command"""
    manager = FileSearchManager()
    handler = UploadHandler(manager)

    state = handler.load_state()
    store_name = state.get("store_name")

    if not store_name:
        print("Error: No file search store found. Run 'upload' first.")
        sys.exit(1)

    query_interface = QueryInterface(manager, store_name)
    query_interface.interactive_search()


def cmd_status(args):
    """Handle status command"""
    manager = FileSearchManager()
    handler = UploadHandler(manager)

    status = handler.get_status()

    print("\n=== Upload Status ===")
    print(f"Store: {status.get('store_name', 'Not created')}")
    print(f"Total files: {status.get('total', 0)}")
    print(f"Uploaded: {status.get('uploaded', 0)}")
    print(f"Failed: {status.get('failed', 0)}")
    print(f"Remaining: {status.get('total', 0) - status.get('uploaded', 0) - status.get('failed', 0)}")
    print(f"Last updated: {status.get('last_updated', 'Never')}")


def cmd_list_docs(args):
    """Handle list-docs command"""
    manager = FileSearchManager()
    handler = UploadHandler(manager)

    state = handler.load_state()
    store_name = state.get("store_name")

    if not store_name:
        print("Error: No file search store found. Run 'upload' first.")
        sys.exit(1)

    print(f"\n=== Documents in {store_name} ===")
    documents = manager.list_documents(store_name)

    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc.display_name if hasattr(doc, 'display_name') else doc.name}")

    print(f"\nTotal: {len(documents)} documents")


def cmd_delete_store(args):
    """Handle delete-store command"""
    manager = FileSearchManager()
    handler = UploadHandler(manager)

    state = handler.load_state()
    store_name = state.get("store_name")

    if not store_name:
        print("Error: No file search store found.")
        sys.exit(1)

    if not args.force:
        confirm = input(f"Are you sure you want to delete {store_name}? (yes/no): ")
        if confirm.lower() != "yes":
            print("Cancelled.")
            return

    manager.delete_store(store_name, force=True)

    # Clear state
    handler.save_state({
        "store_name": None,
        "uploaded_files": [],
        "failed_files": [],
        "total_count": 0,
        "uploaded_count": 0,
        "failed_count": 0,
    })

    print(f"Deleted store: {store_name}")


def cmd_list_stores(args):
    """Handle list-stores command"""
    manager = FileSearchManager()

    print("\n=== File Search Stores ===")
    stores = manager.list_stores()

    for i, store in enumerate(stores, 1):
        print(f"{i}. {store.name} ({store.display_name})")

    print(f"\nTotal: {len(stores)} stores")


def main():
    parser = argparse.ArgumentParser(
        description="Gemini File Search CLI for Kyeongin News Articles"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # upload command
    upload_parser = subparsers.add_parser("upload", help="Upload files to file search store")
    upload_parser.add_argument(
        "--resume", action="store_true", help="Resume from previous state"
    )
    upload_parser.add_argument(
        "--limit", type=int, help="Limit number of files to upload"
    )
    upload_parser.set_defaults(func=cmd_upload)

    # search command
    search_parser = subparsers.add_parser("search", help="Search in file search store")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--author", help="Filter by author")
    search_parser.add_argument("--category", help="Filter by category")
    search_parser.add_argument("--date", help="Filter by year-month (YYYY-MM)")
    search_parser.set_defaults(func=cmd_search)

    # interactive command
    interactive_parser = subparsers.add_parser(
        "interactive", help="Interactive search mode"
    )
    interactive_parser.set_defaults(func=cmd_interactive)

    # status command
    status_parser = subparsers.add_parser("status", help="Show upload status")
    status_parser.set_defaults(func=cmd_status)

    # list-docs command
    list_docs_parser = subparsers.add_parser(
        "list-docs", help="List documents in store"
    )
    list_docs_parser.set_defaults(func=cmd_list_docs)

    # list-stores command
    list_stores_parser = subparsers.add_parser(
        "list-stores", help="List all file search stores"
    )
    list_stores_parser.set_defaults(func=cmd_list_stores)

    # delete-store command
    delete_parser = subparsers.add_parser(
        "delete-store", help="Delete file search store"
    )
    delete_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation"
    )
    delete_parser.set_defaults(func=cmd_delete_store)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
