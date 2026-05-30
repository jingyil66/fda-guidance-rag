"""
Manual smoke test for retrieval_service.retrieve_embedding.

Usage (from project root):
    python experiment/scripts/test_retrieval.py
    python experiment/scripts/test_retrieval.py --collection test --top-k 5
    python experiment/scripts/test_retrieval.py --query "Your question here"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_QUERY = (
    "How should sponsors integrate feedback from the FDA on HF study protocols "
    "into their development timelines, and what specific documentation is required "
    "for submission in their NDA, BLA, or ANDA applications?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test embedding retrieval against Qdrant.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Query string")
    parser.add_argument("--collection", default="test", help="Qdrant collection name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of passages to retrieve")
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=200,
        help="Characters of chunk text to print per result",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from backend.app.core.config import settings
    from backend.app.services.retrieval_service import retrieve_embedding

    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""
    settings.validate()

    passages = retrieve_embedding(args.query, args.collection, top_k=args.top_k)

    print(f"collection: {args.collection}")
    print(f"top_k: {args.top_k}")
    print(f"count: {len(passages)}")
    print()

    if not passages:
        print("No passages returned. Check Qdrant is running and the collection has data.")
        return 0

    for passage in passages:
        metadata = passage.get("metadata") or {}
        text = passage.get("text") or ""
        preview = text[: args.preview_chars]
        if len(text) > args.preview_chars:
            preview += "..."

        print("---")
        print(f"id: {passage.get('id')}")
        print(f"title: {metadata.get('title', 'N/A')}")
        print(f"page: {metadata.get('page', 'N/A')}")
        print(f"pdf_id: {metadata.get('pdf_id', 'N/A')}")
        print(f"text: {preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
