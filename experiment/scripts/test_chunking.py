"""
Manual smoke test for chunking_service.chunk_fixed.

Usage (from project root):
    python experiment/scripts/test_chunking.py
    python experiment/scripts/test_chunking.py --pdf data/122971.pdf
    python experiment/scripts/test_chunking.py --chunk-size 600 --chunk-overlap 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SYNTHETIC_TEXT = "word " * 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test fixed-size document chunking.")
    parser.add_argument(
        "--pdf",
        type=Path,
        help="Optional PDF path; defaults to synthetic long text",
    )
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=120,
        help="Characters of chunk text to print for the first chunk",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from langchain_core.documents import Document

    from backend.app.services.chunking_service import (
        chunk_fixed,
        chunks_to_records,
        load_pdf_pages,
    )

    if args.pdf:
        if not args.pdf.exists():
            print(f"PDF not found: {args.pdf}")
            return 1
        source_docs = load_pdf_pages(args.pdf)
        source_label = str(args.pdf)
    else:
        source_docs = [
            Document(
                page_content=SYNTHETIC_TEXT,
                metadata={"title": "synthetic", "pdf_id": "synthetic", "page": 1},
            )
        ]
        source_label = "synthetic text"

    chunks = chunk_fixed(
        source_docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    records = chunks_to_records(chunks)
    lengths = [len(record["text"]) for record in records]

    print(f"source: {source_label}")
    print(f"source_docs: {len(source_docs)}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"chunk_overlap: {args.chunk_overlap}")
    print(f"chunk_count: {len(chunks)}")
    if lengths:
        print(f"avg_chunk_len: {mean(lengths):.1f}")
        print(f"max_chunk_len: {max(lengths)}")
    print()

    if not chunks:
        print("No chunks produced.")
        return 0

    first = records[0]
    metadata = first.get("metadata") or {}
    preview = first["text"][: args.preview_chars]
    if len(first["text"]) > args.preview_chars:
        preview += "..."

    print("--- first chunk ---")
    print(f"id: {first.get('id')}")
    print(f"title: {metadata.get('title', 'N/A')}")
    print(f"page: {metadata.get('page', 'N/A')}")
    print(f"pdf_id: {metadata.get('pdf_id', 'N/A')}")
    print(f"text_len: {len(first['text'])}")
    print(f"text: {preview}")

    if len(source_docs) == 1 and len(chunks) <= 1:
        print()
        print("Warning: only one chunk produced; check input length or chunk_size.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
