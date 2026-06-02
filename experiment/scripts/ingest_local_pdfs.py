"""
Ingest local PDFs into Qdrant using chunking_service.

Usage (from project root):
    python experiment/scripts/ingest_local_pdfs.py
    python experiment/scripts/ingest_local_pdfs.py --data-dir data --collection fda_guidance_chunk600_overlap200
    python experiment/scripts/ingest_local_pdfs.py --limit 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest local PDFs into Qdrant.")
    from backend.app.core.config import settings

    parser.add_argument(
        "--data-dirs",
        nargs="*",
        type=Path,
        default=None,
        help="PDF directories to scan (default: data/ and data/subset_200/)",
    )
    parser.add_argument(
        "--collection",
        default=settings.DEFAULT_QDRANT_COLLECTION,
        help="Target Qdrant collection name",
    )
    parser.add_argument("--qdrant-url", default=settings.DEFAULT_QDRANT_URL)
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of PDFs to ingest (0 = all)",
    )
    parser.add_argument(
        "--subset-file",
        type=Path,
        default=None,
        help="Optional subset JSON; only ingest listed pdf_ids",
    )
    return parser.parse_args()


def load_subset_pdf_ids(path: Path) -> set[str]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    pdf_ids = payload.get("pdf_ids") or []
    return {str(pdf_id) for pdf_id in pdf_ids}


def collect_pdf_files(data_dirs: list[Path], subset_file: Path | None) -> list[Path]:
    by_stem: dict[str, Path] = {}
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for pdf_path in sorted(data_dir.glob("*.pdf")):
            by_stem.setdefault(pdf_path.stem, pdf_path)

    pdf_files = sorted(by_stem.values(), key=lambda path: path.stem)
    if subset_file:
        allowed_ids = load_subset_pdf_ids(subset_file)
        pdf_files = [path for path in pdf_files if path.stem in allowed_ids]
    return pdf_files


def main() -> int:
    args = parse_args()

    from langchain_core.documents import Document

    from backend.app.core.config import settings
    from backend.app.db.qdrant_client import init_qdrant
    from backend.app.services.chunking_service import chunk_fixed, load_pdf_pages

    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""
    settings.validate()

    data_dirs = args.data_dirs or [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "subset_200",
    ]
    pdf_files = collect_pdf_files(data_dirs, args.subset_file)
    if args.limit > 0:
        pdf_files = pdf_files[: args.limit]

    if not pdf_files:
        print(f"No PDFs found in {', '.join(str(path) for path in data_dirs)}")
        return 1

    vector_store = init_qdrant(args.collection, args.qdrant_url)
    buffer: list[Document] = []
    total_pages = 0
    total_chunks = 0

    for pdf_path in pdf_files:
        pages = load_pdf_pages(pdf_path)
        total_pages += len(pages)
        chunks = chunk_fixed(
            pages,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        total_chunks += len(chunks)
        buffer.extend(chunks)

        if len(buffer) >= args.batch_size:
            vector_store.add_documents(buffer)
            print(f"Uploaded batch: {len(buffer)} chunks")
            buffer = []

    if buffer:
        vector_store.add_documents(buffer)
        print(f"Uploaded final batch: {len(buffer)} chunks")

    print(f"collection: {args.collection}")
    print(f"pdfs: {len(pdf_files)}")
    print(f"pages: {total_pages}")
    print(f"chunks: {total_chunks}")
    if total_pages:
        print(f"chunks_per_page_avg: {total_chunks / total_pages:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
