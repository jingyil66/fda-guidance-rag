"""
Resumable S3 → Qdrant ingest with checkpoint and idempotent point IDs.

Usage (from project root):
    venv\\Scripts\\python.exe jobs/ingest_s3.py
    venv\\Scripts\\python.exe jobs/ingest_s3.py --limit 5
    venv\\Scripts\\python.exe jobs/ingest_s3.py --resume
    venv\\Scripts\\python.exe jobs/ingest_s3.py --retry-failed
    venv\\Scripts\\python.exe jobs/ingest_s3.py --reset-checkpoint
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    from backend.app.core.config import settings

    parser = argparse.ArgumentParser(
        description="Ingest PDFs from S3 into Qdrant (checkpoint + idempotent upsert)."
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get("QDRANT_COLLECTION", settings.DEFAULT_QDRANT_COLLECTION),
    )
    parser.add_argument(
        "--qdrant-url",
        default=settings.DEFAULT_QDRANT_URL,
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("S3_BUCKET", settings.BUCKET_NAME),
    )
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument(
        "--embed-parallelism",
        type=int,
        default=1,
        help="Concurrent embed batches per PDF in the writer (default 1)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max PDFs this run (0=all pending)")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Override checkpoint JSON path",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Clear completed/failed keys before running",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only re-process keys marked failed in checkpoint",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore completed keys (re-queue all S3 keys; upsert remains idempotent)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from backend.app.core.config import settings
    from backend.app.etl.ingest_pipeline import (
        checkpoint_path_for,
        run_s3_ingest,
        write_ingest_report,
    )

    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""
    settings.validate()

    if args.no_resume and not args.reset_checkpoint:
        print("Note: --no-resume re-processes all keys; upsert IDs prevent duplicate points.")

    checkpoint = run_s3_ingest(
        collection_name=args.collection,
        qdrant_url=args.qdrant_url,
        bucket_name=args.bucket,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        num_workers=args.workers,
        embed_batch_size=args.embed_batch_size,
        embed_parallelism=args.embed_parallelism,
        checkpoint_path=args.checkpoint_path,
        reset_checkpoint=args.reset_checkpoint,
        retry_failed=args.retry_failed,
        resume=not args.no_resume,
        limit=args.limit,
    )

    report_path = settings.INGEST_STATE_DIR / f"{args.collection}_report.json"
    write_ingest_report(checkpoint, report_path)
    print(f"Report: {report_path}")
    print(f"Checkpoint: {args.checkpoint_path or checkpoint_path_for(args.collection)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
