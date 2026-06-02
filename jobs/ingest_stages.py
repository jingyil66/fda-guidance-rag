"""
Three-stage S3 ingest: download (optional) → chunk → jsonl → embed → Qdrant.

Usage (from project root):
    venv\\Scripts\\python.exe jobs/ingest_stages.py --stage all
    venv\\Scripts\\python.exe jobs/ingest_stages.py --stage chunk --limit 5
    venv\\Scripts\\python.exe jobs/ingest_stages.py --stage embed --embed-parallelism 4
    venv\\Scripts\\python.exe jobs/ingest_stages.py --stage download --cache-pdfs
    venv\\Scripts\\python.exe jobs/ingest_stages.py --stage all --cache-pdfs --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    from backend.app.core.config import settings

    parser = argparse.ArgumentParser(
        description="Three-stage ingest: download → chunk (jsonl) → embed (Qdrant)."
    )
    parser.add_argument(
        "--stage",
        choices=["download", "chunk", "embed", "all"],
        default="all",
        help="Run one stage or all stages (default: all)",
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
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--chunk-workers", type=int, default=4)
    parser.add_argument("--embed-workers", type=int, default=1)
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument(
        "--embed-parallelism",
        type=int,
        default=4,
        help="Concurrent embed batches per PDF (default 4)",
    )
    parser.add_argument(
        "--cache-pdfs",
        action="store_true",
        help="Run download stage and read PDFs from local cache during chunk",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max PDFs per stage (0=all pending)")
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Clear checkpoint for the selected stage(s)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only re-process keys marked failed in checkpoint",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore completed keys for the selected stage(s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from backend.app.core.config import settings
    from backend.app.etl.ingest_paths import layout_for_collection
    from backend.app.etl.ingest_stages import (
        run_all_stages,
        run_stage_chunk,
        run_stage_download,
        run_stage_embed,
    )

    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""
    settings.validate()

    layout = layout_for_collection(args.collection)
    resume = not args.no_resume
    common = {
        "layout": layout,
        "bucket_name": args.bucket,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "resume": resume,
        "retry_failed": args.retry_failed,
        "reset_checkpoint": args.reset_checkpoint,
        "limit": args.limit,
    }

    print(f"Collection: {args.collection}")
    print(f"Artifacts: {layout.base_dir}")
    print(f"  pdf cache: {layout.pdf_cache_dir}")
    print(f"  chunks:    {layout.chunks_dir}")

    if args.stage == "download":
        checkpoint = run_stage_download(
            workers=args.download_workers,
            **common,
        )
        checkpoints = {"download": checkpoint}
    elif args.stage == "chunk":
        checkpoint = run_stage_chunk(
            workers=args.chunk_workers,
            use_pdf_cache=args.cache_pdfs,
            **common,
        )
        checkpoints = {"chunk": checkpoint}
    elif args.stage == "embed":
        checkpoint = run_stage_embed(
            qdrant_url=args.qdrant_url,
            embed_batch_size=args.embed_batch_size,
            embed_parallelism=args.embed_parallelism,
            workers=args.embed_workers,
            **common,
        )
        checkpoints = {"embed": checkpoint}
    else:
        checkpoints = run_all_stages(
            collection_name=args.collection,
            qdrant_url=args.qdrant_url,
            bucket_name=args.bucket,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            download_workers=args.download_workers,
            chunk_workers=args.chunk_workers,
            embed_workers=args.embed_workers,
            embed_batch_size=args.embed_batch_size,
            embed_parallelism=args.embed_parallelism,
            use_pdf_cache=args.cache_pdfs,
            skip_download=not args.cache_pdfs,
            resume=resume,
            retry_failed=args.retry_failed,
            reset_checkpoint=args.reset_checkpoint,
            limit=args.limit,
        )

    report_path = settings.INGEST_STATE_DIR / f"{args.collection}_stages_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {name: cp.to_dict() for name, cp in checkpoints.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
