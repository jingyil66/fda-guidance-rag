"""
Compare monolithic ingest_s3 pipeline vs three-stage ingest_stages pipeline.

Usage (from project root):
    venv\\Scripts\\python.exe jobs/benchmark_pipeline_compare.py
    venv\\Scripts\\python.exe jobs/benchmark_pipeline_compare.py --limit 50 --workers 4
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LEGACY_COLLECTION = "fda_guidance_bench_legacy"
STAGES_COLLECTION = "fda_guidance_bench_stages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark legacy vs three-stage ingest.")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--embed-parallelism", type=int, default=4)
    return parser.parse_args()


def _delete_collection(collection_name: str, qdrant_url: str) -> None:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)


def _cleanup_legacy(collection_name: str, qdrant_url: str) -> None:
    from backend.app.core.config import settings
    from backend.app.etl.ingest_pipeline import checkpoint_path_for

    _delete_collection(collection_name, qdrant_url)
    checkpoint = checkpoint_path_for(collection_name)
    if checkpoint.exists():
        checkpoint.unlink()
    report = settings.INGEST_STATE_DIR / f"{collection_name}_report.json"
    if report.exists():
        report.unlink()


def _cleanup_stages(collection_name: str, qdrant_url: str) -> None:
    from backend.app.etl.ingest_paths import layout_for_collection

    _delete_collection(collection_name, qdrant_url)
    layout = layout_for_collection(collection_name)
    for path in (
        layout.download_checkpoint_path,
        layout.chunk_checkpoint_path,
        layout.embed_checkpoint_path,
    ):
        if path.exists():
            path.unlink()
    if layout.base_dir.exists():
        shutil.rmtree(layout.base_dir)


def _run_legacy(
    *,
    collection_name: str,
    qdrant_url: str,
    bucket_name: str,
    limit: int,
    workers: int,
    embed_batch_size: int,
    embed_parallelism: int,
):
    from backend.app.etl.ingest_pipeline import checkpoint_path_for, run_s3_ingest

    _cleanup_legacy(collection_name, qdrant_url)
    t0 = time.perf_counter()
    checkpoint = run_s3_ingest(
        collection_name=collection_name,
        qdrant_url=qdrant_url,
        bucket_name=bucket_name,
        num_workers=workers,
        embed_batch_size=embed_batch_size,
        embed_parallelism=embed_parallelism,
        checkpoint_path=checkpoint_path_for(collection_name),
        reset_checkpoint=True,
        resume=False,
        limit=limit,
    )
    elapsed = time.perf_counter() - t0
    return elapsed, len(checkpoint.completed_keys), len(checkpoint.failed_keys)


def _run_stages(
    *,
    collection_name: str,
    qdrant_url: str,
    bucket_name: str,
    limit: int,
    workers: int,
    embed_batch_size: int,
    embed_parallelism: int,
):
    from backend.app.etl.ingest_paths import layout_for_collection
    from backend.app.etl.ingest_stages import run_all_stages, run_stage_chunk, run_stage_embed

    _cleanup_stages(collection_name, qdrant_url)
    layout = layout_for_collection(collection_name)

    total_t0 = time.perf_counter()

    chunk_t0 = time.perf_counter()
    chunk_cp = run_stage_chunk(
        layout=layout,
        bucket_name=bucket_name,
        workers=workers,
        use_pdf_cache=False,
        reset_checkpoint=True,
        resume=False,
        limit=limit,
    )
    chunk_elapsed = time.perf_counter() - chunk_t0

    embed_t0 = time.perf_counter()
    embed_cp = run_stage_embed(
        layout=layout,
        qdrant_url=qdrant_url,
        embed_batch_size=embed_batch_size,
        embed_parallelism=embed_parallelism,
        workers=1,
        reset_checkpoint=True,
        resume=False,
        limit=limit,
    )
    embed_elapsed = time.perf_counter() - embed_t0

    total_elapsed = time.perf_counter() - total_t0
    completed = len(embed_cp.completed_keys)
    failed = len(embed_cp.failed_keys)
    return {
        "total": total_elapsed,
        "chunk": chunk_elapsed,
        "embed": embed_elapsed,
        "completed": completed,
        "failed": failed,
        "chunk_completed": len(chunk_cp.completed_keys),
    }


def main() -> int:
    args = parse_args()

    from backend.app.core.config import settings

    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""
    settings.validate()

    qdrant_url = settings.DEFAULT_QDRANT_URL
    bucket_name = settings.BUCKET_NAME

    print(
        f"Pipeline benchmark: limit={args.limit}, workers={args.workers}, "
        f"embed_batch={args.embed_batch_size}, embed_parallel={args.embed_parallelism}"
    )
    print("=" * 72)

    print("\n[1/2] Legacy pipeline (ingest_s3 / run_s3_ingest)")
    legacy_elapsed, legacy_ok, legacy_fail = _run_legacy(
        collection_name=LEGACY_COLLECTION,
        qdrant_url=qdrant_url,
        bucket_name=bucket_name,
        limit=args.limit,
        workers=args.workers,
        embed_batch_size=args.embed_batch_size,
        embed_parallelism=args.embed_parallelism,
    )
    print(
        f"Legacy total: {legacy_elapsed:7.1f}s  "
        f"completed={legacy_ok}  failed={legacy_fail}"
    )

    print("\n[2/2] Three-stage pipeline (chunk jsonl → embed)")
    stages = _run_stages(
        collection_name=STAGES_COLLECTION,
        qdrant_url=qdrant_url,
        bucket_name=bucket_name,
        limit=args.limit,
        workers=args.workers,
        embed_batch_size=args.embed_batch_size,
        embed_parallelism=args.embed_parallelism,
    )
    print(
        f"Stages chunk:  {stages['chunk']:7.1f}s  "
        f"completed={stages['chunk_completed']}"
    )
    print(
        f"Stages embed:  {stages['embed']:7.1f}s  "
        f"completed={stages['completed']}  failed={stages['failed']}"
    )
    print(
        f"Stages total:  {stages['total']:7.1f}s  "
        f"completed={stages['completed']}  failed={stages['failed']}"
    )

    print("\n" + "=" * 72)
    delta = stages["total"] - legacy_elapsed
    delta_pct = (delta / legacy_elapsed * 100) if legacy_elapsed else 0.0
    sign = "+" if delta_pct >= 0 else ""
    faster = "legacy" if legacy_elapsed < stages["total"] else "stages"
    print(f"Legacy:  {legacy_elapsed:.1f}s")
    print(f"Stages:  {stages['total']:.1f}s  (chunk {stages['chunk']:.1f}s + embed {stages['embed']:.1f}s)")
    print(f"Delta:   {sign}{delta:.1f}s ({sign}{delta_pct:.1f}%) — {faster} is faster")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
