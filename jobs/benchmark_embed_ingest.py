"""
Compare embed batch size and parallel writer settings on the same PDF subset.

Usage (from project root):
    venv\\Scripts\\python.exe jobs/benchmark_embed_ingest.py
    venv\\Scripts\\python.exe jobs/benchmark_embed_ingest.py --limit 50 --workers 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_COLLECTION = "fda_guidance_embed_bench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ingest embed settings.")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["128,1", "256,1", "256,4"],
        help="Comma-separated embed_batch_size,embed_parallelism pairs",
    )
    return parser.parse_args()


def _parse_config(raw: str) -> tuple[int, int]:
    batch_s, parallel_s = raw.split(",", 1)
    return int(batch_s), int(parallel_s)


def _delete_collection(collection_name: str, qdrant_url: str) -> None:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)


def main() -> int:
    args = parse_args()

    from backend.app.core.config import settings
    from backend.app.etl.ingest_checkpoint import IngestCheckpoint
    from backend.app.etl.ingest_pipeline import checkpoint_path_for, run_s3_ingest

    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""
    settings.validate()

    qdrant_url = settings.DEFAULT_QDRANT_URL
    checkpoint_path = checkpoint_path_for(BENCHMARK_COLLECTION)

    print(
        f"Benchmark: collection={BENCHMARK_COLLECTION}, limit={args.limit}, "
        f"workers={args.workers}, configs={args.configs}"
    )
    print("-" * 72)

    results: list[tuple[str, float, int, int]] = []

    for raw in args.configs:
        batch_size, parallelism = _parse_config(raw)
        label = f"batch={batch_size}, parallel={parallelism}"

        _delete_collection(BENCHMARK_COLLECTION, qdrant_url)
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        t0 = time.perf_counter()
        checkpoint = run_s3_ingest(
            collection_name=BENCHMARK_COLLECTION,
            qdrant_url=qdrant_url,
            bucket_name=settings.BUCKET_NAME,
            num_workers=args.workers,
            embed_batch_size=batch_size,
            embed_parallelism=parallelism,
            checkpoint_path=checkpoint_path,
            reset_checkpoint=True,
            resume=False,
            limit=args.limit,
        )
        elapsed = time.perf_counter() - t0

        completed = len(checkpoint.completed_keys)
        failed = len(checkpoint.failed_keys)
        results.append((label, elapsed, completed, failed))

        print(
            f"{label:28}  {elapsed:7.1f}s  "
            f"completed={completed}  failed={failed}"
        )

    print("-" * 72)
    if results:
        baseline = results[0][1]
        print(f"Baseline ({results[0][0]}): {baseline:.1f}s")
        for label, elapsed, completed, failed in results[1:]:
            delta_pct = (elapsed - baseline) / baseline * 100 if baseline else 0.0
            sign = "+" if delta_pct >= 0 else ""
            print(
                f"{label:28}  {elapsed:7.1f}s  ({sign}{delta_pct:.1f}% vs baseline)  "
                f"completed={completed}  failed={failed}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
