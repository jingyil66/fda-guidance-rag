from __future__ import annotations

import json
import re
import time
from multiprocessing import Process, Queue
from pathlib import Path
from threading import Thread

from backend.app.core.config import settings
from backend.app.etl.ingest_checkpoint import IngestCheckpoint
from backend.app.etl.ingest_to_qdrant import (
    build_pdf_metadata_index,
    list_s3_pdf_keys,
    processor,
    qdrant_writer,
)
from backend.app.services.chunking_service import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


def checkpoint_path_for(collection_name: str, state_dir: Path | None = None) -> Path:
    base = state_dir or settings.INGEST_STATE_DIR
    safe_name = re.sub(r"[^\w.-]+", "_", collection_name)
    return base / f"{safe_name}_checkpoint.json"


def filter_work_keys(
    s3_keys: list[str],
    checkpoint: IngestCheckpoint,
    *,
    retry_failed: bool = False,
    resume: bool = True,
) -> list[str]:
    if retry_failed:
        return [key for key in s3_keys if key in checkpoint.failed_keys]
    if not resume:
        return list(s3_keys)
    return [key for key in s3_keys if not checkpoint.is_completed(key)]


def run_s3_ingest(
    *,
    collection_name: str | None = None,
    qdrant_url: str | None = None,
    bucket_name: str | None = None,
    chunk_strategy: str = "fixed",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    num_workers: int = 3,
    embed_batch_size: int = 64,
    embed_parallelism: int = 1,
    pdf_queue_size: int = 10,
    chunk_queue_size: int = 20,
    checkpoint_path: Path | None = None,
    reset_checkpoint: bool = False,
    retry_failed: bool = False,
    resume: bool = True,
    limit: int = 0,
    aws_region: str | None = None,
) -> IngestCheckpoint:
    """Run resumable S3 → Qdrant ingest with per-PDF checkpointing."""
    collection_name = collection_name or settings.DEFAULT_QDRANT_COLLECTION
    qdrant_url = qdrant_url or settings.DEFAULT_QDRANT_URL
    bucket_name = bucket_name or settings.BUCKET_NAME
    aws_region = aws_region or settings.AWS_REGION
    checkpoint_file = checkpoint_path or checkpoint_path_for(collection_name)

    checkpoint = IngestCheckpoint.load_or_create(
        checkpoint_file,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if reset_checkpoint:
        checkpoint.reset()
        checkpoint.save()

    pdf_metadata = build_pdf_metadata_index(settings.OUTPUT_METADATA_JSON)
    all_keys = list_s3_pdf_keys(bucket_name, aws_region=aws_region)
    work_keys = filter_work_keys(
        all_keys,
        checkpoint,
        retry_failed=retry_failed,
        resume=resume,
    )
    if limit > 0:
        work_keys = work_keys[:limit]

    print(f"{len(all_keys)} PDFs in S3; {len(work_keys)} queued for this run.")
    print(
        f"Ingest config: collection={collection_name}, strategy={chunk_strategy}, "
        f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, "
        f"workers={num_workers}, embed_batch_size={embed_batch_size}, "
        f"embed_parallelism={embed_parallelism}"
    )
    print(
        f"Checkpoint: {checkpoint_file} "
        f"(completed={len(checkpoint.completed_keys)}, failed={len(checkpoint.failed_keys)})"
    )

    if not work_keys:
        print("Nothing to ingest.")
        return checkpoint

    process_queue: Queue = Queue(maxsize=pdf_queue_size)
    chunk_queue: Queue = Queue(maxsize=chunk_queue_size)

    writer_thread = Thread(
        target=qdrant_writer,
        kwargs={
            "chunk_queue": chunk_queue,
            "collection_name": collection_name,
            "qdrant_url": qdrant_url,
            "embed_batch_size": embed_batch_size,
            "embed_parallelism": embed_parallelism,
            "checkpoint": checkpoint,
        },
    )
    writer_thread.start()

    processor_kwargs = {
        "bucket_name": bucket_name,
        "chunk_strategy": chunk_strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "aws_region": aws_region,
    }

    procs: list[Process] = []
    for _ in range(num_workers):
        proc = Process(
            target=processor,
            args=(process_queue, chunk_queue, pdf_metadata),
            kwargs=processor_kwargs,
        )
        proc.start()
        procs.append(proc)

    for key in work_keys:
        while process_queue.full():
            time.sleep(0.1)
        process_queue.put(key)

    for _ in procs:
        process_queue.put(None)

    for proc in procs:
        proc.join()

    chunk_queue.put(None)
    writer_thread.join()

    print(
        f"Run finished. completed={len(checkpoint.completed_keys)}, "
        f"failed={len(checkpoint.failed_keys)}"
    )
    return checkpoint


def write_ingest_report(checkpoint: IngestCheckpoint, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(checkpoint.to_dict(), indent=2), encoding="utf-8")
