from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process, Queue
from pathlib import Path
from threading import Thread

from backend.app.core.config import settings
from backend.app.db.qdrant_client import init_qdrant
from backend.app.etl.ingest_checkpoint import IngestCheckpoint
from backend.app.etl.ingest_embed import upsert_chunk_records
from backend.app.etl.ingest_jsonl import (
    documents_to_records,
    read_chunk_jsonl,
    read_chunk_meta,
    write_chunk_jsonl,
    write_chunk_meta,
)
from backend.app.etl.ingest_paths import (
    IngestStageLayout,
    chunk_jsonl_path,
    chunk_meta_path,
    layout_for_collection,
    pdf_cache_path,
)
from backend.app.etl.ingest_to_qdrant import (
    build_pdf_metadata_index,
    chunk_page_documents,
    list_s3_pdf_keys,
    _page_documents_from_pdf,
    _pdf_id_from_s3_key,
)
from backend.app.services.chunking_service import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


def _load_stage_checkpoint(
    layout: IngestStageLayout,
    stage: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    reset: bool = False,
) -> IngestCheckpoint:
    paths = {
        "download": layout.download_checkpoint_path,
        "chunk": layout.chunk_checkpoint_path,
        "embed": layout.embed_checkpoint_path,
    }
    if stage not in paths:
        raise ValueError(f"Unknown stage: {stage}")

    checkpoint = IngestCheckpoint.load_or_create(
        paths[stage],
        collection_name=layout.collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if reset:
        checkpoint.reset()
        checkpoint.save()
    return checkpoint


def _filter_work_keys(
    s3_keys: list[str],
    checkpoint: IngestCheckpoint,
    *,
    retry_failed: bool = False,
    resume: bool = True,
    limit: int = 0,
) -> list[str]:
    if retry_failed:
        work = [key for key in s3_keys if key in checkpoint.failed_keys]
    elif resume:
        work = [key for key in s3_keys if not checkpoint.is_completed(key)]
    else:
        work = list(s3_keys)
    if limit > 0:
        work = work[:limit]
    return work


def _download_one_pdf(
    *,
    s3_client,
    bucket_name: str,
    s3_key: str,
    dest_path: Path,
) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
    body = response["Body"].read()
    tmp_path = dest_path.with_suffix(".pdf.tmp")
    tmp_path.write_bytes(body)
    tmp_path.replace(dest_path)


def run_stage_download(
    *,
    layout: IngestStageLayout,
    bucket_name: str | None = None,
    aws_region: str | None = None,
    workers: int = 8,
    resume: bool = True,
    retry_failed: bool = False,
    reset_checkpoint: bool = False,
    limit: int = 0,
) -> IngestCheckpoint:
    import boto3

    bucket_name = bucket_name or settings.BUCKET_NAME
    aws_region = aws_region or settings.AWS_REGION
    checkpoint = _load_stage_checkpoint(
        layout,
        "download",
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        reset=reset_checkpoint,
    )

    s3_keys = list_s3_pdf_keys(bucket_name, aws_region=aws_region)
    work_keys = _filter_work_keys(
        s3_keys,
        checkpoint,
        retry_failed=retry_failed,
        resume=resume,
        limit=limit,
    )

    print(
        f"[download] {len(work_keys)} PDFs queued "
        f"(completed={len(checkpoint.completed_keys)}, failed={len(checkpoint.failed_keys)})"
    )
    if not work_keys:
        return checkpoint

    layout.pdf_cache_dir.mkdir(parents=True, exist_ok=True)
    s3_client = boto3.client("s3", region_name=aws_region)

    def _task(s3_key: str) -> tuple[str, str | None]:
        pdf_id = _pdf_id_from_s3_key(s3_key)
        dest = pdf_cache_path(layout, pdf_id)
        try:
            if not dest.exists():
                _download_one_pdf(
                    s3_client=s3_client,
                    bucket_name=bucket_name,
                    s3_key=s3_key,
                    dest_path=dest,
                )
            return s3_key, None
        except Exception as exc:
            return s3_key, str(exc)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_task, key): key for key in work_keys}
        for future in as_completed(futures):
            s3_key, error = future.result()
            if error:
                checkpoint.mark_failed(s3_key, error)
                print(f"[download] FAILED {s3_key}: {error}", flush=True)
            else:
                checkpoint.mark_completed(s3_key)
                print(f"[download] OK {s3_key}", flush=True)
            checkpoint.save()

    return checkpoint


def _resolve_pdf_path(
    *,
    layout: IngestStageLayout,
    s3_client,
    bucket_name: str,
    s3_key: str,
    pdf_id: str,
    use_pdf_cache: bool,
) -> tuple[str | None, bool, str | None]:
    """Return (local_path, is_temp, error)."""
    cached = pdf_cache_path(layout, pdf_id)
    if use_pdf_cache and cached.exists():
        return str(cached), False, None

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        pdf_content = response["Body"].read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_content)
            return tmp.name, True, None
    except Exception as exc:
        return None, False, str(exc)


def chunk_one_pdf(
    *,
    layout: IngestStageLayout,
    s3_key: str,
    pdf_metadata: dict,
    bucket_name: str,
    aws_region: str,
    chunk_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    use_pdf_cache: bool,
) -> tuple[str, str | None]:
    import boto3

    pdf_id = _pdf_id_from_s3_key(s3_key)
    s3_client = boto3.client("s3", region_name=aws_region)
    tmp_path: str | None = None

    try:
        local_path, is_temp, error = _resolve_pdf_path(
            layout=layout,
            s3_client=s3_client,
            bucket_name=bucket_name,
            s3_key=s3_key,
            pdf_id=pdf_id,
            use_pdf_cache=use_pdf_cache,
        )
        if error:
            return s3_key, error
        if not local_path:
            return s3_key, "missing PDF path"

        if is_temp:
            tmp_path = local_path

        page_docs = _page_documents_from_pdf(
            local_path,
            pdf_id=pdf_id,
            meta=pdf_metadata.get(pdf_id, {}),
        )
        chunks = chunk_page_documents(
            page_docs,
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index

        records = documents_to_records(chunks, pdf_id)
        jsonl_path = chunk_jsonl_path(layout, pdf_id)
        meta_path = chunk_meta_path(layout, pdf_id)
        write_chunk_jsonl(jsonl_path, records)
        write_chunk_meta(
            meta_path,
            s3_key=s3_key,
            pdf_id=pdf_id,
            chunk_count=len(records),
        )
        return s3_key, None
    except Exception as exc:
        return s3_key, str(exc)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _chunk_worker(
    work_queue: Queue,
    result_queue: Queue,
    *,
    layout: IngestStageLayout,
    pdf_metadata: dict,
    bucket_name: str,
    aws_region: str,
    chunk_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    use_pdf_cache: bool,
):
    while True:
        s3_key = work_queue.get()
        if s3_key is None:
            break
        result = chunk_one_pdf(
            layout=layout,
            s3_key=s3_key,
            pdf_metadata=pdf_metadata,
            bucket_name=bucket_name,
            aws_region=aws_region,
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_pdf_cache=use_pdf_cache,
        )
        result_queue.put(result)


def run_stage_chunk(
    *,
    layout: IngestStageLayout,
    bucket_name: str | None = None,
    aws_region: str | None = None,
    chunk_strategy: str = "fixed",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    workers: int = 4,
    use_pdf_cache: bool = False,
    resume: bool = True,
    retry_failed: bool = False,
    reset_checkpoint: bool = False,
    limit: int = 0,
) -> IngestCheckpoint:
    bucket_name = bucket_name or settings.BUCKET_NAME
    aws_region = aws_region or settings.AWS_REGION
    checkpoint = _load_stage_checkpoint(
        layout,
        "chunk",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        reset=reset_checkpoint,
    )

    s3_keys = list_s3_pdf_keys(bucket_name, aws_region=aws_region)
    work_keys = _filter_work_keys(
        s3_keys,
        checkpoint,
        retry_failed=retry_failed,
        resume=resume,
        limit=limit,
    )

    print(
        f"[chunk] strategy={chunk_strategy}, size={chunk_size}, overlap={chunk_overlap}, "
        f"use_pdf_cache={use_pdf_cache}, workers={workers}"
    )
    print(
        f"[chunk] {len(work_keys)} PDFs queued "
        f"(completed={len(checkpoint.completed_keys)}, failed={len(checkpoint.failed_keys)})"
    )
    if not work_keys:
        return checkpoint

    layout.chunks_dir.mkdir(parents=True, exist_ok=True)
    pdf_metadata = build_pdf_metadata_index(settings.OUTPUT_METADATA_JSON)

    work_queue: Queue = Queue()
    result_queue: Queue = Queue()
    procs: list[Process] = []
    for _ in range(max(1, workers)):
        proc = Process(
            target=_chunk_worker,
            args=(work_queue, result_queue),
            kwargs={
                "layout": layout,
                "pdf_metadata": pdf_metadata,
                "bucket_name": bucket_name,
                "aws_region": aws_region,
                "chunk_strategy": chunk_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "use_pdf_cache": use_pdf_cache,
            },
        )
        proc.start()
        procs.append(proc)

    for key in work_keys:
        work_queue.put(key)
    for _ in procs:
        work_queue.put(None)

    remaining = len(work_keys)
    while remaining > 0:
        s3_key, error = result_queue.get()
        remaining -= 1
        if error:
            checkpoint.mark_failed(s3_key, error)
            print(f"[chunk] FAILED {s3_key}: {error}", flush=True)
        else:
            checkpoint.mark_completed(s3_key)
            print(f"[chunk] OK {s3_key}", flush=True)
        checkpoint.save()

    for proc in procs:
        proc.join()

    return checkpoint


def _embed_worker(
    work_queue: Queue,
    result_queue: Queue,
    *,
    layout: IngestStageLayout,
    collection_name: str,
    qdrant_url: str,
    embed_batch_size: int,
    embed_parallelism: int,
):
    vector_store = init_qdrant(collection_name, qdrant_url)

    while True:
        item = work_queue.get()
        if item is None:
            break

        s3_key = item["s3_key"]
        pdf_id = item["pdf_id"]
        jsonl_path = item["jsonl_path"]

        try:
            records = read_chunk_jsonl(jsonl_path)
            upserted = upsert_chunk_records(
                vector_store,
                records,
                embed_batch_size=embed_batch_size,
                embed_parallelism=embed_parallelism,
            )
            result_queue.put(
                {
                    "s3_key": s3_key,
                    "pdf_id": pdf_id,
                    "upserted": upserted,
                    "error": None,
                }
            )
        except Exception as exc:
            result_queue.put(
                {
                    "s3_key": s3_key,
                    "pdf_id": pdf_id,
                    "upserted": 0,
                    "error": str(exc),
                }
            )


def _discover_embed_work(
    layout: IngestStageLayout,
    checkpoint: IngestCheckpoint,
    *,
    resume: bool,
    retry_failed: bool,
    limit: int,
) -> list[dict]:
    items: list[dict] = []
    for meta_file in sorted(layout.chunks_dir.glob("*.meta.json")):
        meta = read_chunk_meta(meta_file)
        s3_key = meta["s3_key"]
        pdf_id = meta["pdf_id"]
        jsonl_path = chunk_jsonl_path(layout, pdf_id)
        if not jsonl_path.exists():
            continue
        if retry_failed:
            if s3_key not in checkpoint.failed_keys:
                continue
        elif resume and checkpoint.is_completed(s3_key):
            continue
        items.append(
            {
                "s3_key": s3_key,
                "pdf_id": pdf_id,
                "jsonl_path": jsonl_path,
            }
        )
        if limit > 0 and len(items) >= limit:
            break
    return items


def run_stage_embed(
    *,
    layout: IngestStageLayout,
    qdrant_url: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embed_batch_size: int = 128,
    embed_parallelism: int = 4,
    workers: int = 1,
    resume: bool = True,
    retry_failed: bool = False,
    reset_checkpoint: bool = False,
    limit: int = 0,
) -> IngestCheckpoint:
    qdrant_url = qdrant_url or settings.DEFAULT_QDRANT_URL
    checkpoint = _load_stage_checkpoint(
        layout,
        "embed",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        reset=reset_checkpoint,
    )

    work_items = _discover_embed_work(
        layout,
        checkpoint,
        resume=resume,
        retry_failed=retry_failed,
        limit=limit,
    )

    print(
        f"[embed] collection={layout.collection_name}, batch={embed_batch_size}, "
        f"parallel={embed_parallelism}, workers={workers}"
    )
    print(
        f"[embed] {len(work_items)} PDFs queued "
        f"(completed={len(checkpoint.completed_keys)}, failed={len(checkpoint.failed_keys)})"
    )
    if not work_items:
        return checkpoint

    if workers <= 1:
        vector_store = init_qdrant(layout.collection_name, qdrant_url)
        total_chunks = 0
        for item in work_items:
            s3_key = item["s3_key"]
            pdf_id = item["pdf_id"]
            try:
                records = read_chunk_jsonl(item["jsonl_path"])
                upserted = upsert_chunk_records(
                    vector_store,
                    records,
                    embed_batch_size=embed_batch_size,
                    embed_parallelism=embed_parallelism,
                )
                total_chunks += upserted
                checkpoint.mark_completed(s3_key)
                checkpoint.save()
                print(
                    f"[embed] OK {s3_key}: pdf_id={pdf_id}, chunks={upserted}, "
                    f"total_chunks={total_chunks}",
                    flush=True,
                )
            except Exception as exc:
                checkpoint.mark_failed(s3_key, str(exc))
                checkpoint.save()
                print(f"[embed] FAILED {s3_key}: {exc}", flush=True)
        return checkpoint

    work_queue: Queue = Queue()
    result_queue: Queue = Queue()
    threads: list[Thread] = []
    for _ in range(workers):
        thread = Thread(
            target=_embed_worker,
            kwargs={
                "work_queue": work_queue,
                "result_queue": result_queue,
                "layout": layout,
                "collection_name": layout.collection_name,
                "qdrant_url": qdrant_url,
                "embed_batch_size": embed_batch_size,
                "embed_parallelism": embed_parallelism,
            },
        )
        thread.start()
        threads.append(thread)

    for item in work_items:
        work_queue.put(item)
    for _ in threads:
        work_queue.put(None)

    total_chunks = 0
    remaining = len(work_items)
    while remaining > 0:
        result = result_queue.get()
        remaining -= 1
        s3_key = result["s3_key"]
        pdf_id = result["pdf_id"]
        if result["error"]:
            checkpoint.mark_failed(s3_key, result["error"])
            print(f"[embed] FAILED {s3_key}: {result['error']}", flush=True)
        else:
            total_chunks += result["upserted"]
            checkpoint.mark_completed(s3_key)
            print(
                f"[embed] OK {s3_key}: pdf_id={pdf_id}, chunks={result['upserted']}, "
                f"total_chunks={total_chunks}",
                flush=True,
            )
        checkpoint.save()

    for thread in threads:
        thread.join()

    return checkpoint


def run_all_stages(
    *,
    collection_name: str | None = None,
    qdrant_url: str | None = None,
    bucket_name: str | None = None,
    aws_region: str | None = None,
    chunk_strategy: str = "fixed",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    download_workers: int = 8,
    chunk_workers: int = 4,
    embed_workers: int = 1,
    embed_batch_size: int = 128,
    embed_parallelism: int = 4,
    use_pdf_cache: bool = False,
    skip_download: bool = True,
    resume: bool = True,
    retry_failed: bool = False,
    reset_checkpoint: bool = False,
    limit: int = 0,
) -> dict[str, IngestCheckpoint]:
    collection_name = collection_name or settings.DEFAULT_QDRANT_COLLECTION
    layout = layout_for_collection(collection_name)

    print(f"Stage layout: {layout.base_dir}")
    checkpoints: dict[str, IngestCheckpoint] = {}

    if not skip_download:
        checkpoints["download"] = run_stage_download(
            layout=layout,
            bucket_name=bucket_name,
            aws_region=aws_region,
            workers=download_workers,
            resume=resume,
            retry_failed=retry_failed,
            reset_checkpoint=reset_checkpoint,
            limit=limit,
        )
        use_pdf_cache = True

    checkpoints["chunk"] = run_stage_chunk(
        layout=layout,
        bucket_name=bucket_name,
        aws_region=aws_region,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        workers=chunk_workers,
        use_pdf_cache=use_pdf_cache,
        resume=resume,
        retry_failed=retry_failed,
        reset_checkpoint=reset_checkpoint,
        limit=limit,
    )

    checkpoints["embed"] = run_stage_embed(
        layout=layout,
        qdrant_url=qdrant_url,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_batch_size=embed_batch_size,
        embed_parallelism=embed_parallelism,
        workers=embed_workers,
        resume=resume,
        retry_failed=retry_failed,
        reset_checkpoint=reset_checkpoint,
        limit=limit,
    )
    return checkpoints
