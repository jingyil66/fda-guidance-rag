from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process, Queue
from pathlib import Path
from threading import Thread

import boto3
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from backend.app.core.config import settings
from backend.app.db.qdrant_client import init_qdrant
from backend.app.etl.ingest_checkpoint import IngestCheckpoint
from backend.app.etl.ingest_ids import chunk_point_id
from backend.app.services.chunking_service import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_fixed,
    chunk_unstructured_by_section,
)


def build_pdf_metadata_index(metadata_path: Path) -> dict[str, dict]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    pdf_metadata: dict[str, dict] = {}
    for meta in data:
        media_link = meta.get("field_associated_media_2", "")
        match = re.search(r"/media/(\d+)/download", media_link)
        if match:
            pdf_metadata[match.group(1)] = meta
    return pdf_metadata


def list_s3_pdf_keys(bucket_name: str, *, aws_region: str = "us-east-1") -> list[str]:
    s3_client = boto3.client("s3", region_name=aws_region)
    keys: list[str] = []
    continuation_token = None

    while True:
        kwargs = {"Bucket": bucket_name, "Prefix": "pdfs/"}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        response = s3_client.list_objects_v2(**kwargs)
        keys.extend(obj["Key"] for obj in response.get("Contents", []))
        if response.get("IsTruncated"):
            continuation_token = response.get("NextContinuationToken")
        else:
            break
    return keys


def chunk_page_documents(
    page_docs: list[Document],
    *,
    chunk_strategy: str = "fixed",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_chars: int = 1200,
    min_chars: int = 200,
) -> list[Document]:
    if not page_docs:
        return []

    if chunk_strategy == "fixed":
        return chunk_fixed(
            page_docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if chunk_strategy == "unstructured_section":
        return chunk_unstructured_by_section(
            page_docs,
            max_chars=max_chars,
            min_chars=min_chars,
        )

    raise ValueError(f"Unsupported chunk strategy: {chunk_strategy}")


def _pdf_id_from_s3_key(s3_key: str) -> str:
    match = re.search(r"(\d+)", s3_key)
    return match.group(1) if match else s3_key


def _page_documents_from_pdf(
    tmp_path: str,
    *,
    pdf_id: str,
    meta: dict,
) -> list[Document]:
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    raw_title = meta.get("title", "")
    clean_title = BeautifulSoup(raw_title, "html.parser").get_text()

    documents: list[Document] = []
    for index, page_doc in enumerate(pages):
        page_number = index + 1
        page_meta = {
            "title": clean_title,
            "url": meta.get("url", ""),
            "summary": meta.get("summary", ""),
            "field_communication_type": meta.get("field_communication_type", ""),
            "pdf_id": pdf_id,
            "page": page_number,
            "field_issue_datetime": meta.get("field_issue_datetime", ""),
            "field_center": meta.get("field_center", ""),
            "field_issuing_office_taxonomy": meta.get("field_issuing_office_taxonomy", ""),
            "term_node_tid": meta.get("term_node_tid", ""),
            "field_topics": meta.get("field_topics", ""),
            "topics_product": meta.get("topics-product", ""),
            "field_regulated_product_field": meta.get("field_regulated_product_field", ""),
            "changed": meta.get("changed", ""),
        }
        documents.append(Document(page_content=page_doc.page_content, metadata=page_meta))
    return documents


def processor(
    process_queue: Queue,
    chunk_queue: Queue,
    pdf_metadata: dict,
    *,
    bucket_name: str = "04-bucket",
    chunk_strategy: str = "fixed",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_chars: int = 1200,
    min_chars: int = 200,
    aws_region: str = "us-east-1",
):
    import tempfile

    s3_client = boto3.client("s3", region_name=aws_region)

    while True:
        item = process_queue.get()
        if item is None:
            break

        s3_key = item
        pdf_id = _pdf_id_from_s3_key(s3_key)
        tmp_path: str | None = None

        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            pdf_content = response["Body"].read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_content)
                tmp_path = tmp.name

            page_docs = _page_documents_from_pdf(
                tmp_path,
                pdf_id=pdf_id,
                meta=pdf_metadata.get(pdf_id, {}),
            )
            chunks = chunk_page_documents(
                page_docs,
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_chars=max_chars,
                min_chars=min_chars,
            )
            for index, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = index

            while chunk_queue.full():
                time.sleep(0.1)

            chunk_queue.put(
                {
                    "s3_key": s3_key,
                    "pdf_id": pdf_id,
                    "chunks": chunks,
                }
            )
        except Exception as exc:
            while chunk_queue.full():
                time.sleep(0.1)
            chunk_queue.put(
                {
                    "s3_key": s3_key,
                    "pdf_id": pdf_id,
                    "error": str(exc),
                }
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)


def _is_rate_limit_error(exc: Exception) -> bool:
    try:
        import openai

        if isinstance(exc, openai.RateLimitError):
            return True
    except Exception:
        pass
    try:
        import openai.error

        if isinstance(exc, openai.error.RateLimitError):
            return True
    except Exception:
        pass
    return "rate limit" in str(exc).lower()


def _safe_add_documents(vector_store, docs: list[Document], ids: list[str]) -> None:
    while True:
        try:
            vector_store.add_documents(docs, ids=ids)
            return
        except Exception as exc:
            if _is_rate_limit_error(exc):
                print(f"Rate limit reached, waiting 1s... ({exc})", flush=True)
                time.sleep(1)
                continue
            raise


def _prepare_embed_batch(
    batch: list[Document],
    pdf_id: str,
) -> tuple[list[Document], list[str]]:
    safe_docs = []
    point_ids = []
    for chunk in batch:
        safe_meta = {
            key: str(value) if value is not None else ""
            for key, value in (chunk.metadata or {}).items()
        }
        safe_docs.append(Document(page_content=str(chunk.page_content), metadata=safe_meta))
        chunk_index = int(chunk.metadata.get("chunk_index", 0))
        point_ids.append(chunk_point_id(pdf_id, chunk_index))
    return safe_docs, point_ids


def _upsert_pdf_chunks(
    vector_store,
    *,
    chunks: list[Document],
    pdf_id: str,
    embed_batch_size: int,
    embed_parallelism: int,
    upsert_lock: threading.Lock | None = None,
) -> int:
    if not chunks:
        return 0

    batches: list[tuple[list[Document], list[str]]] = []
    for start in range(0, len(chunks), embed_batch_size):
        batch = chunks[start : start + embed_batch_size]
        batches.append(_prepare_embed_batch(batch, pdf_id))

    if embed_parallelism <= 1 or len(batches) <= 1:
        total = 0
        for safe_docs, point_ids in batches:
            if upsert_lock:
                with upsert_lock:
                    _safe_add_documents(vector_store, safe_docs, point_ids)
            else:
                _safe_add_documents(vector_store, safe_docs, point_ids)
            total += len(safe_docs)
        return total

    lock = upsert_lock or threading.Lock()
    total = 0

    def _run_batch(safe_docs: list[Document], point_ids: list[str]) -> int:
        with lock:
            _safe_add_documents(vector_store, safe_docs, point_ids)
        return len(safe_docs)

    with ThreadPoolExecutor(max_workers=embed_parallelism) as pool:
        futures = [
            pool.submit(_run_batch, safe_docs, point_ids)
            for safe_docs, point_ids in batches
        ]
        for future in as_completed(futures):
            total += future.result()
    return total


def qdrant_writer(
    chunk_queue: Queue,
    *,
    collection_name: str | None = None,
    qdrant_url: str | None = None,
    embed_batch_size: int = 64,
    embed_parallelism: int = 1,
    checkpoint: IngestCheckpoint | None = None,
):
    collection_name = collection_name or settings.DEFAULT_QDRANT_COLLECTION
    qdrant_url = qdrant_url or settings.DEFAULT_QDRANT_URL
    vector_store = init_qdrant(collection_name, qdrant_url)

    processed_pdfs = 0
    processed_chunks = 0

    while True:
        item = chunk_queue.get()
        if item is None:
            break

        s3_key = item["s3_key"]
        pdf_id = item.get("pdf_id") or _pdf_id_from_s3_key(s3_key)

        if item.get("error"):
            if checkpoint is not None:
                checkpoint.mark_failed(s3_key, item["error"])
                checkpoint.save()
            print(f"FAILED {s3_key}: {item['error']}", flush=True)
            continue

        chunks: list[Document] = item.get("chunks") or []
        if not chunks:
            if checkpoint is not None:
                checkpoint.mark_completed(s3_key)
                checkpoint.save()
            continue

        try:
            from backend.app.etl.ingest_embed import upsert_documents

            upserted = upsert_documents(
                vector_store,
                chunks,
                pdf_id=pdf_id,
                embed_batch_size=embed_batch_size,
                embed_parallelism=embed_parallelism,
            )
            processed_chunks += upserted

            processed_pdfs += 1
            if checkpoint is not None:
                checkpoint.mark_completed(s3_key)
                checkpoint.save()

            print(
                f"OK {s3_key}: pdf_id={pdf_id}, chunks={len(chunks)}, "
                f"total_pdfs={processed_pdfs}, total_chunks={processed_chunks}",
                flush=True,
            )
        except Exception as exc:
            if checkpoint is not None:
                checkpoint.mark_failed(s3_key, str(exc))
                checkpoint.save()
            print(f"FAILED {s3_key}: {exc}", flush=True)


# Backward-compatible alias used by older scripts.
def downloader_from_s3(bucket_name, key_list, process_queue: Queue, aws_region="us-east-1"):
    for key in key_list:
        while process_queue.full():
            time.sleep(0.1)
        process_queue.put(key)
