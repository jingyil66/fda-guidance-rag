from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from qdrant_client import models

from backend.app.etl.ingest_jsonl import ChunkRecord, records_to_documents
from backend.app.etl.ingest_to_qdrant import _is_rate_limit_error


def _safe_embed_documents(embeddings, texts: list[str]) -> list[list[float]]:
    while True:
        try:
            return embeddings.embed_documents(texts)
        except Exception as exc:
            if _is_rate_limit_error(exc):
                print(f"Rate limit reached, waiting 1s... ({exc})", flush=True)
                time.sleep(1)
                continue
            raise


def _safe_upsert_points(vector_store, points: list[models.PointStruct]) -> None:
    while True:
        try:
            vector_store.client.upsert(
                collection_name=vector_store.collection_name,
                points=points,
            )
            return
        except Exception as exc:
            if _is_rate_limit_error(exc):
                print(f"Rate limit reached, waiting 1s... ({exc})", flush=True)
                time.sleep(1)
                continue
            raise


def _build_points(
    vector_store,
    *,
    texts: list[str],
    metadatas: list[dict],
    point_ids: list[str],
    vectors: list[list[float]],
) -> list[models.PointStruct]:
    points: list[models.PointStruct] = []
    vector_name = vector_store.vector_name
    content_key = vector_store.content_payload_key
    metadata_key = vector_store.metadata_payload_key
    for point_id, text, metadata, vector in zip(point_ids, texts, metadatas, vectors):
        points.append(
            models.PointStruct(
                id=point_id,
                vector={vector_name: vector},
                payload={
                    content_key: text,
                    metadata_key: metadata,
                },
            )
        )
    return points


def upsert_chunk_records(
    vector_store,
    records: list[ChunkRecord],
    *,
    embed_batch_size: int = 128,
    embed_parallelism: int = 1,
) -> int:
    if not records:
        return 0

    embeddings = vector_store.embeddings
    if embeddings is None:
        raise ValueError("Vector store has no embedding model configured.")

    batches: list[tuple[list[str], list[dict], list[str]]] = []
    for start in range(0, len(records), embed_batch_size):
        batch = records[start : start + embed_batch_size]
        batches.append(
            (
                [record.text for record in batch],
                [dict(record.metadata) for record in batch],
                [record.point_id for record in batch],
            )
        )

    upsert_lock = threading.Lock()
    total = 0

    def _run_batch(texts: list[str], metadatas: list[dict], point_ids: list[str]) -> int:
        vectors = _safe_embed_documents(embeddings, texts)
        points = _build_points(
            vector_store,
            texts=texts,
            metadatas=metadatas,
            point_ids=point_ids,
            vectors=vectors,
        )
        with upsert_lock:
            _safe_upsert_points(vector_store, points)
        return len(points)

    if embed_parallelism <= 1 or len(batches) <= 1:
        for texts, metadatas, point_ids in batches:
            total += _run_batch(texts, metadatas, point_ids)
        return total

    with ThreadPoolExecutor(max_workers=embed_parallelism) as pool:
        futures = [
            pool.submit(_run_batch, texts, metadatas, point_ids)
            for texts, metadatas, point_ids in batches
        ]
        for future in as_completed(futures):
            total += future.result()
    return total


def upsert_documents(
    vector_store,
    chunks: list[Document],
    *,
    pdf_id: str,
    embed_batch_size: int = 128,
    embed_parallelism: int = 1,
) -> int:
    from backend.app.etl.ingest_jsonl import documents_to_records

    records = documents_to_records(chunks, pdf_id)
    return upsert_chunk_records(
        vector_store,
        records,
        embed_batch_size=embed_batch_size,
        embed_parallelism=embed_parallelism,
    )
