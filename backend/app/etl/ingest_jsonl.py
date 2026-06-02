from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from backend.app.etl.ingest_ids import chunk_point_id


@dataclass(frozen=True)
class ChunkRecord:
    point_id: str
    text: str
    metadata: dict[str, str]


def documents_to_records(chunks: list[Document], pdf_id: str) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for chunk in chunks:
        chunk_index = int(chunk.metadata.get("chunk_index", 0))
        safe_meta = {
            key: str(value) if value is not None else ""
            for key, value in (chunk.metadata or {}).items()
        }
        records.append(
            ChunkRecord(
                point_id=chunk_point_id(pdf_id, chunk_index),
                text=str(chunk.page_content),
                metadata=safe_meta,
            )
        )
    return records


def records_to_documents(records: list[ChunkRecord]) -> list[Document]:
    return [
        Document(page_content=record.text, metadata=dict(record.metadata))
        for record in records
    ]


def write_chunk_jsonl(path: Path, records: list[ChunkRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(
                {
                    "point_id": record.point_id,
                    "text": record.text,
                    "metadata": record.metadata,
                },
                ensure_ascii=False,
            )
            handle.write(line + "\n")
    tmp_path.replace(path)


def read_chunk_jsonl(path: Path) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(
                ChunkRecord(
                    point_id=str(payload["point_id"]),
                    text=str(payload["text"]),
                    metadata={
                        str(key): str(value) if value is not None else ""
                        for key, value in (payload.get("metadata") or {}).items()
                    },
                )
            )
    return records


def write_chunk_meta(path: Path, *, s3_key: str, pdf_id: str, chunk_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "s3_key": s3_key,
        "pdf_id": pdf_id,
        "chunk_count": chunk_count,
    }
    tmp_path = path.with_suffix(".meta.json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_chunk_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
