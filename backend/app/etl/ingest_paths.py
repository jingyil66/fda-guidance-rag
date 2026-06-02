from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from backend.app.core.config import settings


def safe_collection_name(collection_name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", collection_name)


@dataclass(frozen=True)
class IngestStageLayout:
    collection_name: str
    base_dir: Path
    pdf_cache_dir: Path
    chunks_dir: Path
    download_checkpoint_path: Path
    chunk_checkpoint_path: Path
    embed_checkpoint_path: Path


def layout_for_collection(
    collection_name: str,
    *,
    state_dir: Path | None = None,
) -> IngestStageLayout:
    state = state_dir or settings.INGEST_STATE_DIR
    safe = safe_collection_name(collection_name)
    base = state / safe
    return IngestStageLayout(
        collection_name=collection_name,
        base_dir=base,
        pdf_cache_dir=base / "pdfs",
        chunks_dir=base / "chunks",
        download_checkpoint_path=state / f"{safe}_stage_download.json",
        chunk_checkpoint_path=state / f"{safe}_stage_chunk.json",
        embed_checkpoint_path=state / f"{safe}_stage_embed.json",
    )


def pdf_cache_path(layout: IngestStageLayout, pdf_id: str) -> Path:
    return layout.pdf_cache_dir / f"{pdf_id}.pdf"


def chunk_jsonl_path(layout: IngestStageLayout, pdf_id: str) -> Path:
    return layout.chunks_dir / f"{pdf_id}.jsonl"


def chunk_meta_path(layout: IngestStageLayout, pdf_id: str) -> Path:
    return layout.chunks_dir / f"{pdf_id}.meta.json"
