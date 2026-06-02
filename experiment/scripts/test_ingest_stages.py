from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from backend.app.etl.ingest_ids import chunk_point_id
from backend.app.etl.ingest_jsonl import (
    documents_to_records,
    read_chunk_jsonl,
    read_chunk_meta,
    write_chunk_jsonl,
    write_chunk_meta,
)
from backend.app.etl.ingest_paths import (
    chunk_jsonl_path,
    chunk_meta_path,
    layout_for_collection,
    pdf_cache_path,
)


def test_layout_paths():
    layout = layout_for_collection("fda_guidance_chunk600_overlap200")
    assert layout.pdf_cache_dir.name == "pdfs"
    assert layout.chunks_dir.name == "chunks"
    assert pdf_cache_path(layout, "100002") == layout.pdf_cache_dir / "100002.pdf"
    assert chunk_jsonl_path(layout, "100002") == layout.chunks_dir / "100002.jsonl"


def test_jsonl_roundtrip(tmp_path: Path):
    chunks = [
        Document(page_content="hello world", metadata={"pdf_id": "42", "page": "1", "chunk_index": 0}),
        Document(page_content="second chunk", metadata={"pdf_id": "42", "page": "2", "chunk_index": 1}),
    ]
    records = documents_to_records(chunks, "42")
    assert records[0].point_id == chunk_point_id("42", 0)
    assert records[1].point_id == chunk_point_id("42", 1)

    jsonl_path = tmp_path / "42.jsonl"
    write_chunk_jsonl(jsonl_path, records)
    loaded = read_chunk_jsonl(jsonl_path)
    assert len(loaded) == 2
    assert loaded[0].text == "hello world"
    assert loaded[1].metadata["chunk_index"] == "1"


def test_chunk_meta_roundtrip(tmp_path: Path):
    meta_path = tmp_path / "42.meta.json"
    write_chunk_meta(meta_path, s3_key="pdfs/42.pdf", pdf_id="42", chunk_count=7)
    payload = read_chunk_meta(meta_path)
    assert payload["s3_key"] == "pdfs/42.pdf"
    assert payload["chunk_count"] == 7


def test_stage_checkpoint_paths_are_distinct(tmp_path: Path):
    from backend.app.core.config import settings

    original = settings.INGEST_STATE_DIR
    settings.INGEST_STATE_DIR = tmp_path
    try:
        layout = layout_for_collection("test_collection")
        paths = {
            layout.download_checkpoint_path,
            layout.chunk_checkpoint_path,
            layout.embed_checkpoint_path,
        }
        assert len(paths) == 3
        assert chunk_meta_path(layout, "1").suffix == ".json"
    finally:
        settings.INGEST_STATE_DIR = original


if __name__ == "__main__":
    import tempfile

    test_layout_paths()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        test_jsonl_roundtrip(path)
        test_chunk_meta_roundtrip(path)
        test_stage_checkpoint_paths_are_distinct(path)
    print("OK: ingest stages tests passed")
