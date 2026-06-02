from __future__ import annotations

import json
from pathlib import Path

from backend.app.etl.ingest_checkpoint import IngestCheckpoint
from backend.app.etl.ingest_ids import chunk_point_id


def test_chunk_point_id_is_deterministic():
    first = chunk_point_id("123456", 0)
    second = chunk_point_id("123456", 0)
    third = chunk_point_id("123456", 1)
    assert first == second
    assert first != third


def test_checkpoint_roundtrip_and_resume(tmp_path: Path):
    path = tmp_path / "checkpoint.json"
    checkpoint = IngestCheckpoint.load_or_create(
        path,
        collection_name="test_collection",
        chunk_size=600,
        chunk_overlap=200,
    )
    checkpoint.mark_completed("pdfs/100.pdf")
    checkpoint.mark_failed("pdfs/200.pdf", "parse error")
    checkpoint.save()

    reloaded = IngestCheckpoint.load_or_create(
        path,
        collection_name="test_collection",
        chunk_size=600,
        chunk_overlap=200,
    )
    assert reloaded.is_completed("pdfs/100.pdf")
    assert "pdfs/200.pdf" in reloaded.failed_keys

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["completed_count"] == 1
    assert payload["failed_count"] == 1


def test_checkpoint_config_mismatch_raises(tmp_path: Path):
    path = tmp_path / "checkpoint.json"
    IngestCheckpoint.load_or_create(
        path,
        collection_name="test_collection",
        chunk_size=600,
        chunk_overlap=200,
    ).save()

    try:
        IngestCheckpoint.load_or_create(
            path,
            collection_name="other_collection",
            chunk_size=600,
            chunk_overlap=200,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


if __name__ == "__main__":
    from pathlib import Path as _Path
    import tempfile

    test_chunk_point_id_is_deterministic()
    with tempfile.TemporaryDirectory() as tmp:
        test_checkpoint_roundtrip_and_resume(_Path(tmp))
        test_checkpoint_config_mismatch_raises(_Path(tmp))
    print("OK: ingest checkpoint tests passed")
