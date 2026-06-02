from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class IngestCheckpoint:
    """Tracks completed/failed S3 keys for resumable, idempotent ingest."""

    def __init__(
        self,
        path: Path,
        *,
        collection_name: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self.path = path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._lock = threading.Lock()
        self.completed_keys: set[str] = set()
        self.failed_keys: dict[str, str] = {}
        self.started_at: str | None = None
        self.updated_at: str | None = None

    @classmethod
    def load_or_create(
        cls,
        path: Path,
        *,
        collection_name: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> IngestCheckpoint:
        checkpoint = cls(
            path,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if path.exists():
            checkpoint._load_from_disk()
        else:
            checkpoint.started_at = _utc_now_iso()
            checkpoint.save()
        return checkpoint

    def _load_from_disk(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self._validate_config(payload)
        self.completed_keys = set(payload.get("completed_keys") or [])
        self.failed_keys = dict(payload.get("failed_keys") or {})
        self.started_at = payload.get("started_at")
        self.updated_at = payload.get("updated_at")

    def _validate_config(self, payload: dict[str, Any]) -> None:
        mismatches = []
        for field, expected in (
            ("collection_name", self.collection_name),
            ("chunk_size", self.chunk_size),
            ("chunk_overlap", self.chunk_overlap),
        ):
            if payload.get(field) != expected:
                mismatches.append(f"{field}: checkpoint={payload.get(field)!r} run={expected!r}")
        if mismatches:
            raise ValueError(
                "Checkpoint config mismatch (use --reset-checkpoint to start fresh): "
                + "; ".join(mismatches)
            )

    def is_completed(self, s3_key: str) -> bool:
        return s3_key in self.completed_keys

    def mark_completed(self, s3_key: str) -> None:
        with self._lock:
            self.completed_keys.add(s3_key)
            self.failed_keys.pop(s3_key, None)

    def mark_failed(self, s3_key: str, error: str) -> None:
        with self._lock:
            self.failed_keys[s3_key] = error
            self.completed_keys.discard(s3_key)

    def reset(self) -> None:
        with self._lock:
            self.completed_keys.clear()
            self.failed_keys.clear()
            self.started_at = _utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_keys": sorted(self.completed_keys),
            "failed_keys": dict(sorted(self.failed_keys.items())),
            "completed_count": len(self.completed_keys),
            "failed_count": len(self.failed_keys),
        }

    def save(self) -> None:
        with self._lock:
            self.updated_at = _utc_now_iso()
            payload = self.to_dict()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(self.path)
