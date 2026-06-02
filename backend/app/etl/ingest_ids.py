from __future__ import annotations

import uuid

# Stable namespace for deterministic Qdrant point IDs across re-runs.
INGEST_POINT_NAMESPACE = uuid.UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")


def chunk_point_id(pdf_id: str, chunk_index: int) -> str:
    """Return a deterministic point ID for a PDF chunk (idempotent upsert)."""
    key = f"{pdf_id}:{chunk_index}"
    return str(uuid.uuid5(INGEST_POINT_NAMESPACE, key))
