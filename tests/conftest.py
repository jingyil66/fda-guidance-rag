from __future__ import annotations

import os

import pytest


@pytest.fixture
def sample_passage():
    return {
        "id": 0,
        "text": "Chunk body text.",
        "metadata": {
            "title": "122971",
            "pdf_id": "122971",
            "page": 15,
        },
    }


def _qdrant_reachable() -> bool:
    try:
        from qdrant_client import QdrantClient

        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url, timeout=2)
        client.get_collections()
        return True
    except Exception:
        return False


@pytest.fixture
def require_qdrant():
    if not _qdrant_reachable():
        pytest.skip("Qdrant not reachable at QDRANT_URL")


@pytest.fixture
def require_openai():
    from backend.app.core.config import settings

    if not settings.OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set")
