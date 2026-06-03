from __future__ import annotations

from backend.app.services.agent_service import _merge_sources


def test_merge_sources_dedupes_by_pdf_page_snippet():
    existing = [
        {"pdf_id": "1", "page": 2, "snippet": "alpha"},
        {"pdf_id": "2", "page": 1, "snippet": "beta"},
    ]
    new = [
        {"pdf_id": "1", "page": 2, "snippet": "alpha"},
        {"pdf_id": "3", "page": 1, "snippet": "gamma"},
    ]
    merged = _merge_sources(existing, new)
    assert len(merged) == 3
    assert merged[0]["pdf_id"] == "1"
    assert merged[-1]["pdf_id"] == "3"
