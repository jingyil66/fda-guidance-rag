from __future__ import annotations

import json

import pytest

from backend.app.services import agent_tools


@pytest.fixture(autouse=True)
def clear_metadata_cache():
    agent_tools._load_metadata_items.cache_clear()
    yield
    agent_tools._load_metadata_items.cache_clear()


def test_list_guidance_filters(tmp_path, monkeypatch):
    metadata = [
        {
            "title": "<a href='/x'>Adaptive Design Guidance</a>",
            "summary": "Clinical trials adaptive designs.",
            "field_center": "CDER",
            "field_communication_type": "Final",
            "field_issue_datetime": "2024-01-15",
            "field_associated_media_2": '<a href="/media/111/download">pdf</a>',
            "url": "https://www.fda.gov/example",
        },
        {
            "title": "Seafood Safety",
            "summary": "Crabmeat filth.",
            "field_center": "CFSAN",
            "field_communication_type": "Draft",
            "field_issue_datetime": "2020-06-01",
            "field_associated_media_2": '<a href="/media/222/download">pdf</a>',
        },
    ]
    path = tmp_path / "metadata_with_summary.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(agent_tools.settings, "OUTPUT_METADATA_JSON", path)

    out = agent_tools.list_guidance(keyword="adaptive", center="CDER", year=2024, limit=5)
    assert "Adaptive Design" in out
    assert "pdf_id=111" in out
    assert "Seafood" not in out


def test_get_guidance_detail_by_pdf_id(tmp_path, monkeypatch):
    metadata = [
        {
            "title": "Test Guidance",
            "summary": "Short summary.",
            "field_center": "CDER",
            "field_communication_type": "Final",
            "field_issue_datetime": "2023-05-01",
            "field_associated_media_2": '<a href="/media/999/download">pdf</a>',
            "url": "https://www.fda.gov/test",
        }
    ]
    path = tmp_path / "metadata_with_summary.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(agent_tools.settings, "OUTPUT_METADATA_JSON", path)

    out = agent_tools.get_guidance_detail(pdf_id="999")
    assert "pdf_id: 999" in out
    assert "Short summary" in out


def test_search_guidance_empty_query():
    text, sources = agent_tools.search_guidance("")
    assert "empty" in text.lower()
    assert sources == []
