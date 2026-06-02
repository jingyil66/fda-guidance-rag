from __future__ import annotations

import asyncio

import pytest

from backend.mcp.fda_guidance_server import (
    get_guidance_detail,
    list_guidance,
    mcp,
    search_guidance,
)
from backend.app.services import agent_tools


@pytest.fixture(autouse=True)
def clear_metadata_cache():
    agent_tools._load_metadata_items.cache_clear()
    yield
    agent_tools._load_metadata_items.cache_clear()


def test_mcp_registers_three_tools():
    tools = asyncio.run(mcp.list_tools())
    names = {tool.name for tool in tools}
    assert names == {"search_guidance", "list_guidance", "get_guidance_detail"}


def test_mcp_list_guidance_tool(tmp_path, monkeypatch):
    import json

    metadata = [
        {
            "title": "Test Guidance",
            "summary": "Summary text.",
            "field_center": "CDER",
            "field_communication_type": "Final",
            "field_issue_datetime": "2024-01-01",
            "field_associated_media_2": '<a href="/media/42/download">pdf</a>',
        }
    ]
    path = tmp_path / "metadata_with_summary.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(agent_tools.settings, "OUTPUT_METADATA_JSON", path)

    out = list_guidance(keyword="Test", limit=5)
    assert "pdf_id=42" in out


def test_mcp_search_guidance_empty_query():
    out = search_guidance("")
    assert "empty" in out.lower()
