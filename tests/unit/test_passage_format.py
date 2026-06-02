from __future__ import annotations

import pytest

from backend.app.services.passage_format import (
    PASSAGE_FORMAT_TITLE_SECTION_CHUNK,
    format_passage_for_context,
    format_passage_for_rerank,
    restore_passage_text,
)


def test_format_passage_for_rerank(sample_passage):
    enriched = format_passage_for_rerank(sample_passage, PASSAGE_FORMAT_TITLE_SECTION_CHUNK)
    assert "Title: 122971" in enriched
    assert "Section: Page 15" in enriched
    assert "Chunk body text." in enriched


def test_restore_passage_text(sample_passage):
    enriched = format_passage_for_rerank(sample_passage, PASSAGE_FORMAT_TITLE_SECTION_CHUNK)
    ranked = dict(sample_passage)
    ranked["text"] = enriched
    ranked["score"] = 0.99

    restored = restore_passage_text(ranked, [sample_passage])
    assert restored["text"] == "Chunk body text."
    assert restored["score"] == 0.99


def test_format_passage_for_context_matches_rerank(sample_passage):
    enriched = format_passage_for_rerank(sample_passage, PASSAGE_FORMAT_TITLE_SECTION_CHUNK)
    context = format_passage_for_context(sample_passage, PASSAGE_FORMAT_TITLE_SECTION_CHUNK)
    assert context == enriched
