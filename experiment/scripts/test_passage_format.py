"""Unit tests for passage_format helpers."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.passage_format import (
    PASSAGE_FORMAT_TITLE_SECTION_CHUNK,
    format_passage_for_context,
    format_passage_for_rerank,
    restore_passage_text,
)


def main() -> int:
    passage = {
        "id": 0,
        "text": "Chunk body text.",
        "metadata": {
            "title": "122971",
            "pdf_id": "122971",
            "page": 15,
        },
    }

    enriched = format_passage_for_rerank(passage, PASSAGE_FORMAT_TITLE_SECTION_CHUNK)
    assert "Title: 122971" in enriched
    assert "Section: Page 15" in enriched
    assert "Chunk body text." in enriched

    ranked = dict(passage)
    ranked["text"] = enriched
    ranked["score"] = 0.99
    restored = restore_passage_text(ranked, [passage])
    assert restored["text"] == "Chunk body text."
    assert restored["score"] == 0.99

    context = format_passage_for_context(passage, PASSAGE_FORMAT_TITLE_SECTION_CHUNK)
    assert context == enriched

    print("passage_format tests: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
