"""
Unit-style checks for experiment.gold_alignment helpers.

Usage (from project root):
    python experiment/scripts/test_gold_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    from experiment.gold_alignment import align_gold_from_chunks, score_context_overlap

    gold = "feedback on protocols prior to initiation and conduct"
    chunks = [
        {
            "text": "unrelated chunk",
            "metadata": {"pdf_id": "999", "page": 1},
        },
        {
            "text": "feedback on protocols prior to initiation and conduct of the study",
            "metadata": {"pdf_id": "122971", "page": 4},
        },
    ]

    assert score_context_overlap(gold, chunks[1]["text"]) >= 0.9
    match = align_gold_from_chunks(gold, chunks)
    assert match is not None
    assert match["gold_pdf_id"] == "122971"
    assert match["gold_page"] == 4

    print("gold_alignment ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
