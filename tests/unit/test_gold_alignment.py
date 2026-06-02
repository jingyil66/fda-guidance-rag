from __future__ import annotations

from experiment.gold_alignment import align_gold_from_chunks, score_context_overlap


def test_score_context_overlap_and_align_gold_from_chunks():
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
