from __future__ import annotations

from experiment.evaluators import (
    aggregate_metrics,
    context_precision_at_k,
    evaluate_record,
    is_retrieval_hit,
    mrr,
    recall_at_k,
)


def test_recall_mrr_and_precision():
    flags = [False, True, False, True]
    assert recall_at_k(flags, 1) == 0.0
    assert recall_at_k(flags, 2) == 1.0
    assert mrr(flags) == 0.5
    assert context_precision_at_k(flags, 4) == 0.5


def test_evaluate_record_text_overlap():
    record = {
        "qa_index": 0,
        "gold_context": "feedback on protocols prior to initiation",
        "gold_source_id": "old-id",
        "documents": [
            {"text": "unrelated chunk", "id": "1"},
            {"text": "feedback on protocols prior to initiation and conduct", "id": "2"},
        ],
    }
    metrics = evaluate_record(record, k_list=[1, 2])
    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_2"] == 1.0
    assert metrics["mrr"] == 0.5


def test_aggregate_metrics():
    record = {
        "qa_index": 0,
        "gold_context": "feedback on protocols prior to initiation",
        "gold_source_id": "old-id",
        "documents": [
            {"text": "unrelated chunk", "id": "1"},
            {"text": "feedback on protocols prior to initiation and conduct", "id": "2"},
        ],
    }
    metrics = evaluate_record(record, k_list=[1, 2])
    summary = aggregate_metrics([metrics], k_list=[1, 2])
    assert summary["query_count"] == 1
    assert summary["recall_at_2"] == 1.0


def test_is_retrieval_hit_by_page():
    assert is_retrieval_hit(
        "any text",
        gold_pdf_id="122971",
        gold_page=3,
        retrieved_metadata={"pdf_id": "122971", "page": 3},
    )
    assert not is_retrieval_hit(
        "any text",
        gold_pdf_id="122971",
        gold_page=3,
        retrieved_metadata={"pdf_id": "122971", "page": 4},
    )


def test_evaluate_record_page_alignment():
    page_record = {
        "qa_index": 1,
        "gold_pdf_id": "122971",
        "gold_page": 2,
        "documents": [
            {"text": "other", "metadata": {"pdf_id": "999", "page": 1}},
            {"text": "hit page", "metadata": {"pdf_id": "122971", "page": 2}},
        ],
    }
    page_metrics = evaluate_record(page_record, k_list=[1, 2])
    assert page_metrics["recall_at_1"] == 0.0
    assert page_metrics["recall_at_2"] == 1.0
