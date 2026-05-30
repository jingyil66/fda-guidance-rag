"""
Smoke test for experiment.io_utils.

Usage (from project root):
    python experiment/scripts/test_io_utils.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    from experiment.io_utils import (
        append_registry_row,
        build_failure_records,
        build_registry_row,
        write_failures_jsonl,
    )

    run_config = {
        "run_id": "test_run",
        "stage": "e2e",
        "qa_dataset_path": "evaluation/qa_dataset.json",
        "qa_dataset_fields": {
            "question_field": "question",
            "answer_field": "answer",
            "context_field": "context",
            "source_id_field": "source_id",
        },
        "vector_store": {"collection_name": "test"},
        "chunking": {"strategy": "fixed", "fixed": {"chunk_size": 600, "chunk_overlap": 200}},
        "retrieval": {
            "mode": "embedding_plus_rerank",
            "top_k_initial": 20,
            "top_k_final": 5,
            "rerank": {"enabled": True, "model": "ms-marco-MiniLM-L-12-v2"},
        },
        "generation": {"model": "gpt-4o-mini"},
    }
    summary = {
        "query_count": 2,
        "recall_at_5": 0.5,
        "recall_at_10": 0.5,
        "recall_at_20": 0.5,
        "mrr": 0.75,
        "context_precision_at_5": 0.4,
    }
    row = build_registry_row(run_config, summary, collection_name="test")
    assert row["run_id"] == "test_run"
    assert row["recall_at_5"] == "0.50"

    records = [
        {
            "qa_index": 0,
            "question": "q1",
            "gold_answer": "a1",
            "gold_context": "expected chunk text",
            "gold_source_id": "gold-1",
            "documents": [{"text": "expected chunk text here", "metadata": {"pdf_id": "122971"}}],
        },
        {
            "qa_index": 1,
            "question": "q2",
            "gold_answer": "a2",
            "gold_context": "missing chunk",
            "gold_source_id": "gold-2",
            "documents": [{"text": "other", "metadata": {"pdf_id": "999"}}],
        },
    ]
    per_query = [
        {"qa_index": 0, "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0},
        {"qa_index": 1, "recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0},
    ]
    failures = build_failure_records(run_config, records, per_query)
    assert len(failures) == 1
    assert failures[0]["failure_type_primary"] == "R1_not_retrieved"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        registry_path = tmp_path / "registry.csv"
        failures_path = tmp_path / "failures.jsonl"
        append_registry_row(row, registry_path=registry_path)
        write_failures_jsonl(failures_path, failures)

        with registry_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["run_id"] == "test_run"

        loaded = [json.loads(line) for line in failures_path.read_text(encoding="utf-8").splitlines()]
        assert loaded[0]["qa_index"] == 1

    print("io_utils ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
