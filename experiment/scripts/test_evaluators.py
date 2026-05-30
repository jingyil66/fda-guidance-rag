"""
Unit-style checks for experiment.evaluators.

Usage (from project root):
    python experiment/scripts/test_evaluators.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    from experiment.evaluators import (
        aggregate_metrics,
        context_precision_at_k,
        evaluate_record,
        mrr,
        recall_at_k,
    )

    flags = [False, True, False, True]
    assert recall_at_k(flags, 1) == 0.0
    assert recall_at_k(flags, 2) == 1.0
    assert mrr(flags) == 0.5
    assert context_precision_at_k(flags, 4) == 0.5

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

    summary = aggregate_metrics([metrics], k_list=[1, 2])
    assert summary["query_count"] == 1
    assert summary["recall_at_2"] == 1.0

    print("evaluators ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
