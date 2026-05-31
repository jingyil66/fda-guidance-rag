"""
Unit-style checks for experiment.answer_evaluators helpers.

Usage (from project root):
    python experiment/scripts/test_answer_evaluators.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    from experiment.answer_evaluators import (
        aggregate_answer_metrics,
        compute_e2e_success_rate,
        format_document_context,
        merge_retrieval_and_answer_metrics,
    )

    docs = [{"text": "chunk one"}, {"page_content": "chunk two"}]
    assert "chunk one" in format_document_context(docs)
    assert "chunk two" in format_document_context(docs)

    per_query = [
        {"qa_index": 0, "correctness": 1.0, "groundedness": 1.0, "relevance": 1.0},
        {"qa_index": 1, "correctness": 0.0, "groundedness": 1.0, "relevance": 1.0},
    ]
    summary = aggregate_answer_metrics(
        per_query,
        ["correctness", "groundedness", "relevance"],
    )
    assert summary["correctness"] == 0.5
    assert summary["groundedness"] == 1.0

    merged = merge_retrieval_and_answer_metrics(
        [{"qa_index": 0, "recall_at_5": 1.0}, {"qa_index": 1, "recall_at_5": 0.0}],
        per_query,
    )
    assert merged[0]["correctness"] == 1.0
    assert merged[1]["correctness"] == 0.0

    e2e = compute_e2e_success_rate(
        merged,
        rule="correctness_and_groundedness_pass",
    )
    assert e2e == 0.5

    from experiment.answer_evaluators import AnswerJudge, evaluate_answer_record

    judge = MagicMock(spec=AnswerJudge)
    judge.judge_correctness = MagicMock(return_value=1.0)
    judge.judge_groundedness = MagicMock(return_value=1.0)
    judge.judge_relevance = MagicMock(return_value=0.0)

    record = {
        "qa_index": 0,
        "question": "q",
        "answer": "a",
        "gold_answer": "gold",
        "documents": [{"text": "fact"}],
    }
    result = evaluate_answer_record(record, judge)
    assert result["correctness"] == 1.0
    assert result["relevance"] == 0.0

    print("answer_evaluators ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
