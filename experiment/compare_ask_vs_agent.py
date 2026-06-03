"""
Compare fixed RAG (/ask pipeline) vs tool-calling agent on the same questions.

Usage:
    python experiment/compare_ask_vs_agent.py --limit 5
    python experiment/compare_ask_vs_agent.py --dataset experiment/subsets/agent_task_gold_dev.json
    LANGCHAIN_TRACING_V2=true python experiment/compare_ask_vs_agent.py --limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare get_answer vs get_agent_answer.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "experiment" / "subsets" / "agent_task_gold_dev.json",
    )
    parser.add_argument("--limit", type=int, default=10, help="Max questions (0 = all)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL output (default: experiment/runs/compare_ask_agent_<timestamp>/comparison.jsonl)",
    )
    parser.add_argument("--max-steps", type=int, default=6)
    return parser.parse_args()


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list")
    return data


def sources_to_documents(sources: list[dict]) -> list[dict]:
    from experiment.agent_evaluators import sources_to_documents as _convert

    return _convert(sources)


def main() -> int:
    args = parse_args()
    rows = load_questions(args.dataset)
    if args.limit > 0:
        rows = rows[: args.limit]

    from backend.app.core.config import settings
    from backend.app.services.agent_service import get_agent_answer
    from backend.app.services.rag_service import get_answer

    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "experiment" / "runs" / f"compare_ask_agent_{stamp}"
    out_path = args.output or out_dir / "comparison.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    judge = None
    if any(row.get("answer") for row in rows):
        from experiment.answer_evaluators import AnswerJudge

        judge = AnswerJudge(model="gpt-4o-mini")

    comparisons = []
    for row in rows:
        qid = row.get("id", "")
        question = row.get("question", "")
        gold = row.get("answer", "")

        t0 = time.perf_counter()
        try:
            rag_out = get_answer(question)
            rag_err = None
        except Exception as exc:
            rag_out = {"answer": "", "sources": []}
            rag_err = str(exc)
        rag_ms = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        try:
            agent_out = get_agent_answer(question, max_steps=args.max_steps)
            agent_err = None
        except Exception as exc:
            agent_out = {"answer": "", "sources": [], "steps": []}
            agent_err = str(exc)
        agent_ms = int((time.perf_counter() - t1) * 1000)

        entry = {
            "id": qid,
            "question": question,
            "rag": {
                "answer": rag_out.get("answer", ""),
                "source_count": len(rag_out.get("sources") or []),
                "latency_ms": rag_ms,
                "error": rag_err,
            },
            "agent": {
                "answer": agent_out.get("answer", ""),
                "source_count": len(agent_out.get("sources") or []),
                "step_count": len(agent_out.get("steps") or []),
                "tools": [s.get("tool") for s in agent_out.get("steps") or []],
                "latency_ms": agent_ms,
                "error": agent_err,
            },
            "gold_answer": gold,
        }

        if gold and judge is not None:
            from experiment.answer_evaluators import evaluate_answer_record

            rag_judge = evaluate_answer_record(
                {
                    "question": question,
                    "answer": entry["rag"]["answer"],
                    "gold_answer": gold,
                    "documents": sources_to_documents(rag_out.get("sources") or []),
                    "error": rag_err,
                },
                judge,
            )
            agent_judge = evaluate_answer_record(
                {
                    "question": question,
                    "answer": entry["agent"]["answer"],
                    "gold_answer": gold,
                    "documents": sources_to_documents(agent_out.get("sources") or []),
                    "error": agent_err,
                },
                judge,
            )
            entry["rag"]["judge"] = rag_judge
            entry["agent"]["judge"] = agent_judge

        comparisons.append(entry)
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"{qid}: rag {rag_ms}ms / agent {agent_ms}ms — tools {entry['agent'].get('tools')}")

    summary = {
        "count": len(comparisons),
        "dataset": str(args.dataset.relative_to(PROJECT_ROOT)),
        "output": str(out_path.relative_to(PROJECT_ROOT)),
        "langsmith_tracing": os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true", "yes"),
        "avg_rag_latency_ms": sum(c["rag"]["latency_ms"] for c in comparisons) / len(comparisons) if comparisons else 0,
        "avg_agent_latency_ms": sum(c["agent"]["latency_ms"] for c in comparisons) / len(comparisons) if comparisons else 0,
    }
    judged = [c for c in comparisons if c.get("rag", {}).get("judge")]
    if judged:
        for side in ("rag", "agent"):
            for metric in ("correctness", "groundedness", "relevance"):
                vals = [
                    c[side]["judge"][metric]
                    for c in judged
                    if c[side].get("judge", {}).get(metric) is not None
                ]
                if vals:
                    summary[f"avg_{side}_{metric}"] = sum(vals) / len(vals)

    summary_path = out_path.parent / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
