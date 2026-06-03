"""
Run agent evaluation: tool routing (P0), task E2E + judge (P1), failure taxonomy (P2).

Usage (from project root):
    python experiment/run_agent_eval.py
    python experiment/run_agent_eval.py --config experiment/configs/run_agent_eval_full_dev.json
    python experiment/run_agent_eval.py --limit 3 --preview
    python experiment/run_agent_eval.py --score-only --skip-judge
    python experiment/run_agent_eval.py --skip-tool-eval --skip-task-eval
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
    parser = argparse.ArgumentParser(description="Evaluate agent tool routing and task success.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "experiment" / "configs" / "run_agent_eval_full_dev.json",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max questions to run (0 = all)")
    parser.add_argument("--output", type=Path, default=None, help="Override agent_results.jsonl path")
    parser.add_argument("--score-only", action="store_true", help="Score saved results only")
    parser.add_argument("--results", type=Path, default=None, help="JSONL for --score-only")
    parser.add_argument("--preview", action="store_true", help="Print per-row details")
    parser.add_argument("--skip-judge", action="store_true", help="Skip GPT answer judge (task rows with gold answer)")
    parser.add_argument("--skip-tool-eval", action="store_true", help="Skip tool-routing metrics")
    parser.add_argument("--skip-task-eval", action="store_true", help="Skip task E2E metrics")
    return parser.parse_args()


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_optional_dataset(run_config: dict, key: str) -> list[dict]:
    path_str = run_config.get(key)
    if not path_str:
        return []
    path = resolve_project_path(path_str)
    if not path.exists():
        return []
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON list: {path}")
    return data


def default_output_dir(run_id: str) -> Path:
    return PROJECT_ROOT / "experiment" / "runs" / run_id


def run_agent_on_questions(
    question_rows: list[dict],
    *,
    max_steps: int,
) -> list[dict]:
    from backend.app.core.config import settings
    from backend.app.services.agent_service import get_agent_answer

    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""

    results = []
    for row in question_rows:
        question = row.get("question", "")
        started = time.perf_counter()
        try:
            agent_out = get_agent_answer(question, max_steps=max_steps)
            error = None
        except Exception as exc:
            agent_out = {"answer": "", "sources": [], "steps": []}
            error = str(exc)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        sources = agent_out.get("sources") or []
        results.append(
            {
                "id": row.get("id"),
                "category": row.get("category"),
                "level": row.get("level"),
                "question": question,
                "answer": agent_out.get("answer", ""),
                "steps": agent_out.get("steps", []),
                "sources": sources,
                "source_count": len(sources),
                "latency_ms": elapsed_ms,
                "error": error,
            }
        )
    return results


def score_tool_rows(tool_gold: list[dict], run_by_id: dict[str, dict]) -> list[dict]:
    from experiment.agent_evaluators import evaluate_agent_trajectory

    scored = []
    for gold in tool_gold:
        gid = gold.get("id")
        run_row = run_by_id.get(gid)
        if run_row is None:
            continue
        eval_row = evaluate_agent_trajectory(
            gold,
            run_row.get("steps") or [],
            run_row=run_row,
        )
        eval_row["latency_ms"] = run_row.get("latency_ms")
        eval_row["error"] = run_row.get("error")
        eval_row["answer_preview"] = (run_row.get("answer") or "")[:200]
        eval_row["source_count"] = run_row.get("source_count")
        scored.append(eval_row)
    return scored


def score_task_rows(
    task_gold: list[dict],
    run_by_id: dict[str, dict],
    *,
    judge,
    metrics: list[str],
    skip_judge: bool,
) -> list[dict]:
    from experiment.agent_evaluators import (
        evaluate_task_success,
        sources_to_documents,
    )
    from experiment.answer_evaluators import evaluate_answer_record

    scored = []
    for task in task_gold:
        tid = task.get("id")
        run_row = run_by_id.get(tid)
        if run_row is None:
            continue

        judge_row = None
        if task.get("answer") and not skip_judge and judge is not None:
            record = {
                "question": task.get("question", ""),
                "answer": run_row.get("answer", ""),
                "gold_answer": task.get("answer", ""),
                "documents": sources_to_documents(run_row.get("sources") or []),
                "error": run_row.get("error"),
            }
            judge_row = evaluate_answer_record(record, judge, metrics=metrics)

        eval_row = evaluate_task_success(task, run_row, judge_row, skip_judge=skip_judge)
        eval_row["latency_ms"] = run_row.get("latency_ms")
        eval_row["error"] = run_row.get("error")
        eval_row["linked_tool_id"] = task.get("linked_tool_id")
        scored.append(eval_row)
    return scored


def build_failure_report(
    tool_gold: list[dict],
    tool_scored: list[dict],
    task_gold: list[dict],
    task_scored: list[dict],
    run_by_id: dict[str, dict],
) -> list[dict]:
    from experiment.agent_evaluators import classify_agent_failure, evaluate_agent_trajectory

    gold_by_id = {g["id"]: g for g in tool_gold if g.get("id")}
    tool_eval_by_id = {r["id"]: r for r in tool_scored if r.get("id")}
    task_eval_by_id = {t["id"]: t for t in task_scored if t.get("id")}
    task_gold_by_id = {t["id"]: t for t in task_gold if t.get("id")}

    rows = []
    seen: set[str] = set()

    for gid, gold in gold_by_id.items():
        if gid not in run_by_id:
            continue
        run_row = run_by_id[gid]
        traj = tool_eval_by_id.get(gid) or evaluate_agent_trajectory(
            gold, run_row.get("steps") or [], run_row=run_row
        )
        task_row = None
        judge_row = None
        for task in task_gold:
            if task.get("linked_tool_id") == gid or task.get("id") == gid:
                task_row = task
                te = task_eval_by_id.get(task["id"])
                if te:
                    judge_row = {
                        "correctness": te.get("correctness"),
                        "groundedness": te.get("groundedness"),
                        "relevance": te.get("relevance"),
                    }
                break
        code, reason = classify_agent_failure(
            gold, traj, run_row=run_row, judge_row=judge_row, task_row=task_row
        )
        if code == "PASS" and not traj.get("passed"):
            code, reason = traj.get("failure_code", "A2"), traj.get("failure_reason", "")
        rows.append(
            {
                "id": gid,
                "eval_type": "tool",
                "failure_code": code,
                "failure_reason": reason,
                "passed": traj.get("passed"),
                "question": gold.get("question"),
            }
        )
        seen.add(gid)

    for tid, task in task_gold_by_id.items():
        if tid in seen or tid not in run_by_id:
            continue
        run_row = run_by_id[tid]
        te = task_eval_by_id.get(tid, {})
        judge_row = {
            "correctness": te.get("correctness"),
            "groundedness": te.get("groundedness"),
            "relevance": te.get("relevance"),
        }
        if te.get("task_success"):
            code, reason = "PASS", ""
        elif run_row.get("error") or "could not finish within" in (run_row.get("answer") or "").lower():
            code, reason = "A5", te.get("task_failure_reason") or "max_steps"
        elif judge_row and judge_row.get("correctness") != 1.0:
            code, reason = "A4", "correctness"
        elif judge_row and judge_row.get("groundedness") != 1.0:
            code, reason = "A4", "groundedness"
        elif te.get("task_failure_reason", "").startswith("missing_tool"):
            code, reason = "A1", te.get("task_failure_reason")
        else:
            code, reason = "A4", te.get("task_failure_reason") or "task_rules"
        rows.append(
            {
                "id": tid,
                "eval_type": "task",
                "failure_code": code,
                "failure_reason": reason,
                "passed": te.get("task_success"),
                "question": task.get("question"),
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    run_config = load_json(args.config)
    run_id = str(run_config.get("run_id", "agent_eval_run"))
    out_dir = default_output_dir(run_id)
    results_path = args.output or out_dir / "agent_results.jsonl"
    metrics_path = out_dir / "agent_metrics.json"
    failures_path = out_dir / "agent_failures.jsonl"
    task_results_path = out_dir / "agent_task_results.jsonl"

    tool_gold = [] if args.skip_tool_eval else load_optional_dataset(run_config, "agent_tool_dataset_path")
    if not tool_gold and not args.skip_tool_eval:
        tool_gold = load_optional_dataset(run_config, "agent_dataset_path")

    task_gold = [] if args.skip_task_eval else load_optional_dataset(run_config, "agent_task_dataset_path")

    from experiment.agent_evaluators import (
        dedupe_questions_for_run,
        expand_run_results,
        merge_eval_questions,
        summarize_agent_eval,
        summarize_full_agent_eval,
        summarize_task_eval,
        validate_task_tool_links,
    )

    link_errors = validate_task_tool_links(tool_gold, task_gold)
    if link_errors:
        print("Warning: task/tool gold link issues:", file=sys.stderr)
        for err in link_errors:
            print(f"  - {err}", file=sys.stderr)

    question_rows = merge_eval_questions(tool_gold, task_gold)
    if args.limit > 0:
        question_rows = question_rows[: args.limit]
    question_meta = {r["id"]: r for r in question_rows if r.get("id")}
    unique_rows, aliases = dedupe_questions_for_run(question_rows)

    agent_cfg = run_config.get("agent") or {}
    max_steps = int(agent_cfg.get("max_steps", 6))
    eval_cfg = run_config.get("evaluation") or {}
    judge_model = str(eval_cfg.get("judge_model", "gpt-4o-mini"))
    judge_metrics = list(eval_cfg.get("judge_metrics", ["correctness", "groundedness", "relevance"]))
    skip_judge = args.skip_judge or bool(eval_cfg.get("skip_judge"))

    if args.score_only:
        results_path = args.results or results_path
        if not results_path.exists():
            print(f"Results not found: {results_path}", file=sys.stderr)
            return 1
        run_rows = []
        with results_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    run_rows.append(json.loads(line))
    else:
        dup_count = len(question_rows) - len(unique_rows)
        dup_note = f", {dup_count} duplicate question(s) skipped" if dup_count else ""
        print(
            f"Running agent on {len(unique_rows)} unique questions "
            f"({len(question_rows)} eval ids{dup_note}, max_rounds={max_steps})..."
        )
        run_rows = expand_run_results(
            run_agent_on_questions(unique_rows, max_steps=max_steps),
            aliases,
            question_meta,
        )
        write_jsonl(results_path, run_rows)
        print(f"Wrote {len(run_rows)} rows to {results_path}")

    run_by_id = {r["id"]: r for r in run_rows if r.get("id")}

    judge = None
    if task_gold and not skip_judge and any(t.get("answer") for t in task_gold):
        from experiment.answer_evaluators import AnswerJudge

        judge = AnswerJudge(model=judge_model)

    tool_gold_run = [g for g in tool_gold if g.get("id") in run_by_id] if tool_gold else []
    task_gold_run = [t for t in task_gold if t.get("id") in run_by_id] if task_gold else []

    tool_scored: list[dict] = []
    if tool_gold_run and not args.skip_tool_eval:
        tool_scored = score_tool_rows(tool_gold_run, run_by_id)

    task_scored: list[dict] = []
    if task_gold_run and not args.skip_task_eval:
        task_scored = score_task_rows(
            task_gold_run,
            run_by_id,
            judge=judge,
            metrics=judge_metrics,
            skip_judge=skip_judge,
        )
        write_jsonl(task_results_path, task_scored)

    tool_summary = summarize_agent_eval(tool_scored) if tool_scored else {"query_count": 0}
    task_summary = summarize_task_eval(task_scored) if task_scored else {"query_count": 0}

    failure_rows = build_failure_report(
        tool_gold_run, tool_scored, task_gold_run, task_scored, run_by_id
    )
    summary = summarize_full_agent_eval(tool_summary, task_summary, failure_rows=failure_rows)

    summary["run_id"] = run_id
    summary["tool_dataset_path"] = run_config.get("agent_tool_dataset_path") or run_config.get("agent_dataset_path")
    summary["task_dataset_path"] = run_config.get("agent_task_dataset_path")
    summary["results_path"] = str(results_path.relative_to(PROJECT_ROOT))
    summary["scored_at"] = datetime.now(timezone.utc).isoformat()
    summary["langsmith"] = {
        "enabled": os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true", "yes"),
        "project": os.getenv("LANGCHAIN_PROJECT", "fda-guidance-agent"),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {metrics_path}")

    failed = [r for r in failure_rows if r.get("failure_code") != "PASS"]
    write_jsonl(failures_path, failed)
    print(f"Wrote {len(failed)} failures to {failures_path}")

    print("\n--- Agent eval summary ---")
    if tool_scored:
        print(f"  tool pass_rate:              {tool_summary.get('pass_rate', 0):.2f}")
        print(f"  tool_selection_accuracy:     {tool_summary.get('tool_selection_accuracy', 0):.2f}")
        print(f"  trajectory_match_loose:      {tool_summary.get('trajectory_match_loose', 0):.2f}")
        if tool_summary.get("failure_counts"):
            print(f"  tool failure_counts:         {tool_summary['failure_counts']}")
    if task_scored:
        print(f"  task_success_rate:           {task_summary.get('task_success_rate', 0):.2f}")
        print(f"  avg_correctness:             {task_summary.get('avg_correctness')}")
        print(f"  avg_groundedness:            {task_summary.get('avg_groundedness')}")
    if summary.get("failure_taxonomy"):
        print(f"  failure taxonomy:            {summary['failure_taxonomy'].get('counts')}")

    if args.preview:
        for row in tool_scored:
            status = "PASS" if row.get("passed") else "FAIL"
            print(f"\n[TOOL {status}] {row.get('id')} ({row.get('category')}) [{row.get('failure_code')}]")
            if row.get("actual_steps"):
                print(f"  tools: {[s.get('tool') for s in row['actual_steps']]}")
        for row in task_scored:
            status = "PASS" if row.get("task_success") else "FAIL"
            print(f"\n[TASK {status}] {row.get('id')} ({row.get('level')}) — {row.get('task_failure_reason')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
