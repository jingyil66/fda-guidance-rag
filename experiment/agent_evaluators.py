"""
Agent evaluation: tool routing (BFCL-inspired), task success (GAIA-inspired), failure taxonomy.

Compares predicted tool steps from get_agent_answer against agent_tool_gold JSON.
Task rows use must_include / require_tools plus optional GPT judge scores.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

MAX_STEPS_MESSAGE = "could not finish within the allowed number of agent turns"


def _normalize_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _args_match(expected: dict, actual: dict) -> tuple[bool, list[str]]:
    """Loose match: each non-empty expected key must appear in actual with compatible value."""
    mismatches: list[str] = []
    for key, expected_val in expected.items():
        if expected_val is None:
            continue
        if key not in actual:
            mismatches.append(f"missing key {key!r}")
            continue
        actual_val = actual[key]
        if isinstance(expected_val, bool):
            if bool(actual_val) != expected_val:
                mismatches.append(f"{key}: expected {expected_val!r}, got {actual_val!r}")
            continue
        if isinstance(expected_val, int) and not isinstance(expected_val, bool):
            try:
                if int(actual_val) != int(expected_val):
                    mismatches.append(f"{key}: expected {expected_val}, got {actual_val!r}")
            except (TypeError, ValueError):
                mismatches.append(f"{key}: expected int {expected_val}, got {actual_val!r}")
            continue
        exp_s = _normalize_str(expected_val).lower()
        act_s = _normalize_str(actual_val).lower()
        if not exp_s:
            continue
        if exp_s not in act_s and act_s not in exp_s:
            mismatches.append(f"{key}: expected contains {exp_s!r}, got {act_s!r}")
    return len(mismatches) == 0, mismatches


def step_match(expected: dict, actual: dict) -> tuple[bool, list[str]]:
    exp_tool = expected.get("tool", "")
    act_tool = actual.get("tool", "")
    if exp_tool != act_tool:
        return False, [f"tool: expected {exp_tool!r}, got {act_tool!r}"]

    exp_args = expected.get("args") or {}
    act_args = actual.get("args") or {}
    ok, mismatches = _args_match(exp_args, act_args)
    if not ok:
        return False, mismatches
    return True, []


def ordered_subsequence_match(expected_steps: list[dict], actual_steps: list[dict]) -> tuple[bool, list[str]]:
    """Each expected step matches some later actual step in order (extra steps allowed)."""
    if not expected_steps:
        return True, []

    actual_idx = 0
    details: list[str] = []
    for exp_i, exp in enumerate(expected_steps):
        matched = False
        while actual_idx < len(actual_steps):
            ok, errs = step_match(exp, actual_steps[actual_idx])
            actual_idx += 1
            if ok:
                matched = True
                break
            details.extend([f"skip actual[{actual_idx - 1}]: {e}" for e in errs[:2]])
        if not matched:
            details.append(f"expected step {exp_i} ({exp.get('tool')}) not found in order")
            return False, details
    return True, details


def first_tool_accuracy(expected_steps: list[dict], actual_steps: list[dict]) -> bool:
    if not expected_steps:
        return True
    if not actual_steps:
        return False
    return expected_steps[0].get("tool") == actual_steps[0].get("tool")


def required_tools_present(expected_steps: list[dict], actual_steps: list[dict]) -> bool:
    required = {s.get("tool") for s in expected_steps if s.get("tool")}
    actual_tools = {s.get("tool") for s in actual_steps if s.get("tool")}
    return required.issubset(actual_tools)


def forbidden_tools_violated(forbidden: list[str], actual_steps: list[dict]) -> list[str]:
    forbidden_set = {t for t in forbidden if t}
    if not forbidden_set:
        return []
    hits = []
    for step in actual_steps:
        tool = step.get("tool")
        if tool in forbidden_set:
            hits.append(tool)
    return hits


def evaluate_agent_trajectory(
    gold_row: dict,
    actual_steps: list[dict],
    *,
    run_row: dict | None = None,
    judge_row: dict | None = None,
    task_row: dict | None = None,
) -> dict:
    """Score one gold row against predicted steps from get_agent_answer."""
    expected = gold_row.get("expected_steps") or []
    forbidden = gold_row.get("forbidden_tools") or []

    tsa = first_tool_accuracy(expected, actual_steps)
    tmr_loose, tmr_details = ordered_subsequence_match(expected, actual_steps)
    required_recall = required_tools_present(expected, actual_steps)
    forbidden_hits = forbidden_tools_violated(forbidden, actual_steps)

    strict_len = len(actual_steps) == len(expected) and tmr_loose
    if strict_len and expected:
        strict_ok = all(
            step_match(expected[i], actual_steps[i])[0]
            for i in range(len(expected))
        )
    else:
        strict_ok = False

    passed = tsa and tmr_loose and required_recall and not forbidden_hits

    base = {
        "id": gold_row.get("id"),
        "category": gold_row.get("category"),
        "question": gold_row.get("question"),
        "passed": passed,
        "tool_selection_accuracy": tsa,
        "trajectory_match_loose": tmr_loose,
        "trajectory_match_strict": strict_ok,
        "required_tool_recall": required_recall,
        "forbidden_tool_violations": forbidden_hits,
        "expected_step_count": len(expected),
        "actual_step_count": len(actual_steps),
        "actual_steps": actual_steps,
        "match_details": tmr_details[:5],
    }
    failure_code, failure_reason = classify_agent_failure(
        gold_row,
        base,
        run_row=run_row,
        judge_row=judge_row,
        task_row=task_row,
    )
    base["failure_code"] = failure_code
    base["failure_reason"] = failure_reason
    return base


def _actual_tools(actual_steps: list[dict]) -> list[str]:
    return [s.get("tool") for s in actual_steps if s.get("tool")]


def classify_agent_failure(
    gold_row: dict | None,
    traj_eval: dict,
    *,
    run_row: dict | None = None,
    judge_row: dict | None = None,
    task_row: dict | None = None,
) -> tuple[str, str]:
    """Return (failure_code, reason). PASS when no failure detected."""
    run_row = run_row or {}
    answer = (run_row.get("answer") or "").lower()

    if run_row.get("error"):
        return "A5", "runtime_error"
    if MAX_STEPS_MESSAGE in answer:
        return "A5", "max_steps"

    if traj_eval.get("passed"):
        if task_row is not None:
            task_ok, task_reason = _task_rules_pass(task_row, run_row, judge_row)
            if not task_ok:
                if judge_row and (
                    judge_row.get("correctness") != 1.0
                    or judge_row.get("groundedness") != 1.0
                ):
                    return "A4", task_reason or "answer_quality"
                return "A4", task_reason or "task_rules"
        elif judge_row and (
            judge_row.get("correctness") != 1.0 or judge_row.get("groundedness") != 1.0
        ):
            return "A4", "answer_quality"
        return "PASS", ""

    if traj_eval.get("forbidden_tool_violations"):
        return "A1", "forbidden_tool"

    if not traj_eval.get("tool_selection_accuracy"):
        return "A1", "wrong_first_tool"

    if gold_row:
        expected = gold_row.get("expected_steps") or []
        needs_search = any(s.get("tool") == "search_guidance" for s in expected)
        if needs_search and int(run_row.get("source_count") or 0) == 0:
            tools = _actual_tools(traj_eval.get("actual_steps") or [])
            if "search_guidance" in tools:
                return "A3", "empty_retrieval"

    if not traj_eval.get("trajectory_match_loose"):
        return "A2", "trajectory_mismatch"

    if not traj_eval.get("required_tool_recall"):
        return "A2", "missing_required_tool"

    return "A2", "routing_partial"


def _task_rules_pass(
    task_row: dict,
    run_row: dict,
    judge_row: dict | None,
    *,
    skip_judge: bool = False,
) -> tuple[bool, str]:
    answer = run_row.get("answer") or ""
    steps = run_row.get("steps") or []
    tools = _actual_tools(steps)

    for tool in task_row.get("require_tools") or []:
        if tool not in tools:
            return False, f"missing_tool:{tool}"

    for phrase in task_row.get("must_include") or []:
        if phrase and phrase.lower() not in answer.lower():
            return False, f"must_include:{phrase}"

    min_sources = int(task_row.get("min_sources") or 0)
    if min_sources > 0 and int(run_row.get("source_count") or 0) < min_sources:
        return False, "min_sources"

    if task_row.get("answer") and not skip_judge:
        if judge_row is None:
            return False, "judge_missing"
        if judge_row.get("correctness") != 1.0:
            return False, "correctness"
        if judge_row.get("groundedness") != 1.0:
            return False, "groundedness"
        level = (task_row.get("level") or "").upper()
        if level in ("L2", "L3") and judge_row.get("relevance") != 1.0:
            return False, "relevance"

    return True, ""


def evaluate_task_success(
    task_row: dict,
    run_row: dict,
    judge_row: dict | None = None,
    *,
    skip_judge: bool = False,
) -> dict:
    ok, reason = _task_rules_pass(task_row, run_row, judge_row, skip_judge=skip_judge)
    return {
        "id": task_row.get("id"),
        "level": task_row.get("level"),
        "question": task_row.get("question"),
        "task_success": ok,
        "task_failure_reason": reason if not ok else "",
        "source_count": run_row.get("source_count"),
        "step_count": len(run_row.get("steps") or []),
        "tools_used": _actual_tools(run_row.get("steps") or []),
        "correctness": (judge_row or {}).get("correctness"),
        "groundedness": (judge_row or {}).get("groundedness"),
        "relevance": (judge_row or {}).get("relevance"),
    }


def sources_to_documents(sources: list[dict]) -> list[dict]:
    documents = []
    for source in sources or []:
        text = source.get("snippet") or source.get("text") or source.get("page_content") or ""
        text = str(text).strip()
        if text:
            documents.append({"text": text})
    return documents


def merge_eval_questions(tool_rows: list[dict], task_rows: list[dict]) -> list[dict]:
    """Union tool + task gold by id; task rows win on duplicate id."""
    by_id: dict[str, dict] = {}
    for row in tool_rows:
        rid = row.get("id")
        if rid:
            by_id[rid] = {"id": rid, "question": row.get("question", ""), "category": row.get("category")}
    for row in task_rows:
        rid = row.get("id")
        if not rid:
            continue
        entry = by_id.get(rid, {})
        entry.update(
            {
                "id": rid,
                "question": row.get("question", entry.get("question", "")),
                "level": row.get("level"),
            }
        )
        by_id[rid] = entry
    return list(by_id.values())


def _normalize_question(question: str) -> str:
    return " ".join((question or "").split()).lower()


def dedupe_questions_for_run(question_rows: list[dict]) -> tuple[list[dict], dict[str, list[str]]]:
    """One agent run per unique question; map normalized question -> all eval ids."""
    unique: list[dict] = []
    aliases: dict[str, list[str]] = {}
    seen_questions: dict[str, str] = {}

    for row in question_rows:
        rid = row.get("id")
        if not rid:
            continue
        qnorm = _normalize_question(row.get("question", ""))
        aliases.setdefault(qnorm, []).append(rid)
        if qnorm in seen_questions:
            continue
        seen_questions[qnorm] = rid
        unique.append(row)

    return unique, aliases


def expand_run_results(
    run_rows: list[dict],
    aliases: dict[str, list[str]],
    question_meta: dict[str, dict],
) -> list[dict]:
    """Copy each unique run to every eval id that shares the same question text."""
    by_id = {r["id"]: r for r in run_rows if r.get("id")}
    expanded: list[dict] = []

    for qnorm, ids in aliases.items():
        source_id = next((i for i in ids if i in by_id), ids[0] if ids else None)
        if not source_id or source_id not in by_id:
            continue
        base = by_id[source_id]
        for rid in ids:
            meta = question_meta.get(rid, {})
            expanded.append(
                {
                    **base,
                    "id": rid,
                    "category": meta.get("category", base.get("category")),
                    "level": meta.get("level", base.get("level")),
                    "question": meta.get("question", base.get("question")),
                }
            )
    return expanded


def validate_task_tool_links(
    tool_rows: list[dict],
    task_rows: list[dict],
) -> list[str]:
    """Return human-readable errors when linked_tool_id does not match task question."""
    tool_by_id = {r["id"]: r for r in tool_rows if r.get("id")}
    errors: list[str] = []
    for task in task_rows:
        link_id = task.get("linked_tool_id")
        if not link_id:
            continue
        tool = tool_by_id.get(link_id)
        task_id = task.get("id", "?")
        if tool is None:
            errors.append(f"{task_id}: linked_tool_id {link_id!r} not found in tool gold")
            continue
        if _normalize_question(task.get("question", "")) != _normalize_question(
            tool.get("question", "")
        ):
            errors.append(
                f"{task_id}: question mismatch with {link_id} "
                f"(task vs tool gold text differ)"
            )
    return errors


def summarize_agent_eval(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"query_count": 0}

    def rate(key: str) -> float:
        return sum(1 for r in records if r.get(key)) / n

    by_category: dict[str, dict] = {}
    for row in records:
        cat = row.get("category") or "unknown"
        by_category.setdefault(cat, {"count": 0, "passed": 0})
        by_category[cat]["count"] += 1
        if row.get("passed"):
            by_category[cat]["passed"] += 1

    category_rates = {
        cat: {
            "count": stats["count"],
            "pass_rate": stats["passed"] / stats["count"] if stats["count"] else 0.0,
        }
        for cat, stats in by_category.items()
    }

    failure_counts = Counter(r.get("failure_code") for r in records if r.get("failure_code") and r.get("failure_code") != "PASS")

    return {
        "query_count": n,
        "pass_rate": rate("passed"),
        "tool_selection_accuracy": rate("tool_selection_accuracy"),
        "trajectory_match_loose": rate("trajectory_match_loose"),
        "trajectory_match_strict": rate("trajectory_match_strict"),
        "required_tool_recall": rate("required_tool_recall"),
        "forbidden_violation_rate": sum(1 for r in records if r.get("forbidden_tool_violations")) / n,
        "avg_actual_steps": sum(r.get("actual_step_count", 0) for r in records) / n,
        "by_category": category_rates,
        "failure_counts": dict(failure_counts),
    }


def summarize_task_eval(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"query_count": 0}

    by_level: dict[str, dict] = {}
    for row in records:
        level = row.get("level") or "unknown"
        by_level.setdefault(level, {"count": 0, "passed": 0})
        by_level[level]["count"] += 1
        if row.get("task_success"):
            by_level[level]["passed"] += 1

    return {
        "query_count": n,
        "task_success_rate": sum(1 for r in records if r.get("task_success")) / n,
        "avg_correctness": _mean_metric(records, "correctness"),
        "avg_groundedness": _mean_metric(records, "groundedness"),
        "avg_relevance": _mean_metric(records, "relevance"),
        "by_level": {
            level: {
                "count": stats["count"],
                "task_success_rate": stats["passed"] / stats["count"] if stats["count"] else 0.0,
            }
            for level, stats in by_level.items()
        },
    }


def _mean_metric(records: list[dict], key: str) -> float | None:
    values = [r[key] for r in records if r.get(key) is not None]
    return sum(values) / len(values) if values else None


def summarize_full_agent_eval(
    tool_summary: dict,
    task_summary: dict,
    *,
    failure_rows: list[dict] | None = None,
) -> dict:
    out = {
        "tool_routing": tool_summary,
        "task_e2e": task_summary,
    }
    if failure_rows:
        out["failure_taxonomy"] = {
            "counts": dict(Counter(r.get("failure_code") for r in failure_rows if r.get("failure_code"))),
            "definitions": {
                "A1": "Wrong tool selection (first tool or forbidden tool)",
                "A2": "Wrong or incomplete tool arguments / trajectory",
                "A3": "Tool ran but empty retrieval (e.g. search returned no sources)",
                "A4": "Answer quality (judge or must_include / task rules)",
                "A5": "Max steps, timeout, or runtime error",
            },
        }
    return out
