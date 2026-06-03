from __future__ import annotations

from experiment.agent_evaluators import (
    classify_agent_failure,
    dedupe_questions_for_run,
    evaluate_agent_trajectory,
    evaluate_task_success,
    expand_run_results,
    summarize_agent_eval,
    summarize_task_eval,
)


def test_t1_first_tool_pass():
    gold = {
        "id": "t1",
        "category": "T1",
        "expected_steps": [
            {"tool": "search_guidance", "args": {"query": "pediatric study"}},
        ],
        "forbidden_tools": ["list_guidance"],
    }
    actual = [
        {"tool": "search_guidance", "args": {"query": "FDA pediatric study plans"}},
    ]
    result = evaluate_agent_trajectory(gold, actual)
    assert result["passed"] is True
    assert result["tool_selection_accuracy"] is True
    assert result["forbidden_tool_violations"] == []


def test_t2_list_not_search():
    gold = {
        "id": "t2",
        "category": "T2",
        "expected_steps": [
            {"tool": "list_guidance", "args": {"keyword": "biosimilar"}},
        ],
        "forbidden_tools": [],
    }
    actual = [{"tool": "search_guidance", "args": {"query": "biosimilar"}}]
    result = evaluate_agent_trajectory(gold, actual)
    assert result["passed"] is False
    assert result["tool_selection_accuracy"] is False


def test_t4_ordered_subsequence_with_extra_step():
    gold = {
        "id": "t4",
        "category": "T4",
        "expected_steps": [
            {"tool": "list_guidance", "args": {"keyword": "REMS"}},
            {"tool": "search_guidance", "args": {"query": "REMS patient"}},
        ],
        "forbidden_tools": [],
    }
    actual = [
        {"tool": "list_guidance", "args": {"keyword": "REMS", "limit": 10}},
        {"tool": "get_guidance_detail", "args": {"pdf_id": "123"}},
        {"tool": "search_guidance", "args": {"query": "REMS patient requirements", "pdf_id": "123"}},
    ]
    result = evaluate_agent_trajectory(gold, actual)
    assert result["trajectory_match_loose"] is True
    assert result["required_tool_recall"] is True


def test_forbidden_tool_fails():
    gold = {
        "id": "x",
        "category": "T1",
        "expected_steps": [{"tool": "search_guidance", "args": {"query": "test"}}],
        "forbidden_tools": ["list_guidance"],
    }
    actual = [
        {"tool": "list_guidance", "args": {"keyword": "test"}},
        {"tool": "search_guidance", "args": {"query": "test"}},
    ]
    result = evaluate_agent_trajectory(gold, actual)
    assert result["forbidden_tool_violations"] == ["list_guidance"]
    assert result["passed"] is False


def test_classify_a1_wrong_tool():
    gold = {
        "expected_steps": [{"tool": "search_guidance", "args": {"query": "x"}}],
        "forbidden_tools": [],
    }
    traj = {
        "passed": False,
        "tool_selection_accuracy": False,
        "trajectory_match_loose": False,
        "forbidden_tool_violations": [],
        "actual_steps": [{"tool": "list_guidance", "args": {}}],
    }
    code, _ = classify_agent_failure(gold, traj, run_row={"answer": "ok", "source_count": 1})
    assert code == "A1"


def test_classify_a5_max_steps():
    gold = {"expected_steps": [{"tool": "search_guidance", "args": {"query": "x"}}], "forbidden_tools": []}
    traj = {"passed": False, "tool_selection_accuracy": True, "trajectory_match_loose": True, "forbidden_tool_violations": [], "actual_steps": []}
    run_row = {
        "answer": "I could not finish within the allowed number of agent turns. Please narrow your question.",
        "source_count": 0,
    }
    code, reason = classify_agent_failure(gold, traj, run_row=run_row)
    assert code == "A5"
    assert "max_steps" in reason


def test_task_success_must_include():
    task = {
        "id": "t",
        "require_tools": ["search_guidance"],
        "must_include": ["NDA"],
        "answer": "gold",
        "min_sources": 0,
    }
    run_row = {
        "answer": "Submit HF reports in your NDA application.",
        "steps": [{"tool": "search_guidance", "args": {"query": "hf"}}],
        "source_count": 2,
    }
    result = evaluate_task_success(task, run_row, skip_judge=True)
    assert result["task_success"] is True


def test_summarize_task_eval():
    records = [
        {"task_success": True, "level": "L2", "correctness": 1.0},
        {"task_success": False, "level": "L1", "correctness": 0.0},
    ]
    summary = summarize_task_eval(records)
    assert summary["task_success_rate"] == 0.5
    assert summary["by_level"]["L2"]["task_success_rate"] == 1.0


def test_dedupe_and_expand_shared_question():
    rows = [
        {"id": "agent_t_001", "question": "Same question?", "category": "T1"},
        {"id": "agent_task_001", "question": "Same question?", "level": "L2"},
        {"id": "agent_t_002", "question": "Other?", "category": "T1"},
    ]
    unique, aliases = dedupe_questions_for_run(rows)
    assert len(unique) == 2
    assert len(aliases["same question?"]) == 2

    run_rows = [
        {
            "id": "agent_t_001",
            "question": "Same question?",
            "answer": "A",
            "steps": [],
            "sources": [],
            "source_count": 0,
            "latency_ms": 1,
        },
        {
            "id": "agent_t_002",
            "question": "Other?",
            "answer": "B",
            "steps": [],
            "sources": [],
            "source_count": 0,
            "latency_ms": 2,
        },
    ]
    meta = {r["id"]: r for r in rows}
    expanded = expand_run_results(run_rows, aliases, meta)
    by_id = {r["id"]: r for r in expanded}
    assert by_id["agent_task_001"]["answer"] == "A"
    assert by_id["agent_task_001"]["level"] == "L2"


def test_summarize_agent_eval():
    records = [
        {"passed": True, "tool_selection_accuracy": True, "trajectory_match_loose": True,
         "trajectory_match_strict": True, "required_tool_recall": True,
         "forbidden_tool_violations": [], "actual_step_count": 1, "category": "T1"},
        {"passed": False, "tool_selection_accuracy": False, "trajectory_match_loose": False,
         "trajectory_match_strict": False, "required_tool_recall": False,
         "forbidden_tool_violations": ["list_guidance"], "actual_step_count": 2, "category": "T2"},
    ]
    summary = summarize_agent_eval(records)
    assert summary["query_count"] == 2
    assert summary["pass_rate"] == 0.5
    assert summary["by_category"]["T1"]["pass_rate"] == 1.0
