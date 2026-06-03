from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment.agent_evaluators import validate_task_tool_links

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_GOLD = PROJECT_ROOT / "experiment" / "subsets" / "agent_tool_gold_dev.json"
TASK_GOLD = PROJECT_ROOT / "experiment" / "subsets" / "agent_task_gold_dev.json"


@pytest.fixture(scope="module")
def tool_gold() -> list[dict]:
    return json.loads(TOOL_GOLD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def task_gold() -> list[dict]:
    return json.loads(TASK_GOLD.read_text(encoding="utf-8"))


def test_task_tool_links_are_consistent(tool_gold: list[dict], task_gold: list[dict]) -> None:
    errors = validate_task_tool_links(tool_gold, task_gold)
    assert errors == [], "\n".join(errors)


def test_tool_gold_has_expected_new_rows(tool_gold: list[dict]) -> None:
    ids = {row["id"] for row in tool_gold}
    assert "agent_t_031" in ids
    assert "agent_t_032" in ids
