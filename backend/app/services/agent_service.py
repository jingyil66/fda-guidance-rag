"""Tool-calling agent for FDA guidance Q&A."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from backend.app.core.config import settings
from backend.app.services.agent_tools import (
    get_guidance_detail,
    list_guidance,
    search_guidance,
)
from backend.app.services.generation_service import DEFAULT_LLM_MODEL

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""

AGENT_SYSTEM_PROMPT = """You are an FDA medical guidance assistant with tools.

Rules:
- For specific requirements or facts inside documents, call search_guidance.
- For browsing, filters, or document lists, call list_guidance.
- For one document overview by name or pdf_id, call get_guidance_detail; then search_guidance with pdf_id if details are needed.
- Base answers only on tool output. If tools return no relevant content, say you cannot answer from the corpus.
- Cite passage numbers [1], [2] when using search_guidance results.
- Use only the tools needed; when you have enough evidence, reply without further tool calls."""

MAX_AGENT_ROUNDS = 6
MAX_AGENT_STEPS = MAX_AGENT_ROUNDS  # backward-compatible alias

MAX_ROUNDS_MESSAGE = (
    "I could not finish within the allowed number of agent turns. Please narrow your question."
)


@dataclass
class AgentRunResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)


def _source_key(source: dict) -> tuple:
    return (
        str(source.get("pdf_id") or ""),
        str(source.get("page") or ""),
        (source.get("snippet") or source.get("text") or "")[:200],
    )


def _merge_sources(existing: list[dict], new: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    merged: list[dict] = []
    for source in existing + new:
        key = _source_key(source)
        if key in seen:
            continue
        seen.add(key)
        merged.append(source)
    return merged


def _build_tools(ctx: dict[str, Any]) -> list[StructuredTool]:
    def _search(query: str, top_k: int = 5, pdf_id: str = "") -> str:
        text, sources = search_guidance(
            query,
            top_k=top_k,
            pdf_id=pdf_id.strip() or None,
        )
        ctx["sources"] = _merge_sources(ctx.get("sources") or [], sources)
        return text

    def _list(
        keyword: str = "",
        center: str = "",
        communication_type: str = "",
        year: int | None = None,
        limit: int = 10,
    ) -> str:
        return list_guidance(
            keyword=keyword or None,
            center=center or None,
            communication_type=communication_type or None,
            year=year,
            limit=limit,
        )

    def _detail(pdf_id: str = "", title_keyword: str = "") -> str:
        return get_guidance_detail(
            pdf_id=pdf_id.strip() or None,
            title_keyword=title_keyword.strip() or None,
        )

    return [
        StructuredTool.from_function(
            _search,
            name="search_guidance",
            description=(
                "Semantic search over FDA guidance PDF chunks. "
                "Use for specific regulatory questions. Optional pdf_id scopes to one document."
            ),
        ),
        StructuredTool.from_function(
            _list,
            name="list_guidance",
            description=(
                "List/filter guidances by keyword, FDA center, communication type, or issue year. "
                "Use when the user wants a catalog, not passage-level facts."
            ),
        ),
        StructuredTool.from_function(
            _detail,
            name="get_guidance_detail",
            description=(
                "Get title, URL, and summary for one guidance by pdf_id or title_keyword."
            ),
        ),
    ]


def get_agent_answer(query: str, *, max_steps: int = MAX_AGENT_ROUNDS) -> dict:
    """Run the tool-calling agent loop. Returns answer, sources, and step trace.

    max_steps is the maximum number of LLM turns (each turn may invoke multiple tools).

    LangSmith: set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY; optional LANGCHAIN_PROJECT
    (defaults to fda-guidance-agent).
    """
    import os

    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "fda-guidance-agent"))

    query = (query or "").strip()
    if not query:
        return AgentRunResult(answer="Query is empty.").__dict__

    ctx: dict[str, Any] = {"sources": []}
    tools = _build_tools(ctx)
    tool_map = {tool.name: tool for tool in tools}
    llm = ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=0).bind_tools(tools)

    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]
    steps: list[dict[str, Any]] = []

    for _ in range(max_steps):
        ai_msg: AIMessage = llm.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            answer = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
            return AgentRunResult(answer=answer, sources=ctx["sources"], steps=steps).__dict__

        for tool_call in ai_msg.tool_calls:
            name = tool_call["name"]
            args = tool_call.get("args") or {}
            tool = tool_map.get(name)
            if tool is None:
                observation = f"Unknown tool: {name}"
            else:
                try:
                    observation = tool.invoke(args)
                except Exception as exc:
                    observation = f"Tool error ({name}): {exc}"

            steps.append({"tool": name, "args": args})
            messages.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )

    return AgentRunResult(
        answer=MAX_ROUNDS_MESSAGE,
        sources=ctx["sources"],
        steps=steps,
    ).__dict__
