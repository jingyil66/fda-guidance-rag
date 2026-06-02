"""FDA guidance agent tools (search, list, document detail)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from bs4 import BeautifulSoup

from backend.app.core.config import settings
from backend.app.services.generation_service import build_sources, format_context
from backend.app.services.passage_format import PASSAGE_FORMAT_LEGACY
from backend.app.services.pipeline_service import _normalize_config, _resolve_runtime
from backend.app.services.retrieval_service import retrieve_embedding

PDF_ID_PATTERN = re.compile(r"/media/(\d+)/download")


def _clean_title(raw: str) -> str:
    if not raw:
        return ""
    return BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)


def _pdf_id_from_item(item: dict) -> str:
    media = item.get("field_associated_media_2") or ""
    match = PDF_ID_PATTERN.search(media)
    return match.group(1) if match else ""


@lru_cache(maxsize=1)
def _load_metadata_items() -> tuple[dict, ...]:
    path = settings.OUTPUT_METADATA_JSON
    if not path.exists():
        return ()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return ()
    return tuple(data)


def search_guidance(
    query: str,
    *,
    top_k: int = 5,
    pdf_id: str | None = None,
    top_k_initial: int = 20,
) -> tuple[str, list[dict]]:
    """Semantic search over ingested guidance chunks. Returns (text for LLM, sources)."""
    query = (query or "").strip()
    if not query:
        return "Error: query is empty.", []

    cfg = _normalize_config({"top_k_final": top_k, "top_k_initial": max(top_k_initial, top_k * 4)})
    runtime = _resolve_runtime(cfg)

    passages = retrieve_embedding(
        query,
        cfg["collection_name"],
        top_k=cfg["top_k_initial"],
        client=runtime["client"],
        embeddings=runtime["embeddings"],
        qdrant_url=cfg["qdrant_url"],
    )

    if pdf_id:
        pdf_id_str = str(pdf_id).strip()
        passages = [
            p
            for p in passages
            if str((p.get("metadata") or {}).get("pdf_id", "")) == pdf_id_str
        ]

    ranked = passages[: cfg["top_k_final"]]
    if not ranked:
        hint = f" for pdf_id={pdf_id}" if pdf_id else ""
        return f"No passages found{hint}. Try list_guidance to find documents or broaden the query.", []

    context = format_context(ranked, passage_format=PASSAGE_FORMAT_LEGACY, numbered=True)
    sources = build_sources(ranked)
    text = (
        f"Retrieved {len(ranked)} passage(s):\n\n{context}\n\n"
        "Use only this content when answering. Cite segment numbers like [1], [2]."
    )
    return text, sources


def list_guidance(
    *,
    keyword: str | None = None,
    center: str | None = None,
    communication_type: str | None = None,
    year: int | None = None,
    limit: int = 10,
) -> str:
    """Filter guidance catalog metadata (no vector search)."""
    items = list(_load_metadata_items())
    if not items:
        return (
            f"No metadata at {settings.OUTPUT_METADATA_JSON}. "
            "Run metadata harvest (see README Ingest) before using list_guidance."
        )

    keyword_lower = (keyword or "").strip().lower()
    center_lower = (center or "").strip().lower()
    comm_lower = (communication_type or "").strip().lower()
    limit = max(1, min(limit, 25))

    matches: list[dict[str, Any]] = []
    for item in items:
        title = _clean_title(item.get("title", ""))
        summary = (item.get("summary") or "").strip()
        pdf_id = _pdf_id_from_item(item)
        item_center = (item.get("field_center") or "").strip()
        item_comm = (item.get("field_communication_type") or "").strip()
        issued = (item.get("field_issue_datetime") or "").strip()

        if keyword_lower:
            haystack = f"{title} {summary} {item.get('field_topics', '')}".lower()
            if keyword_lower not in haystack:
                continue
        if center_lower and center_lower not in item_center.lower():
            continue
        if comm_lower and comm_lower not in item_comm.lower():
            continue
        if year is not None and str(year) not in issued:
            continue

        matches.append(
            {
                "pdf_id": pdf_id,
                "title": title,
                "center": item_center,
                "communication_type": item_comm,
                "issued": issued,
                "url": item.get("url", ""),
                "summary": summary[:400] + ("…" if len(summary) > 400 else ""),
            }
        )

    if not matches:
        return "No guidances matched the filters."

    lines = [f"Found {len(matches)} match(es) (showing up to {limit}):"]
    for index, row in enumerate(matches[:limit], start=1):
        lines.append(
            f"\n[{index}] pdf_id={row['pdf_id']}\n"
            f"  title: {row['title']}\n"
            f"  center: {row['center']} | type: {row['communication_type']} | issued: {row['issued']}\n"
            f"  url: {row['url']}\n"
            f"  summary: {row['summary']}"
        )
    return "\n".join(lines)


def get_guidance_detail(
    *,
    pdf_id: str | None = None,
    title_keyword: str | None = None,
) -> str:
    """Return metadata summary for one guidance document."""
    pdf_id = (pdf_id or "").strip()
    title_keyword = (title_keyword or "").strip().lower()
    if not pdf_id and not title_keyword:
        return "Error: provide pdf_id or title_keyword."

    items = list(_load_metadata_items())
    if not items:
        return f"No metadata at {settings.OUTPUT_METADATA_JSON}."

    candidates: list[dict] = []
    for item in items:
        item_pdf = _pdf_id_from_item(item)
        title = _clean_title(item.get("title", ""))
        if pdf_id and item_pdf == pdf_id:
            candidates.append(item)
        elif title_keyword and title_keyword in title.lower():
            candidates.append(item)

    if not candidates:
        return "No document matched pdf_id or title_keyword."

    if len(candidates) > 1 and not pdf_id:
        lines = [f"Multiple documents ({len(candidates)}) match; pick a pdf_id:"]
        for item in candidates[:10]:
            lines.append(f"  pdf_id={_pdf_id_from_item(item)} | {_clean_title(item.get('title', ''))}")
        return "\n".join(lines)

    item = candidates[0]
    title = _clean_title(item.get("title", ""))
    summary = (item.get("summary") or "").strip() or "(no summary scraped)"
    return (
        f"pdf_id: {_pdf_id_from_item(item)}\n"
        f"title: {title}\n"
        f"center: {item.get('field_center', '')}\n"
        f"type: {item.get('field_communication_type', '')}\n"
        f"issued: {item.get('field_issue_datetime', '')}\n"
        f"url: {item.get('url', '')}\n"
        f"topics: {item.get('field_topics', '')}\n"
        f"regulated_product: {item.get('field_regulated_product_field', '')}\n\n"
        f"summary:\n{summary}"
    )
