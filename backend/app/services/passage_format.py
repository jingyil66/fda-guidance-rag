from __future__ import annotations

PASSAGE_FORMAT_RAW = "raw"
PASSAGE_FORMAT_TITLE_SECTION_CHUNK = "title_section_chunk"
PASSAGE_FORMAT_LEGACY = "legacy"


def get_passage_body(passage: dict) -> str:
    return passage.get("text") or passage.get("page_content") or ""


def get_title(metadata: dict) -> str:
    return str(metadata.get("title") or metadata.get("pdf_id") or "Unknown")


def get_section_label(metadata: dict) -> str:
    section = metadata.get("section_title") or metadata.get("section")
    if section:
        return str(section)

    page = metadata.get("page")
    if page is not None:
        return f"Page {page}"

    return "Unknown"


def format_passage_title_section_chunk(passage: dict, *, include_body: bool = True) -> str:
    metadata = passage.get("metadata") or {}
    header = f"Title: {get_title(metadata)}\nSection: {get_section_label(metadata)}"
    if not include_body:
        return header

    body = get_passage_body(passage)
    return f"{header}\n\n{body}" if body else header


def format_passage_for_rerank(passage: dict, passage_format: str = PASSAGE_FORMAT_RAW) -> str:
    if passage_format == PASSAGE_FORMAT_TITLE_SECTION_CHUNK:
        return format_passage_title_section_chunk(passage)
    return get_passage_body(passage)


def format_passage_for_context(passage: dict, passage_format: str = PASSAGE_FORMAT_LEGACY) -> str:
    if passage_format == PASSAGE_FORMAT_TITLE_SECTION_CHUNK:
        return format_passage_title_section_chunk(passage)

    metadata = passage.get("metadata") or {}
    return (
        f"Content: {get_passage_body(passage)}\n"
        f"Title: {get_title(metadata)}\n"
        f"Page: {metadata.get('page', '?')}"
    )


def restore_passage_text(ranked_passage: dict, source_passages: list[dict]) -> dict:
    restored = dict(ranked_passage)
    passage_id = ranked_passage.get("id")
    if passage_id is not None:
        for source in source_passages:
            if source.get("id") == passage_id:
                restored["text"] = get_passage_body(source)
                return restored

    ranked_text = get_passage_body(ranked_passage)
    for source in source_passages:
        if get_passage_body(source) == ranked_text:
            restored["text"] = get_passage_body(source)
            return restored

    return restored
