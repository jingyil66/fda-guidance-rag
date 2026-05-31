from __future__ import annotations

from pathlib import Path

from experiment.evaluators import normalize_text


def context_probe(gold_context: str, *, min_probe_len: int = 80) -> str:
    norm_gold = normalize_text(gold_context)
    if not norm_gold:
        return ""
    probe_len = min(min_probe_len, len(norm_gold))
    return norm_gold[:probe_len]


def score_context_overlap(gold_context: str, candidate_text: str, *, min_probe_len: int = 80) -> float:
    norm_gold = normalize_text(gold_context)
    norm_candidate = normalize_text(candidate_text)
    if not norm_gold or not norm_candidate:
        return 0.0

    if norm_gold in norm_candidate or norm_candidate in norm_gold:
        return 1.0

    probe = context_probe(gold_context, min_probe_len=min_probe_len)
    if probe and probe in norm_candidate:
        return 0.9
    if probe and probe in norm_gold and any(
        token in norm_candidate for token in probe.split()[:8] if len(token) > 4
    ):
        return 0.5
    return 0.0


def align_gold_from_chunks(
    gold_context: str,
    chunks: list[dict],
    *,
    min_probe_len: int = 80,
    min_score: float = 0.9,
) -> dict | None:
    best_match: dict | None = None
    best_score = 0.0

    for chunk in chunks:
        text = chunk.get("text") or chunk.get("page_content") or ""
        score = score_context_overlap(
            gold_context,
            text,
            min_probe_len=min_probe_len,
        )
        if score > best_score:
            best_score = score
            metadata = chunk.get("metadata") or {}
            best_match = {
                "gold_pdf_id": str(metadata.get("pdf_id") or ""),
                "gold_page": metadata.get("page"),
                "alignment_score": score,
                "alignment_method": "local_chunk_scan",
            }

    if best_match and best_score >= min_score and best_match["gold_pdf_id"]:
        best_match["gold_page"] = int(best_match["gold_page"]) if best_match["gold_page"] is not None else None
        return best_match
    return None


def build_local_chunks(
    data_dir: Path,
    *,
    pdf_ids: set[str] | None = None,
    chunk_size: int = 600,
    chunk_overlap: int = 200,
) -> list[dict]:
    from backend.app.services.chunking_service import chunk_fixed, load_pdf_pages

    chunks: list[dict] = []
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if pdf_ids:
        pdf_files = [path for path in pdf_files if path.stem in pdf_ids]

    for pdf_path in pdf_files:
        pages = load_pdf_pages(pdf_path)
        for chunk in chunk_fixed(
            pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ):
            chunks.append(
                {
                    "text": chunk.page_content,
                    "metadata": chunk.metadata or {},
                }
            )
    return chunks


def align_qa_row(
    row: dict,
    chunks: list[dict],
    *,
    context_field: str = "context",
    min_probe_len: int = 80,
) -> dict:
    aligned = dict(row)
    gold_context = row.get(context_field, "")
    match = align_gold_from_chunks(
        gold_context,
        chunks,
        min_probe_len=min_probe_len,
    )
    if match:
        aligned.update(match)
    else:
        aligned["gold_pdf_id"] = aligned.get("gold_pdf_id", "")
        aligned["gold_page"] = aligned.get("gold_page")
        aligned["alignment_score"] = 0.0
        aligned["alignment_method"] = "unresolved"
    return aligned


def align_qa_dataset(
    rows: list[dict],
    chunks: list[dict],
    *,
    context_field: str = "context",
) -> tuple[list[dict], dict]:
    aligned_rows = [
        align_qa_row(row, chunks, context_field=context_field)
        for row in rows
    ]
    resolved = sum(1 for row in aligned_rows if row.get("alignment_method") != "unresolved")
    summary = {
        "total": len(rows),
        "resolved": resolved,
        "unresolved": len(rows) - resolved,
    }
    return aligned_rows, summary
