from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def is_retrieval_hit(
    retrieved_text: str,
    *,
    gold_context: str = "",
    gold_source_id: str = "",
    retrieved_id: str = "",
    min_probe_len: int = 80,
) -> bool:
    if gold_source_id and retrieved_id and gold_source_id == retrieved_id:
        return True

    norm_gold = normalize_text(gold_context)
    norm_retrieved = normalize_text(retrieved_text)
    if not norm_gold or not norm_retrieved:
        return False

    if norm_gold in norm_retrieved or norm_retrieved in norm_gold:
        return True

    probe_len = min(min_probe_len, len(norm_gold))
    if probe_len <= 0:
        return False

    probe = norm_gold[:probe_len]
    return probe in norm_retrieved


def build_relevance_flags(
    retrieved_items: list[dict],
    *,
    gold_context: str = "",
    gold_source_id: str = "",
) -> list[bool]:
    flags = []
    for item in retrieved_items:
        text = item.get("text") or item.get("page_content") or ""
        retrieved_id = str(item.get("id") or "")
        flags.append(
            is_retrieval_hit(
                text,
                gold_context=gold_context,
                gold_source_id=gold_source_id,
                retrieved_id=retrieved_id,
            )
        )
    return flags


def recall_at_k(relevance_flags: list[bool], k: int) -> float:
    if not relevance_flags:
        return 0.0
    return 1.0 if any(relevance_flags[:k]) else 0.0


def mrr(relevance_flags: list[bool]) -> float:
    for rank, hit in enumerate(relevance_flags, start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def context_precision_at_k(relevance_flags: list[bool], k: int) -> float:
    top_k = relevance_flags[:k]
    if not top_k:
        return 0.0
    return sum(1 for flag in top_k if flag) / len(top_k)


def evaluate_record(
    record: dict,
    *,
    k_list: list[int] | None = None,
) -> dict:
    k_list = k_list or [5, 10, 20]
    retrieved_items = record.get("documents") or record.get("retrieved_documents") or []

    flags = build_relevance_flags(
        retrieved_items,
        gold_context=record.get("gold_context", ""),
        gold_source_id=record.get("gold_source_id", ""),
    )

    per_query = {
        "qa_index": record.get("qa_index"),
        "hit_count": sum(1 for flag in flags if flag),
        "retrieved_count": len(flags),
        "mrr": mrr(flags),
    }
    for k in k_list:
        per_query[f"recall_at_{k}"] = recall_at_k(flags, k)
        per_query[f"context_precision_at_{k}"] = context_precision_at_k(flags, k)

    return per_query


def aggregate_metrics(per_query_metrics: list[dict], *, k_list: list[int] | None = None) -> dict:
    k_list = k_list or [5, 10, 20]
    if not per_query_metrics:
        return {"query_count": 0}

    query_count = len(per_query_metrics)
    summary = {
        "query_count": query_count,
        "mrr": sum(row["mrr"] for row in per_query_metrics) / query_count,
    }

    for k in k_list:
        recall_key = f"recall_at_{k}"
        precision_key = f"context_precision_at_{k}"
        summary[recall_key] = sum(row[recall_key] for row in per_query_metrics) / query_count
        summary[precision_key] = sum(row[precision_key] for row in per_query_metrics) / query_count

    return summary


def evaluate_records(
    records: list[dict],
    *,
    k_list: list[int] | None = None,
) -> tuple[list[dict], dict]:
    k_list = k_list or [5, 10, 20]
    per_query = [evaluate_record(record, k_list=k_list) for record in records]
    summary = aggregate_metrics(per_query, k_list=k_list)
    return per_query, summary
