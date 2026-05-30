from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from experiment.evaluators import build_relevance_flags, mrr

REGISTRY_COLUMNS = [
    "run_id",
    "date",
    "stage",
    "qa_dataset_path",
    "qa_fields",
    "pdf_subset",
    "collection_name",
    "chunk_strategy",
    "chunk_size",
    "chunk_overlap",
    "retrieval_mode",
    "top_k_initial",
    "top_k_final",
    "rerank_enabled",
    "rerank_model",
    "llm_model",
    "judge_model",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "mrr",
    "context_precision_at_5",
    "correctness",
    "groundedness",
    "relevance",
    "e2e_success_rate",
    "notes",
]

DEFAULT_REGISTRY_PATH = Path("experiment/registry.csv")


def _format_metric(value: float | int | None) -> str:
    if value is None:
        return "0.00"
    return f"{float(value):.2f}"


def _qa_fields_string(run_config: dict) -> str:
    fields = run_config.get("qa_dataset_fields") or {}
    ordered = [
        fields.get("question_field", "question"),
        fields.get("answer_field", "answer"),
        fields.get("context_field", "context"),
        fields.get("source_id_field", "source_id"),
    ]
    return "|".join(ordered)


def build_registry_row(
    run_config: dict,
    summary: dict,
    *,
    collection_name: str | None = None,
    notes: str = "",
) -> dict[str, str]:
    vector_store = run_config.get("vector_store") or {}
    chunking = run_config.get("chunking") or {}
    fixed = chunking.get("fixed") or {}
    retrieval = run_config.get("retrieval") or {}
    rerank = retrieval.get("rerank") or {}
    generation = run_config.get("generation") or {}
    pdf_subset = run_config.get("pdf_subset") or {}

    query_count = summary.get("query_count", 0) or 0
    recall_at_5 = summary.get("recall_at_5", 0.0)
    e2e_success_rate = recall_at_5 if query_count else 0.0

    return {
        "run_id": str(run_config.get("run_id", "unnamed_run")),
        "date": date.today().isoformat(),
        "stage": str(run_config.get("stage", "")),
        "qa_dataset_path": str(run_config.get("qa_dataset_path", "")),
        "qa_fields": _qa_fields_string(run_config),
        "pdf_subset": str(pdf_subset.get("subset_name") or pdf_subset.get("subset_file") or ""),
        "collection_name": collection_name or vector_store.get("collection_name", ""),
        "chunk_strategy": str(chunking.get("strategy", "")),
        "chunk_size": str(fixed.get("chunk_size", "")),
        "chunk_overlap": str(fixed.get("chunk_overlap", "")),
        "retrieval_mode": str(retrieval.get("mode", "")),
        "top_k_initial": str(retrieval.get("top_k_initial", "")),
        "top_k_final": str(retrieval.get("top_k_final", "")),
        "rerank_enabled": str(bool(rerank.get("enabled", True))).lower(),
        "rerank_model": str(rerank.get("model", "")),
        "llm_model": str(generation.get("model", "")),
        "judge_model": str(generation.get("model", "gpt-4o-mini")),
        "recall_at_5": _format_metric(summary.get("recall_at_5")),
        "recall_at_10": _format_metric(summary.get("recall_at_10")),
        "recall_at_20": _format_metric(summary.get("recall_at_20")),
        "mrr": _format_metric(summary.get("mrr")),
        "context_precision_at_5": _format_metric(summary.get("context_precision_at_5")),
        "correctness": "0.00",
        "groundedness": "0.00",
        "relevance": "0.00",
        "e2e_success_rate": _format_metric(e2e_success_rate),
        "notes": notes or str(run_config.get("notes", "")),
    }


def append_registry_row(
    row: dict[str, str],
    registry_path: Path | None = None,
) -> Path:
    registry_path = registry_path or DEFAULT_REGISTRY_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not registry_path.exists() or registry_path.stat().st_size == 0
    with registry_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in REGISTRY_COLUMNS})

    return registry_path


def _extract_retrieved_source_ids(record: dict) -> list[str]:
    ids: list[str] = []
    documents = record.get("documents") or []
    for document in documents:
        metadata = document.get("metadata") or {}
        pdf_id = metadata.get("pdf_id")
        if pdf_id:
            ids.append(str(pdf_id))
            continue
        title = metadata.get("title")
        if title:
            ids.append(str(title))

    if ids:
        return ids

    sources = record.get("sources") or []
    for source in sources:
        pdf_id = source.get("pdf_id") or source.get("title")
        if pdf_id:
            ids.append(str(pdf_id))
    return ids


def _rank_of_gold(record: dict) -> int | None:
    flags = build_relevance_flags(
        record.get("documents") or [],
        gold_context=record.get("gold_context", ""),
        gold_source_id=record.get("gold_source_id", ""),
    )
    for rank, hit in enumerate(flags, start=1):
        if hit:
            return rank
    return None


def classify_failure(record: dict, per_query: dict) -> tuple[str, str | None]:
    if record.get("error"):
        return "E1_runtime_error", None
    if per_query.get("recall_at_5", 0.0) == 0.0:
        return "R1_not_retrieved", None
    if per_query.get("mrr", 0.0) < 1.0:
        return "R2_suboptimal_rank", None
    return "PASS", None


def build_failure_record(
    run_config: dict,
    record: dict,
    per_query: dict,
) -> dict:
    primary, secondary = classify_failure(record, per_query)
    return {
        "run_id": run_config.get("run_id"),
        "qa_index": record.get("qa_index"),
        "question": record.get("question", ""),
        "source_id": record.get("gold_source_id", ""),
        "gold_answer": record.get("gold_answer", ""),
        "retrieved_source_ids": _extract_retrieved_source_ids(record),
        "retrieval": {
            "recall_at_5": per_query.get("recall_at_5", 0.0),
            "recall_at_10": per_query.get("recall_at_10", 0.0),
            "rank_of_gold": _rank_of_gold(record),
            "mrr": per_query.get("mrr", mrr(
                build_relevance_flags(
                    record.get("documents") or [],
                    gold_context=record.get("gold_context", ""),
                    gold_source_id=record.get("gold_source_id", ""),
                )
            )),
        },
        "answer_eval": {
            "correctness": None,
            "groundedness": None,
            "relevance": None,
        },
        "failure_type_primary": primary,
        "failure_type_secondary": secondary,
        "notes": record.get("error") or "",
    }


def build_failure_records(
    run_config: dict,
    records: list[dict],
    per_query_metrics: list[dict],
    *,
    include_pass: bool = False,
) -> list[dict]:
    per_query_by_index = {
        item.get("qa_index"): item for item in per_query_metrics if item.get("qa_index") is not None
    }
    failures = []

    for record in records:
        qa_index = record.get("qa_index")
        per_query = per_query_by_index.get(qa_index, {})
        failure_record = build_failure_record(run_config, record, per_query)
        if include_pass or failure_record["failure_type_primary"] != "PASS":
            failures.append(failure_record)

    return failures


def default_failures_path(run_config: dict, project_root: Path | None = None) -> Path:
    project_root = project_root or Path(".")
    run_id = run_config.get("run_id", "unnamed_run")
    return project_root / "experiment" / "runs" / run_id / "failures.jsonl"


def write_failures_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def persist_run_artifacts(
    run_config: dict,
    records: list[dict],
    per_query_metrics: list[dict],
    summary: dict,
    *,
    project_root: Path,
    collection_name: str | None = None,
    registry_path: Path | None = None,
    failures_path: Path | None = None,
    notes: str = "",
) -> tuple[Path, Path]:
    registry_row = build_registry_row(
        run_config,
        summary,
        collection_name=collection_name,
        notes=notes,
    )
    registry_file = append_registry_row(
        registry_row,
        registry_path=registry_path or (project_root / DEFAULT_REGISTRY_PATH),
    )

    failures = build_failure_records(run_config, records, per_query_metrics)
    failures_file = write_failures_jsonl(
        failures_path or default_failures_path(run_config, project_root),
        failures,
    )
    return registry_file, failures_file
