"""
Run a configured RAG experiment over a QA dataset.

Usage (from project root):
    python experiment/run_experiment.py --config experiment/configs/run_001.json
    python experiment/run_experiment.py --config experiment/configs/run_001.json --limit 3
    python experiment/run_experiment.py --config experiment/configs/run_001.json --collection test
    python experiment/run_experiment.py --score-only
    python experiment/run_experiment.py --score-only --skip-answer-eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configured RAG experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "experiment" / "configs" / "run_001.json",
        help="Path to run config JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of QA rows to run (0 = all)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Override vector_store.collection_name from config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL output path (defaults to experiment/runs/<run_id>/results.jsonl)",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=200,
        help="Characters of answer text to print per row",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip pipeline run and score an existing results.jsonl",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Results JSONL to score when using --score-only",
    )
    parser.add_argument(
        "--skip-answer-eval",
        action="store_true",
        help="Skip GPT-as-judge answer metrics (retrieval only)",
    )
    parser.add_argument(
        "--answer-limit",
        type=int,
        default=0,
        help="Max records to judge when scoring answers (0 = all)",
    )
    parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="Do not append a row to registry.csv (useful for repeat-run checks)",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_qa_dataset(run_config: dict) -> list[dict]:
    dataset_path = resolve_project_path(run_config["qa_dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"QA dataset not found: {dataset_path}")

    data = load_json(dataset_path)
    if not isinstance(data, list):
        raise ValueError(f"QA dataset must be a JSON list: {dataset_path}")
    return data


def build_pipeline_config(run_config: dict, collection_override: str | None = None) -> dict:
    vector_store = run_config.get("vector_store") or {}
    retrieval = run_config.get("retrieval") or {}
    generation = run_config.get("generation") or {}
    rerank = retrieval.get("rerank") or {}

    retrieval_mode = retrieval.get("mode", "embedding_plus_rerank")
    rerank_enabled = bool(rerank.get("enabled", True))
    if retrieval_mode == "embedding_only":
        rerank_enabled = False

    collection_name = collection_override or vector_store.get("collection_name", "test")

    return {
        "collection_name": collection_name,
        "qdrant_url": vector_store.get("url", "http://localhost:6333"),
        "top_k_initial": retrieval.get("top_k_initial", 20),
        "top_k_final": retrieval.get("top_k_final", 5),
        "rerank_enabled": rerank_enabled,
        "embedding_model": retrieval.get("embedding_model", "text-embedding-3-small"),
        "llm_model": generation.get("model", "gpt-4o-mini"),
        "rerank_model": rerank.get("model", "ms-marco-MiniLM-L-12-v2"),
        "rerank_passage_format": rerank.get("passage_format", "raw"),
        "context_passage_format": generation.get("passage_format", "legacy"),
    }


def default_run_dir(run_config: dict) -> Path:
    run_id = run_config.get("run_id", "unnamed_run")
    return PROJECT_ROOT / "experiment" / "runs" / run_id


def default_config_snapshot_path(run_config: dict) -> Path:
    return default_run_dir(run_config) / "config.json"


def default_manifest_path(run_config: dict) -> Path:
    return default_run_dir(run_config) / "run_manifest.json"


def enrich_record_gold_fields(record: dict, gold_row: dict, fields: dict) -> None:
    context_field = fields.get("context_field", "context")
    source_id_field = fields.get("source_id_field", "source_id")
    answer_field = fields.get("answer_field", "answer")

    record.setdefault("gold_context", gold_row.get(context_field, ""))
    record.setdefault("gold_source_id", gold_row.get(source_id_field, ""))
    record.setdefault("gold_answer", gold_row.get(answer_field, ""))
    record.setdefault("gold_pdf_id", gold_row.get("gold_pdf_id", ""))
    if "gold_page" not in record and gold_row.get("gold_page") is not None:
        record["gold_page"] = gold_row.get("gold_page")


def resolve_retrieval_match_strategy(records: list[dict]) -> str:
    if any(record.get("gold_pdf_id") for record in records):
        return "pdf_id_page_with_text_fallback"
    return "gold_context_overlap_or_source_id"


def save_run_provenance(
    run_config: dict,
    *,
    config_source: Path,
    collection_name: str | None = None,
) -> tuple[Path, Path]:
    run_dir = default_run_dir(run_config)
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot = dict(run_config)
    snapshot["_provenance"] = {
        "config_source": str(config_source),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "collection_name": collection_name or "",
    }
    config_path = default_config_snapshot_path(run_config)
    write_json(config_path, snapshot)

    pdf_subset = run_config.get("pdf_subset") or {}
    split_manifest = PROJECT_ROOT / "experiment" / "subsets" / "qa_split_v1.json"
    alignment_manifest = PROJECT_ROOT / "experiment" / "subsets" / "qa_alignment_v1.json"
    manifest = {
        "run_id": run_config.get("run_id"),
        "saved_at": snapshot["_provenance"]["saved_at"],
        "qa_dataset_path": run_config.get("qa_dataset_path"),
        "pdf_subset": pdf_subset,
        "collection_name": collection_name or (run_config.get("vector_store") or {}).get("collection_name"),
        "qa_split_manifest": str(split_manifest) if split_manifest.exists() else "",
        "qa_alignment_manifest": str(alignment_manifest) if alignment_manifest.exists() else "",
        "retrieval_match_strategy": resolve_retrieval_match_strategy([]),
    }
    manifest_path = default_manifest_path(run_config)
    write_json(manifest_path, manifest)
    return config_path, manifest_path


def default_output_path(run_config: dict) -> Path:
    run_id = run_config.get("run_id", "unnamed_run")
    return PROJECT_ROOT / "experiment" / "runs" / run_id / "results.jsonl"


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def default_metrics_path(run_config: dict) -> Path:
    run_id = run_config.get("run_id", "unnamed_run")
    return PROJECT_ROOT / "experiment" / "runs" / run_id / "metrics.json"


def ensure_openai_env() -> None:
    from backend.app.core.config import settings

    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""
    settings.validate()


def print_metrics(summary: dict) -> None:
    print("--- retrieval metrics ---")
    print(f"query_count: {summary.get('query_count', 0)}")
    print(f"mrr: {summary.get('mrr', 0.0):.4f}")
    for key in sorted(summary):
        if key.startswith("recall_at_") or key.startswith("context_precision_at_"):
            print(f"{key}: {summary[key]:.4f}")

    answer_keys = ("correctness", "groundedness", "relevance", "e2e_success_rate")
    if any(summary.get(key) is not None for key in answer_keys):
        print("--- answer metrics ---")
        for key in answer_keys:
            if summary.get(key) is not None:
                print(f"{key}: {summary[key]:.4f}")


def run_answer_evaluation(
    run_config: dict,
    records: list[dict],
    *,
    answer_limit: int = 0,
) -> tuple[list[dict], dict[str, float | None]]:
    from experiment.answer_evaluators import (
        AnswerJudge,
        aggregate_answer_metrics,
        evaluate_answer_record,
    )

    ensure_openai_env()

    evaluation = run_config.get("evaluation") or {}
    generation = run_config.get("generation") or {}
    metrics = evaluation.get("answer_metrics") or [
        "correctness",
        "groundedness",
        "relevance",
    ]
    judge_model = evaluation.get("judge_model") or generation.get("model", "gpt-4o-mini")

    target_records = records
    if answer_limit > 0:
        target_records = records[:answer_limit]

    judge = AnswerJudge(model=judge_model)
    per_query = [
        evaluate_answer_record(record, judge, metrics=metrics)
        for record in target_records
    ]
    return per_query, aggregate_answer_metrics(per_query, metrics)


def score_results(
    run_config: dict,
    results_path: Path,
    *,
    collection_name: str | None = None,
    skip_answer_eval: bool = False,
    answer_limit: int = 0,
    skip_registry: bool = False,
    config_source: Path | None = None,
) -> int:
    from experiment.answer_evaluators import (
        compute_e2e_success_rate,
        merge_retrieval_and_answer_metrics,
    )
    from experiment.evaluators import evaluate_records
    from experiment.io_utils import persist_run_artifacts

    if not results_path.exists():
        print(f"Results not found: {results_path}")
        return 1

    records = read_jsonl(results_path)
    fields = run_config.get("qa_dataset_fields") or {}

    try:
        qa_rows = load_qa_dataset(run_config)
    except FileNotFoundError:
        qa_rows = []

    for record in records:
        qa_index = record.get("qa_index")
        if qa_index is None or qa_index >= len(qa_rows):
            continue
        enrich_record_gold_fields(record, qa_rows[qa_index], fields)

    match_strategy = resolve_retrieval_match_strategy(records)
    if config_source is not None:
        config_path, manifest_path = save_run_provenance(
            run_config,
            config_source=config_source,
            collection_name=collection_name,
        )
        manifest = load_json(manifest_path)
        if isinstance(manifest, dict):
            manifest["retrieval_match_strategy"] = match_strategy
            write_json(manifest_path, manifest)
        print(f"Wrote config snapshot to {config_path}")
        print(f"Wrote run manifest to {manifest_path}")

    k_list = (run_config.get("evaluation") or {}).get("k_list") or [5, 10, 20]
    per_query, summary = evaluate_records(records, k_list=k_list)

    if not skip_answer_eval:
        evaluation = run_config.get("evaluation") or {}
        answer_metrics = evaluation.get("answer_metrics") or []
        if answer_metrics:
            print("Running answer evaluation (GPT-as-judge)...")
            answer_rows, answer_summary = run_answer_evaluation(
                run_config,
                records,
                answer_limit=answer_limit,
            )
            per_query = merge_retrieval_and_answer_metrics(per_query, answer_rows)
            summary.update(answer_summary)
            rule = evaluation.get("e2e_success_rule", "correctness_and_groundedness_pass")
            summary["e2e_success_rate"] = compute_e2e_success_rate(
                per_query,
                rule=rule,
            )

    metrics_path = default_metrics_path(run_config)
    write_json(
        metrics_path,
        {
            "run_id": run_config.get("run_id"),
            "results_path": str(results_path),
            "match_strategy": match_strategy,
            "summary": summary,
            "per_query": per_query,
            "answer_eval_enabled": not skip_answer_eval,
        },
    )

    print_metrics(summary)
    print(f"match_strategy: {match_strategy}")
    print(f"Wrote metrics to {metrics_path}")

    registry_path, failures_path = persist_run_artifacts(
        run_config,
        records,
        per_query,
        summary,
        project_root=PROJECT_ROOT,
        collection_name=collection_name,
        skip_registry=skip_registry,
    )
    if registry_path:
        print(f"Appended registry row to {registry_path}")
    else:
        print("Skipped registry append")
    print(f"Wrote failures to {failures_path}")
    return 0


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        return 1

    run_config = load_json(args.config)
    if not isinstance(run_config, dict):
        print("Run config must be a JSON object.")
        return 1

    output_path = args.output or default_output_path(run_config)

    if args.score_only:
        results_path = args.results or output_path
        pipeline_config = build_pipeline_config(run_config, args.collection)
        return score_results(
            run_config,
            results_path,
            collection_name=pipeline_config["collection_name"],
            skip_answer_eval=args.skip_answer_eval,
            answer_limit=args.answer_limit,
            skip_registry=args.skip_registry,
            config_source=args.config,
        )

    from backend.app.services.pipeline_service import run_rag_pipeline

    ensure_openai_env()

    fields = run_config.get("qa_dataset_fields") or {}
    question_field = fields.get("question_field", "question")
    answer_field = fields.get("answer_field", "answer")
    context_field = fields.get("context_field", "context")
    source_id_field = fields.get("source_id_field", "source_id")

    qa_rows = load_qa_dataset(run_config)
    if args.limit > 0:
        qa_rows = qa_rows[: args.limit]

    pipeline_config = build_pipeline_config(run_config, args.collection)
    save_run_provenance(
        run_config,
        config_source=args.config,
        collection_name=pipeline_config["collection_name"],
    )

    print(f"run_id: {run_config.get('run_id', 'unnamed_run')}")
    print(f"stage: {run_config.get('stage', 'unknown')}")
    print(f"qa_count: {len(qa_rows)}")
    print(f"collection: {pipeline_config['collection_name']}")
    print(f"retrieval_mode: {run_config.get('retrieval', {}).get('mode', 'unknown')}")
    print(f"output: {output_path}")
    print()

    results: list[dict] = []

    for index, row in enumerate(qa_rows):
        question = row.get(question_field, "")
        print(f"[{index + 1}/{len(qa_rows)}] {question[:120]}...")

        try:
            pipeline_result = run_rag_pipeline(question, pipeline_config)
        except Exception as exc:
            print(f"  error: {exc}")
            record = {
                "qa_index": index,
                "question": question,
                "gold_answer": row.get(answer_field, ""),
                "gold_context": row.get(context_field, ""),
                "gold_source_id": row.get(source_id_field, ""),
                "gold_pdf_id": row.get("gold_pdf_id", ""),
                "gold_page": row.get("gold_page"),
                "error": str(exc),
                "answer": "",
                "sources": [],
                "documents": [],
                "source_count": 0,
            }
            results.append(record)
            continue

        answer = pipeline_result.get("answer", "")
        sources = pipeline_result.get("sources") or []
        documents = pipeline_result.get("documents") or []
        preview = answer[: args.preview_chars]
        if len(answer) > args.preview_chars:
            preview += "..."

        print(f"  sources: {len(sources)}")
        print(f"  answer: {preview}")

        record = {
            "qa_index": index,
            "question": question,
            "gold_answer": row.get(answer_field, ""),
            "gold_context": row.get(context_field, ""),
            "gold_source_id": row.get(source_id_field, ""),
            "gold_pdf_id": row.get("gold_pdf_id", ""),
            "gold_page": row.get("gold_page"),
            "answer": answer,
            "sources": sources,
            "documents": documents,
            "source_count": len(sources),
        }
        results.append(record)
        print()

    write_jsonl(output_path, results)
    print(f"Wrote {len(results)} records to {output_path}")
    print()
    return score_results(
        run_config,
        output_path,
        collection_name=pipeline_config["collection_name"],
        skip_answer_eval=args.skip_answer_eval,
        answer_limit=args.answer_limit,
        skip_registry=args.skip_registry,
        config_source=args.config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
