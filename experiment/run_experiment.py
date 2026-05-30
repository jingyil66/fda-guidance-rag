"""
Run a configured RAG experiment over a QA dataset.

Usage (from project root):
    python experiment/run_experiment.py --config experiment/configs/run_001.json
    python experiment/run_experiment.py --config experiment/configs/run_001.json --limit 3
    python experiment/run_experiment.py --config experiment/configs/run_001.json --collection test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    }


def default_output_path(run_config: dict) -> Path:
    run_id = run_config.get("run_id", "unnamed_run")
    return PROJECT_ROOT / "experiment" / "runs" / run_id / "results.jsonl"


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()

    from backend.app.core.config import settings
    from backend.app.services.pipeline_service import run_rag_pipeline

    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""
    settings.validate()

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        return 1

    run_config = load_json(args.config)
    if not isinstance(run_config, dict):
        print("Run config must be a JSON object.")
        return 1

    fields = run_config.get("qa_dataset_fields") or {}
    question_field = fields.get("question_field", "question")
    answer_field = fields.get("answer_field", "answer")
    source_id_field = fields.get("source_id_field", "source_id")

    qa_rows = load_qa_dataset(run_config)
    if args.limit > 0:
        qa_rows = qa_rows[: args.limit]

    pipeline_config = build_pipeline_config(run_config, args.collection)
    output_path = args.output or default_output_path(run_config)

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
                "gold_source_id": row.get(source_id_field, ""),
                "error": str(exc),
                "answer": "",
                "sources": [],
                "source_count": 0,
            }
            results.append(record)
            continue

        answer = pipeline_result.get("answer", "")
        sources = pipeline_result.get("sources") or []
        preview = answer[: args.preview_chars]
        if len(answer) > args.preview_chars:
            preview += "..."

        print(f"  sources: {len(sources)}")
        print(f"  answer: {preview}")

        record = {
            "qa_index": index,
            "question": question,
            "gold_answer": row.get(answer_field, ""),
            "gold_source_id": row.get(source_id_field, ""),
            "answer": answer,
            "sources": sources,
            "source_count": len(sources),
        }
        results.append(record)
        print()

    write_jsonl(output_path, results)
    print(f"Wrote {len(results)} records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
