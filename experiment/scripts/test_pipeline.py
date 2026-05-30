"""
Manual smoke test for pipeline_service.run_rag_pipeline.

Usage (from project root):
    python experiment/scripts/test_pipeline.py
    python experiment/scripts/test_pipeline.py --collection test
    python experiment/scripts/test_pipeline.py --no-rerank
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_QUERY = (
    "How should sponsors integrate feedback from the FDA on HF study protocols "
    "into their development timelines, and what specific documentation is required "
    "for submission in their NDA, BLA, or ANDA applications?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test full RAG pipeline.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Query string")
    parser.add_argument("--collection", default="test", help="Qdrant collection name")
    parser.add_argument("--top-k-initial", type=int, default=20)
    parser.add_argument("--top-k-final", type=int, default=5)
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip rerank and use top retrieval results only",
    )
    parser.add_argument("--answer-chars", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from backend.app.core.config import settings
    from backend.app.services.pipeline_service import run_rag_pipeline

    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""
    settings.validate()

    result = run_rag_pipeline(
        args.query,
        config={
            "collection_name": args.collection,
            "top_k_initial": args.top_k_initial,
            "top_k_final": args.top_k_final,
            "rerank_enabled": not args.no_rerank,
        },
    )

    answer = result.get("answer") or ""
    sources = result.get("sources") or []
    documents = result.get("documents") or []

    print(f"collection: {args.collection}")
    print(f"rerank_enabled: {not args.no_rerank}")
    print(f"sources_count: {len(sources)}")
    print(f"documents_count: {len(documents)}")
    print()
    print("--- answer ---")
    preview = answer[: args.answer_chars]
    if len(answer) > args.answer_chars:
        preview += "..."
    print(preview)
    print()
    print("--- sources ---")
    for source in sources:
        print(
            f"title={source['title']} page={source['page']} pdf_id={source['pdf_id']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
