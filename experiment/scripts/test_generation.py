"""
Manual smoke test for retrieval + rerank + generation_service.

Usage (from project root):
    python experiment/scripts/test_generation.py
    python experiment/scripts/test_generation.py --collection test
    python experiment/scripts/test_generation.py --query "Your question here"
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
    parser = argparse.ArgumentParser(
        description="Test retrieval + rerank + LLM generation."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Query string")
    parser.add_argument("--collection", default="test", help="Qdrant collection name")
    parser.add_argument(
        "--top-k-initial",
        type=int,
        default=20,
        help="Number of passages to retrieve before rerank",
    )
    parser.add_argument(
        "--top-k-final",
        type=int,
        default=5,
        help="Number of passages to keep after rerank",
    )
    parser.add_argument(
        "--answer-chars",
        type=int,
        default=500,
        help="Characters of answer text to print",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from backend.app.core.config import settings
    from backend.app.services.generation_service import (
        build_sources,
        format_context,
        generate_answer,
        get_llm,
        get_parser,
        get_prompt,
    )
    from backend.app.services.retrieval_service import retrieve_embedding
    from backend.app.services.rerank_service import get_ranker, rerank_passages

    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""
    settings.validate()

    passages = retrieve_embedding(
        args.query,
        args.collection,
        top_k=args.top_k_initial,
    )
    ranker = get_ranker()
    reranked = rerank_passages(
        args.query,
        passages,
        top_k=args.top_k_final,
        ranker=ranker,
    )

    if not reranked:
        print("No passages returned. Check Qdrant is running and the collection has data.")
        return 0

    context = format_context(reranked)
    prompt = get_prompt()
    llm = get_llm()
    parser = get_parser()
    answer = generate_answer(args.query, context, prompt, llm, parser)
    sources = build_sources(reranked)

    print(f"collection: {args.collection}")
    print(f"retrieved_count: {len(passages)}")
    print(f"reranked_count: {len(reranked)}")
    print(f"sources_count: {len(sources)}")
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
