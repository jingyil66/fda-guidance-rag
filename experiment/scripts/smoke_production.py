"""
Production smoke test: Qdrant collection + RAG pipeline (+ optional Flask /ask).

Usage (from project root):
    python experiment/scripts/smoke_production.py
    python experiment/scripts/smoke_production.py --min-points 100
    python experiment/scripts/smoke_production.py --api-url http://127.0.0.1:5000/ask
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
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
    parser = argparse.ArgumentParser(description="Smoke test production RAG defaults.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument(
        "--min-points",
        type=int,
        default=1,
        help="Minimum Qdrant points required in the collection",
    )
    parser.add_argument(
        "--api-url",
        default="",
        help="Optional Flask /ask URL (POST JSON {\"query\": ...})",
    )
    return parser.parse_args()


def check_qdrant(collection_name: str, qdrant_url: str, min_points: int) -> int:
    from qdrant_client import QdrantClient

    client = QdrantClient(qdrant_url)
    if not client.collection_exists(collection_name):
        print(f"FAIL: collection missing: {collection_name}")
        return 1

    points = client.get_collection(collection_name).points_count
    print(f"collection: {collection_name}")
    print(f"points: {points}")
    if points < min_points:
        print(f"FAIL: expected at least {min_points} points, found {points}")
        return 1
    return 0


def check_pipeline(query: str, collection_name: str) -> int:
    import os

    from backend.app.core.config import settings
    from backend.app.services.pipeline_service import run_rag_pipeline

    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""
    settings.validate()

    result = run_rag_pipeline(query, config={"collection_name": collection_name})
    answer = (result.get("answer") or "").strip()
    sources = result.get("sources") or []

    print(f"pipeline sources_count: {len(sources)}")
    print(f"pipeline answer_chars: {len(answer)}")
    if not answer or answer.lower() == "query is empty":
        print("FAIL: empty pipeline answer")
        return 1
    if not sources:
        print("FAIL: pipeline returned no sources")
        return 1

    print("pipeline answer preview:", answer[:200].replace("\n", " "))
    return 0


def check_api(api_url: str, query: str) -> int:
    payload = json.dumps({"query": query}).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        print(f"SKIP api: {exc}")
        return 0

    if not body.get("success"):
        print(f"FAIL: api response not successful: {body}")
        return 1
    if not (body.get("answer") or "").strip():
        print("FAIL: api returned empty answer")
        return 1
    if not body.get("sources"):
        print("FAIL: api returned no sources")
        return 1

    print(f"api sources_count: {len(body['sources'])}")
    return 0


def main() -> int:
    args = parse_args()

    from backend.app.core.config import settings

    collection_name = settings.DEFAULT_QDRANT_COLLECTION
    qdrant_url = settings.DEFAULT_QDRANT_URL

    code = check_qdrant(collection_name, qdrant_url, args.min_points)
    if code != 0:
        return code

    code = check_pipeline(args.query, collection_name)
    if code != 0:
        return code

    if args.api_url:
        code = check_api(args.api_url, args.query)
        if code != 0:
            return code

    print("PASS: production smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
