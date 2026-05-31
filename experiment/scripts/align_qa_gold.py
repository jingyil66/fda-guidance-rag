"""
Align QA gold labels to pdf_id + page using the same chunking as experiment ingest.

Usage (from project root):
    python experiment/scripts/align_qa_gold.py
    python experiment/scripts/align_qa_gold.py --dataset experiment/subsets/qa_gold_dev.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align QA gold context to pdf_id + page.")
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        default=None,
        help="QA JSON file to update (repeatable)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "subset_200",
    )
    parser.add_argument(
        "--subset-file",
        type=Path,
        default=PROJECT_ROOT / "experiment" / "subsets" / "subset_200_v1.json",
    )
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "experiment" / "subsets" / "qa_alignment_v1.json",
    )
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_subset_pdf_ids(path: Path) -> set[str]:
    payload = load_json(path)
    return {str(pdf_id) for pdf_id in (payload.get("pdf_ids") or [])}


def main() -> int:
    args = parse_args()
    datasets = args.dataset or [
        PROJECT_ROOT / "experiment" / "subsets" / "qa_gold_dev.json",
        PROJECT_ROOT / "experiment" / "subsets" / "qa_gold_test.json",
    ]

    from experiment.gold_alignment import align_qa_dataset, build_local_chunks

    if not args.data_dir.exists():
        print(f"Data directory not found: {args.data_dir}")
        return 1

    pdf_ids = load_subset_pdf_ids(args.subset_file) if args.subset_file.exists() else None
    print(f"Building local chunks from {args.data_dir} ...")
    chunks = build_local_chunks(
        args.data_dir,
        pdf_ids=pdf_ids,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"chunk corpus size: {len(chunks)}")

    manifest = {
        "name": "qa_alignment_v1",
        "created_at": date.today().isoformat(),
        "data_dir": str(args.data_dir),
        "subset_file": str(args.subset_file),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "datasets": {},
    }

    for dataset_path in datasets:
        if not dataset_path.exists():
            print(f"Dataset not found: {dataset_path}")
            return 1

        rows = load_json(dataset_path)
        if not isinstance(rows, list):
            print(f"Dataset must be a JSON list: {dataset_path}")
            return 1

        aligned_rows, summary = align_qa_dataset(rows, chunks)
        dataset_path.write_text(
            json.dumps(aligned_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["datasets"][str(dataset_path)] = summary
        print(
            f"{dataset_path.name}: resolved {summary['resolved']}/{summary['total']}, "
            f"unresolved {summary['unresolved']}"
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote alignment manifest to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
