"""
Split evaluation/qa_dataset.json into dev and test gold sets.

Usage (from project root):
    python experiment/scripts/split_qa_dataset.py
    python experiment/scripts/split_qa_dataset.py --dev-size 30 --test-size 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split QA dataset into dev and test JSON files.")
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "qa_dataset.json",
    )
    parser.add_argument("--dev-size", type=int, default=30)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiment" / "subsets",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing split files",
    )
    return parser.parse_args()


def load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"QA dataset must be a JSON list: {path}")
    return data


def build_split_items(items: list[dict], indices: list[int]) -> list[dict]:
    split_items = []
    for index in indices:
        record = dict(items[index])
        record["qa_index"] = index
        split_items.append(record)
    return split_items


def main() -> int:
    args = parse_args()
    if not args.source.exists():
        print(f"Source QA dataset not found: {args.source}")
        return 1

    items = load_items(args.source)
    total = len(items)
    expected = args.dev_size + args.test_size
    if expected != total:
        print(
            f"Warning: dev_size ({args.dev_size}) + test_size ({args.test_size}) "
            f"= {expected}, but source has {total} items."
        )
        if expected > total:
            print("Reduce split sizes or expand the source dataset.")
            return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev_path = args.output_dir / "qa_gold_dev.json"
    test_path = args.output_dir / "qa_gold_test.json"
    manifest_path = args.output_dir / "qa_split_v1.json"

    if not args.force and dev_path.exists() and test_path.exists() and manifest_path.exists():
        print("Split files already exist. Use --force to regenerate.")
        print(f"  {dev_path}")
        print(f"  {test_path}")
        print(f"  {manifest_path}")
        return 0

    indices = list(range(total))
    rng = random.Random(args.seed)
    rng.shuffle(indices)

    dev_indices = sorted(indices[: args.dev_size])
    test_indices = sorted(indices[args.dev_size : args.dev_size + args.test_size])

    dev_items = build_split_items(items, dev_indices)
    test_items = build_split_items(items, test_indices)

    dev_path.write_text(json.dumps(dev_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    test_path.write_text(json.dumps(test_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "name": "qa_split_v1",
        "version": "v1",
        "created_at": date.today().isoformat(),
        "source_path": str(args.source.relative_to(PROJECT_ROOT))
        if args.source.is_relative_to(PROJECT_ROOT)
        else str(args.source),
        "random_seed": args.seed,
        "dev_size": args.dev_size,
        "test_size": args.test_size,
        "dev_indices": dev_indices,
        "test_indices": test_indices,
        "dev_path": str(dev_path.relative_to(PROJECT_ROOT))
        if dev_path.is_relative_to(PROJECT_ROOT)
        else str(dev_path),
        "test_path": str(test_path.relative_to(PROJECT_ROOT))
        if test_path.is_relative_to(PROJECT_ROOT)
        else str(test_path),
        "notes": "Dev set for tuning; test set is sealed for final evaluation only.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {dev_path} ({len(dev_items)} items)")
    print(f"Wrote {test_path} ({len(test_items)} items)")
    print(f"Wrote {manifest_path}")
    print(f"dev_indices: {dev_indices}")
    print(f"test_indices: {test_indices}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
