"""
Build experiment/subsets/subset_200_v1.json from FDA metadata.

Stratified sampling over field_regulated_product_field and Final/Draft status.
Always includes all PDFs already present in the local data directory.

Usage (from project root):
    python experiment/scripts/build_subset_200.py
    python experiment/scripts/build_subset_200.py --data-dir data --target 200
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PDF_ID_PATTERN = re.compile(r"/media/(\d+)/download")
HTML_ENTITY_FIXES = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build subset_200_v1 PDF id list.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to metadata_with_summary.json (default: settings.OUTPUT_METADATA_JSON)",
    )
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiment" / "subsets" / "subset_200_v1.json",
    )
    return parser.parse_args()


def clean_text(value: str) -> str:
    text = value or ""
    for entity, char in HTML_ENTITY_FIXES.items():
        text = text.replace(entity, char)
    return text.strip()


def extract_pdf_id(item: dict) -> str | None:
    match = PDF_ID_PATTERN.search(item.get("field_associated_media_2") or "")
    return match.group(1) if match else None


def primary_product(item: dict) -> str:
    raw = clean_text(item.get("field_regulated_product_field") or "")
    if not raw:
        return "Unknown"
    return raw.split(",")[0].strip()


def guidance_status(item: dict) -> str:
    status = clean_text(item.get("field_final_guidance_1") or "")
    if status in {"Final", "Draft"}:
        return status
    return "Other"


def load_metadata_records(metadata_path: Path) -> dict[str, dict]:
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    records: dict[str, dict] = {}
    for item in data:
        pdf_id = extract_pdf_id(item)
        if pdf_id:
            records[pdf_id] = item
    return records


def local_pdf_ids(data_dir: Path) -> list[str]:
    return sorted(path.stem for path in data_dir.glob("*.pdf"))


def compute_product_targets(
    pool: dict[str, dict],
    *,
    target: int,
    reserved_ids: set[str],
) -> dict[str, int]:
    """Allocate remaining slots proportional to corpus, at least 1 per major bucket."""
    candidates = [item for pdf_id, item in pool.items() if pdf_id not in reserved_ids]
    counts = Counter(primary_product(item) for item in candidates)
    total = sum(counts.values())
    if total == 0:
        return {}

    raw = {product: (count / total) * target for product, count in counts.items()}
    allocated = {product: int(value) for product, value in raw.items()}
    remainder = target - sum(allocated.values())

    fractional = sorted(
        ((product, raw[product] - allocated[product]) for product in counts),
        key=lambda pair: pair[1],
        reverse=True,
    )
    for product, _ in fractional:
        if remainder <= 0:
            break
        allocated[product] += 1
        remainder -= 1

    for product in counts:
        allocated.setdefault(product, 0)
        if counts[product] > 0 and allocated[product] == 0:
            allocated[product] = 1

    while sum(allocated.values()) > target:
        product = max(allocated, key=lambda key: allocated[key])
        if allocated[product] > 1:
            allocated[product] -= 1
        else:
            break

    while sum(allocated.values()) < target:
        product = max(counts, key=lambda key: counts[key])
        allocated[product] = allocated.get(product, 0) + 1

    return allocated


def stratified_sample(
    pool: dict[str, dict],
    *,
    target: int,
    reserved_ids: set[str],
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    selected = sorted(reserved_ids)
    remaining_target = target - len(selected)
    if remaining_target <= 0:
        return selected[:target]

    product_targets = compute_product_targets(
        pool,
        target=remaining_target,
        reserved_ids=reserved_ids,
    )

    by_product: dict[str, list[str]] = defaultdict(list)
    for pdf_id, item in pool.items():
        if pdf_id in reserved_ids:
            continue
        by_product[primary_product(item)].append(pdf_id)

    for product in by_product:
        rng.shuffle(by_product[product])

    draft_target = max(1, round(remaining_target * 0.18))
    picked: list[str] = []

    def pick_from(product: str, want: int, *, status: str | None = None) -> int:
        added = 0
        for pdf_id in by_product.get(product, []):
            if pdf_id in picked or pdf_id in reserved_ids:
                continue
            if status and guidance_status(pool[pdf_id]) != status:
                continue
            picked.append(pdf_id)
            added += 1
            if added >= want:
                break
        return added

    for product, quota in sorted(product_targets.items(), key=lambda pair: pair[1], reverse=True):
        pick_from(product, quota)

    draft_shortfall = draft_target - sum(
        1 for pdf_id in picked if guidance_status(pool[pdf_id]) == "Draft"
    )
    if draft_shortfall > 0:
        draft_candidates = [
            pdf_id
            for pdf_id in pool
            if pdf_id not in reserved_ids
            and pdf_id not in picked
            and guidance_status(pool[pdf_id]) == "Draft"
        ]
        rng.shuffle(draft_candidates)
        for pdf_id in draft_candidates[:draft_shortfall]:
            if len(picked) >= remaining_target:
                break
            product = primary_product(pool[pdf_id])
            over_quota = [
                candidate
                for candidate in picked
                if primary_product(pool[candidate]) == product
            ]
            if over_quota:
                picked.remove(over_quota[0])
            picked.append(pdf_id)

    if len(picked) < remaining_target:
        leftovers = [
            pdf_id
            for pdf_id in pool
            if pdf_id not in reserved_ids and pdf_id not in picked
        ]
        rng.shuffle(leftovers)
        picked.extend(leftovers[: remaining_target - len(picked)])

    selected.extend(picked[:remaining_target])
    return sorted(set(selected))[:target]


def build_stats(pool: dict[str, dict], pdf_ids: list[str]) -> dict:
    products = Counter(primary_product(pool[pdf_id]) for pdf_id in pdf_ids if pdf_id in pool)
    statuses = Counter(guidance_status(pool[pdf_id]) for pdf_id in pdf_ids if pdf_id in pool)
    return {
        "count": len(pdf_ids),
        "by_product": dict(products.most_common()),
        "by_status": dict(statuses.most_common()),
    }


def main() -> int:
    args = parse_args()

    from backend.app.core.config import settings

    metadata_path = args.metadata or settings.OUTPUT_METADATA_JSON
    if not metadata_path.exists():
        print(f"Metadata not found: {metadata_path}")
        return 1

    pool = load_metadata_records(metadata_path)
    existing = local_pdf_ids(args.data_dir)
    missing_local = [pdf_id for pdf_id in existing if pdf_id not in pool]
    if missing_local:
        print(f"Warning: {len(missing_local)} local PDFs not found in metadata (first: {missing_local[:3]})")

    if len(existing) > args.target:
        print(f"Local PDF count ({len(existing)}) exceeds target ({args.target}).")
        return 1

    pdf_ids = stratified_sample(
        pool,
        target=args.target,
        reserved_ids=set(existing),
        seed=args.seed,
    )

    if len(pdf_ids) != args.target:
        print(f"Could only select {len(pdf_ids)} PDFs (target {args.target}).")
        return 1

    payload = {
        "name": "subset_200_v1",
        "version": "v1",
        "created_at": date.today().isoformat(),
        "selection_method": "stratified_by_product_and_status_seed42",
        "target_count": args.target,
        "random_seed": args.seed,
        "metadata_path": str(metadata_path.relative_to(PROJECT_ROOT))
        if metadata_path.is_relative_to(PROJECT_ROOT)
        else str(metadata_path),
        "local_data_dir": str(args.data_dir.relative_to(PROJECT_ROOT))
        if args.data_dir.is_relative_to(PROJECT_ROOT)
        else str(args.data_dir),
        "includes_existing_local_pdfs": True,
        "existing_local_pdf_count": len(existing),
        "pdf_ids": pdf_ids,
        "stats": build_stats(pool, pdf_ids),
        "notes": (
            "200 PDF experiment subset. pdf_id matches data/{pdf_id}.pdf and Qdrant metadata pdf_id. "
            "Always includes all PDFs currently in the local data directory."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"pdf_ids: {len(pdf_ids)}")
    print(f"existing_local_included: {len(set(existing) & set(pdf_ids))}/{len(existing)}")
    print(f"stats: {payload['stats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
