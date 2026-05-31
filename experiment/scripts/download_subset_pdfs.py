"""
Download PDFs listed in a subset JSON file.

Copies already-local PDFs from --source-dir when available, then downloads
the rest from FDA using metadata URLs.

Usage (from project root):
    python experiment/scripts/download_subset_pdfs.py
    python experiment/scripts/download_subset_pdfs.py --subset experiment/subsets/subset_200_v1.json
    python experiment/scripts/download_subset_pdfs.py --output-dir data/subset_200
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PDF_ID_PATTERN = re.compile(r"/media/(\d+)/download")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PDFs for an experiment subset.")
    parser.add_argument(
        "--subset",
        type=Path,
        default=PROJECT_ROOT / "experiment" / "subsets" / "subset_200_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "subset_200",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Existing local PDFs to copy before downloading",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Metadata JSON (default: settings.OUTPUT_METADATA_JSON)",
    )
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between downloads")
    return parser.parse_args()


def load_subset(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pdf_ids = payload.get("pdf_ids") or []
    if not pdf_ids:
        raise ValueError(f"No pdf_ids found in subset file: {path}")
    return [str(pdf_id) for pdf_id in pdf_ids]


def build_pdf_url_index(metadata_path: Path) -> dict[str, str]:
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    index: dict[str, str] = {}
    for item in data:
        media_html = item.get("field_associated_media_2") or ""
        match = PDF_ID_PATTERN.search(media_html)
        if not match:
            continue
        soup = BeautifulSoup(media_html, "html.parser")
        a_tag = soup.find("a")
        if not a_tag or not a_tag.get("href"):
            continue
        pdf_id = match.group(1)
        index[pdf_id] = "https://www.fda.gov" + a_tag["href"]
    return index


def copy_if_available(pdf_id: str, source_dir: Path, output_dir: Path) -> bool:
    source = source_dir / f"{pdf_id}.pdf"
    target = output_dir / f"{pdf_id}.pdf"
    if target.exists():
        return True
    if not source.exists():
        return False
    shutil.copy2(source, target)
    return True


def download_pdf(url: str, target: Path, *, headers: dict, timeout: int) -> None:
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    target.write_bytes(response.content)


def main() -> int:
    args = parse_args()

    from backend.app.core.config import settings

    metadata_path = args.metadata or settings.OUTPUT_METADATA_JSON
    if not args.subset.exists():
        print(f"Subset file not found: {args.subset}")
        return 1
    if not metadata_path.exists():
        print(f"Metadata not found: {metadata_path}")
        return 1

    pdf_ids = load_subset(args.subset)
    url_index = build_pdf_url_index(metadata_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    downloaded = 0
    failed: list[tuple[str, str]] = []

    for index, pdf_id in enumerate(pdf_ids, start=1):
        target = args.output_dir / f"{pdf_id}.pdf"
        if target.exists():
            skipped += 1
            continue

        if copy_if_available(pdf_id, args.source_dir, args.output_dir):
            copied += 1
            print(f"[{index}/{len(pdf_ids)}] Copied {pdf_id}.pdf")
            continue

        url = url_index.get(pdf_id)
        if not url:
            failed.append((pdf_id, "missing metadata URL"))
            print(f"[{index}/{len(pdf_ids)}] Missing URL for {pdf_id}")
            continue

        try:
            print(f"[{index}/{len(pdf_ids)}] Downloading {pdf_id}.pdf ...")
            download_pdf(url, target, headers=settings.HEADERS, timeout=args.timeout)
            downloaded += 1
            time.sleep(args.sleep)
        except Exception as exc:
            failed.append((pdf_id, str(exc)))
            print(f"[{index}/{len(pdf_ids)}] Failed {pdf_id}: {exc}")

    present = sorted(path.stem for path in args.output_dir.glob("*.pdf"))
    print("--- download summary ---")
    print(f"subset_target: {len(pdf_ids)}")
    print(f"present: {len(present)}")
    print(f"copied: {copied}")
    print(f"downloaded: {downloaded}")
    print(f"skipped_existing: {skipped}")
    print(f"failed: {len(failed)}")
    if failed:
        print("failed_ids:", [pdf_id for pdf_id, _ in failed[:10]])

    return 1 if len(present) < len(pdf_ids) else 0


if __name__ == "__main__":
    raise SystemExit(main())
