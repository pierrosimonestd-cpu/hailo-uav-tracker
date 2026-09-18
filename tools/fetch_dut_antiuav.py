#!/usr/bin/env python3
"""Download the DUT Anti-UAV detection benchmark.

DUT Anti-UAV is the public benchmark this project reports detection accuracy
on. The detection subset is 10,000 manually annotated stills split
5,200 / 2,600 / 2,200 (train / val / test), single class ``UAV``, Pascal VOC XML
annotations.

    Zhao, Zhang, Li and Wang, "Vision-based Anti-UAV Detection and Tracking",
    IEEE Transactions on Intelligent Transportation Systems, 2022.
    https://github.com/wangdongdut/DUT-Anti-UAV

The authors host the archives on Google Drive, so this script needs ``gdown``.
Nothing is vendored into the repository: run this once, then
``tools/prepare_dataset.py``.

Usage:
    python tools/fetch_dut_antiuav.py --out data/dut_antiuav
    python tools/fetch_dut_antiuav.py --out data/dut_antiuav --splits test
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

# Google Drive file IDs published in the DUT-Anti-UAV README.
DRIVE_IDS = {
    "train": "1RVsSGPUKTdmoyoPTBTWwroyulLek1eTj",
    "val": "1333uEQfGuqTKslRkkeLSCxylh6AQ0X6n",
    "test": "1L1zeW1EMDLlXHClSDcCjl3rs_A6sVai0",
}

# The tracking subset ships as two archives, frames and annotations, rather than
# one per split. It is fetched separately because it feeds a different benchmark
# (benchmarks/eval_tracking.py) and most users will not want the extra gigabyte.
TRACKING_DRIVE_IDS = {
    "img": "1dlSPDggg6TRFMcC1jlYIJxxzUQS1mIh9",
    "gt": "16PE3tBhT0lUGZLA8-zIRYvNUvxfhFZJq",
}

# Image counts stated in the paper; used as an integrity check after extraction.
EXPECTED_IMAGES = {"train": 5200, "val": 2600, "test": 2200}


def download_split(split: str, archive_dir: Path) -> Path:
    """Fetch one split archive, skipping the download if it is already present."""
    try:
        import gdown
    except ImportError:  # pragma: no cover - environment dependent
        sys.exit("gdown is required: pip install 'uavtrack[bench]'")

    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"{split}.zip"
    if archive.exists():
        print(f"[skip] {archive} already downloaded")
        return archive

    print(f"[get ] {split}: downloading from Google Drive ...")
    gdown.download(id=DRIVE_IDS[split], output=str(archive), quiet=False)
    if not archive.exists():
        sys.exit(
            f"download of {split} failed. Google Drive rate-limits large files; "
            "retry later or use the Baidu mirror listed in the DUT-Anti-UAV README."
        )
    return archive


def extract_split(archive: Path, out_dir: Path, split: str) -> None:
    """Extract ``archive`` so that images land in ``out_dir/<split>/img``."""
    target = out_dir / split
    if (target / "img").is_dir():
        print(f"[skip] {target} already extracted")
        return

    print(f"[open] extracting {archive.name} ...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out_dir)

    # The archives already contain a top-level directory named after the split.
    # Tolerate a differently named one rather than failing late.
    if not (target / "img").is_dir():
        candidates = [p for p in out_dir.iterdir() if p.is_dir() and (p / "img").is_dir()]
        if len(candidates) != 1:
            sys.exit(f"unexpected archive layout in {archive}; expected a single <split>/img/ tree")
        shutil.move(str(candidates[0]), str(target))


def verify_split(out_dir: Path, split: str) -> bool:
    """Compare the extracted file counts with the counts published in the paper."""
    images = sorted((out_dir / split / "img").glob("*.jpg"))
    annotations = sorted((out_dir / split / "xml").glob("*.xml"))
    expected = EXPECTED_IMAGES[split]

    ok = len(images) == expected and len(annotations) == expected
    status = "ok  " if ok else "WARN"
    print(
        f"[{status}] {split}: {len(images)} images, "
        f"{len(annotations)} annotations (expected {expected})"
    )
    return ok


def fetch_tracking(out_dir: Path, keep_archives: bool) -> bool:
    """Fetch and extract the 20 annotated tracking sequences.

    Two archives rather than one per split: frames and ground truth are
    published separately.
    """
    try:
        import gdown
    except ImportError:  # pragma: no cover - environment dependent
        sys.exit("gdown is required: pip install 'uavtrack[bench]'")

    archive_dir = out_dir / ".archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for name, drive_id in TRACKING_DRIVE_IDS.items():
        archive = archive_dir / f"{name}.zip"
        if not archive.exists():
            print(f"[get ] tracking {name}: downloading from Google Drive ...")
            gdown.download(id=drive_id, output=str(archive), quiet=False)
        if not archive.exists():
            sys.exit(f"download of tracking {name} failed; see the Baidu mirror in the README")

        print(f"[open] extracting {archive.name} ...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        if not keep_archives:
            archive.unlink(missing_ok=True)

    sequences = sorted(out_dir.rglob("groundtruth*.txt"))
    frames = sum(1 for _ in out_dir.rglob("*.jpg"))
    status = "ok  " if sequences and frames else "WARN"
    print(f"[{status}] tracking: {len(sequences)} sequences, {frames} frames")
    return bool(sequences and frames)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/dut_antiuav"), help="destination directory"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=[*sorted(DRIVE_IDS), "tracking"],
        help=(
            "splits to fetch. The test split alone reproduces the detection table; "
            "'tracking' fetches the 20 video sequences for benchmarks/eval_tracking.py "
            "and should be given its own --out directory"
        ),
    )
    parser.add_argument(
        "--keep-archives", action="store_true", help="do not delete the .zip files after extraction"
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    archive_dir = args.out / ".archives"

    all_ok = True
    for split in args.splits:
        if split == "tracking":
            all_ok &= fetch_tracking(args.out, args.keep_archives)
            continue
        archive = download_split(split, archive_dir)
        extract_split(archive, args.out, split)
        all_ok &= verify_split(args.out, split)
        if not args.keep_archives:
            archive.unlink(missing_ok=True)

    print(f"\nDataset ready at {args.out}")
    if "tracking" in args.splits:
        print("Next: python benchmarks/eval_tracking.py --root", args.out)
    else:
        print("Next: python tools/prepare_dataset.py --root", args.out)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
