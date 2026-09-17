#!/usr/bin/env python3
"""Convert DUT Anti-UAV from Pascal VOC into YOLO and COCO layouts.

Produces, under ``--root``:

* ``yolo/images/<split>`` and ``yolo/labels/<split>`` - symlinked (or copied)
  images plus YOLO label files, ready for ``tools/train_uav.py``.
* ``coco/<split>.json`` - COCO ground truth used by ``benchmarks/eval_detection.py``.
* ``dut_antiuav.yaml`` - the Ultralytics dataset descriptor.
* ``stats.json`` - object-size statistics, quoted in docs/dataset.md.

Images are symlinked by default so the 1.2 GB of JPEGs is not duplicated. On
Windows without Developer Mode, symlink creation fails and the script falls
back to hardlinks, then to copies.

Usage:
    python tools/prepare_dataset.py --root data/dut_antiuav
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uavtrack.data.voc import parse_voc_xml, to_coco, to_yolo_lines, write_coco  # noqa: E402

CLASS_TO_ID = {"UAV": 0}
SPLITS = ("train", "val", "test")

# COCO small/medium/large thresholds, in pixels of box area.
COCO_SMALL_MAX = 32**2
COCO_MEDIUM_MAX = 96**2


def link_or_copy(src: Path, dst: Path) -> str:
    """Materialise ``dst`` from ``src`` as cheaply as the filesystem allows.

    Returns:
        Which strategy was used: ``"symlink"``, ``"hardlink"`` or ``"copy"``.
    """
    if dst.exists():
        return "exists"
    try:
        os.symlink(src, dst)
        return "symlink"
    except (OSError, NotImplementedError, AttributeError):
        pass
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def convert_split(root: Path, split: str) -> dict:
    """Convert one split and return its statistics."""
    img_dir = root / split / "img"
    xml_dir = root / split / "xml"
    if not img_dir.is_dir() or not xml_dir.is_dir():
        sys.exit(f"missing {img_dir} or {xml_dir}; run tools/fetch_dut_antiuav.py first")

    yolo_images = root / "yolo" / "images" / split
    yolo_labels = root / "yolo" / "labels" / split
    yolo_images.mkdir(parents=True, exist_ok=True)
    yolo_labels.mkdir(parents=True, exist_ok=True)

    annotations = []
    areas: list[float] = []
    relative_areas: list[float] = []
    empty_images = 0
    strategies: dict[str, int] = {}

    for xml_path in sorted(xml_dir.glob("*.xml")):
        image_path = img_dir / f"{xml_path.stem}.jpg"
        if not image_path.exists():
            print(f"[warn] no image for {xml_path.name}, skipping")
            continue

        annotation = parse_voc_xml(xml_path)
        # Trust the file on disk over the <filename> element: some VOC exports
        # carry a stale filename, and the COCO ids must match the actual images.
        annotation.filename = image_path.name
        annotations.append(annotation)

        strategy = link_or_copy(image_path.resolve(), yolo_images / image_path.name)
        strategies[strategy] = strategies.get(strategy, 0) + 1

        lines = to_yolo_lines(annotation, CLASS_TO_ID)
        # An empty label file is how YOLO encodes a pure-background image; the
        # file must exist or Ultralytics reports the image as unlabelled.
        (yolo_labels / f"{xml_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        if not annotation.boxes:
            empty_images += 1
        for box in annotation.boxes:
            areas.append(box.area)
            relative_areas.append(box.area / (annotation.width * annotation.height))

    coco = to_coco(annotations, CLASS_TO_ID, description=f"DUT Anti-UAV detection - {split}")
    write_coco(root / "coco" / f"{split}.json", coco)

    small = sum(1 for a in areas if a < COCO_SMALL_MAX)
    medium = sum(1 for a in areas if COCO_SMALL_MAX <= a < COCO_MEDIUM_MAX)
    large = sum(1 for a in areas if a >= COCO_MEDIUM_MAX)

    stats = {
        "images": len(annotations),
        "objects": len(areas),
        "background_images": empty_images,
        "objects_small": small,
        "objects_medium": medium,
        "objects_large": large,
        "relative_area_min": min(relative_areas) if relative_areas else None,
        "relative_area_max": max(relative_areas) if relative_areas else None,
        "relative_area_median": (
            sorted(relative_areas)[len(relative_areas) // 2] if relative_areas else None
        ),
        "image_link_strategy": strategies,
    }
    print(
        f"[ok  ] {split}: {stats['images']} images, {stats['objects']} objects "
        f"(small {small} / medium {medium} / large {large}), "
        f"{empty_images} background images"
    )
    return stats


def write_yaml(root: Path, path: Path) -> None:
    """Write the Ultralytics dataset descriptor."""
    path.write_text(
        "# DUT Anti-UAV detection subset, converted by tools/prepare_dataset.py\n"
        f"path: {root.resolve().as_posix()}/yolo\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: uav\n",
        encoding="utf-8",
    )
    print(f"[ok  ] wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("data/dut_antiuav"))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    args = parser.parse_args()

    stats = {split: convert_split(args.root, split) for split in args.splits}
    write_yaml(args.root, args.root / "dut_antiuav.yaml")
    (args.root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[ok  ] wrote {args.root / 'stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
