#!/usr/bin/env python3
"""Add bird photographs to the training set as labelled background.

DUT Anti-UAV contains no birds, so a detector trained on it has never been
shown one and told it is not a drone. `benchmarks/eval_hard_negatives.py`
measures what that costs: roughly a quarter of bird images produce a UAV
detection at the operating threshold.

The fix is training data, not threshold tuning. YOLO treats an image with an
empty label file as background: everything in it is a negative, and the loss
penalises any detection on it. This script fetches bird images and installs
them as a second training directory, leaving the original split untouched so
the earlier run stays reproducible.

**The negatives come from the Open Images *train* split; the benchmark scores
the *validation* split.** They are disjoint by construction, and the script
refuses to run if that ever stops being true -- measuring a false-positive rate
on images the model was trained not to fire on would be self-congratulation
rather than evaluation.

Usage:
    python tools/add_hard_negatives.py --count 600
    python tools/train_uav.py --data data/dut_antiuav/dut_antiuav_negatives.yaml \\
        --weights runs/train/uav_yolov8n/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Mirrors benchmarks/eval_hard_negatives.py. Kept in sync deliberately: the
#: negatives trained on and the negatives scored against must mean the same
#: thing, or the measurement afterwards is not about the same question.
NEGATIVE_CLASSES = ["Bird"]
EXCLUDED_CLASSES = ["Airplane", "Helicopter", "Missile", "Rocket"]

#: The split the benchmark scores. Never draw training negatives from it.
EVALUATION_SPLIT = "validation"
TRAINING_SPLIT = "train"


def fetch_birds(out_dir: Path, count: int, split: str) -> list[Path]:
    """Download bird images from Open Images, skipping anything aircraft-like."""
    if split == EVALUATION_SPLIT:
        sys.exit(
            f"refusing to draw training negatives from the {EVALUATION_SPLIT!r} split: "
            "that is what benchmarks/eval_hard_negatives.py scores"
        )

    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        sys.exit("fiftyone is required: pip install fiftyone")

    print(f"downloading up to {count} Open Images V7 bird images from the {split!r} split ...")
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["detections"],
        classes=NEGATIVE_CLASSES,
        max_samples=count,
        dataset_name=f"uavtrack-train-negatives-{count}",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    kept: list[Path] = []
    skipped = 0
    for sample in dataset:
        labels = sample["ground_truth"]
        present = {detection.label for detection in (labels.detections if labels else [])}
        # A photograph containing an aeroplane is not a negative for a detector
        # whose whole job is finding things in the sky.
        if present & set(EXCLUDED_CLASSES):
            skipped += 1
            continue

        source = Path(sample.filepath)
        destination = out_dir / f"oi_{source.name}"
        if not destination.exists():
            shutil.copy2(source, destination)
        kept.append(destination)

    print(f"kept {len(kept)} images, skipped {skipped} containing {'/'.join(EXCLUDED_CLASSES)}")
    fo.delete_dataset(dataset.name)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--count", type=int, default=600, help="bird images to fetch")
    parser.add_argument("--yolo-root", type=Path, default=Path("data/dut_antiuav/yolo"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/dut_antiuav/dut_antiuav_negatives.yaml"),
        help="dataset descriptor to write",
    )
    parser.add_argument("--split", default=TRAINING_SPLIT, help=argparse.SUPPRESS)
    args = parser.parse_args()

    yolo_root = (ROOT / args.yolo_root).resolve()
    if not (yolo_root / "images" / "train").is_dir():
        sys.exit(f"{yolo_root} does not look prepared; run tools/prepare_dataset.py first")

    images_dir = yolo_root / "images" / "train_negatives"
    labels_dir = yolo_root / "labels" / "train_negatives"
    kept = fetch_birds(images_dir, args.count, args.split)
    if not kept:
        sys.exit("no negatives were kept; nothing to do")

    # An empty label file is the whole mechanism: YOLO reads it as "this image
    # contains no objects", and any detection on it becomes loss.
    labels_dir.mkdir(parents=True, exist_ok=True)
    for image in kept:
        (labels_dir / f"{image.stem}.txt").write_text("", encoding="utf-8")

    positives = len(list((yolo_root / "images" / "train").iterdir()))
    fraction = len(kept) / (positives + len(kept))

    descriptor = f"""# Training split plus bird photographs as background images.
#
# Written by tools/add_hard_negatives.py. The negatives are Open Images V7
# {args.split!r}; benchmarks/eval_hard_negatives.py scores the
# {EVALUATION_SPLIT!r} split, so the two are disjoint.
#
# {len(kept)} negatives against {positives} positives ({fraction:.1%} of the set).
path: {yolo_root.as_posix()}
train:
  - images/train
  - images/train_negatives
val: images/val_small
test: images/test
names:
  0: uav
"""
    out = ROOT / args.out
    out.write_text(descriptor, encoding="utf-8")

    manifest = images_dir.parent.parent / "negatives_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source": f"Open Images V7 {args.split}, class {'/'.join(NEGATIVE_CLASSES)}",
                "excluded_classes": EXCLUDED_CLASSES,
                "evaluation_split": EVALUATION_SPLIT,
                "negatives": len(kept),
                "positives": positives,
                "negative_fraction": fraction,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n{len(kept)} background images -> {images_dir}")
    print(f"wrote {out}")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
