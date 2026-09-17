#!/usr/bin/env python3
"""False-positive rate on bird images: the failure mode that matters most.

Birds are the canonical false positive for ground-to-air UAV detection. Similar
apparent size at range, same sky background, similar motion. A detector with an
excellent AP on a UAV dataset can still be useless outdoors if it fires on every
pigeon, and no UAV-only benchmark can tell you that -- DUT Anti-UAV contains no
labelled birds at all.

This measures it directly: run the detector over images that are *guaranteed to
contain no UAV*, and count how often it reports one. The metrics are the
per-image false-positive rate at the operating threshold, and the threshold that
would be needed to reach a target rate.

The negatives come from Open Images V7, which is CC-BY licensed, scriptable to
download, and carries per-image labels -- so images that also contain aircraft,
helicopters or missiles can be excluded, since those are not false positives in
any interesting sense.

Usage:
    python benchmarks/eval_hard_negatives.py \\
        --model runs/train/uav_yolov8n/weights/best.pt --tag yolov8n-fp32
    python benchmarks/eval_hard_negatives.py --model ... --images data/hard_negatives/birds
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from uavtrack.detect.factory import build_detector  # noqa: E402

# Open Images classes that count as negatives, and classes that disqualify an
# image from being one.
NEGATIVE_CLASSES = ["Bird"]
EXCLUDED_CLASSES = ["Airplane", "Helicopter", "Missile", "Rocket"]

# Thresholds at which the false-positive rate is reported.
THRESHOLD_SWEEP = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def fetch_open_images(out_dir: Path, max_samples: int) -> Path:
    """Download bird images from Open Images V7 that contain no aircraft.

    Returns:
        The directory holding the downloaded images.
    """
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
        from fiftyone import ViewField as F
    except ImportError:  # pragma: no cover - environment dependent
        sys.exit(
            "fiftyone is required to fetch hard negatives: pip install fiftyone\n"
            "Alternatively point --images at a directory you have prepared yourself."
        )

    print(f"downloading up to {max_samples} Open Images V7 bird images ...")
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=NEGATIVE_CLASSES,
        max_samples=max_samples,
        dataset_name=f"uavtrack-negatives-{max_samples}",
    )

    # Drop anything that also contains an aircraft: a detector firing on a real
    # airplane is behaving sensibly, not failing.
    view = dataset
    for excluded in EXCLUDED_CLASSES:
        view = view.match(~F("ground_truth.detections.label").contains(excluded))
    print(f"  {len(dataset)} downloaded, {len(view)} after excluding {EXCLUDED_CLASSES}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for sample in view:
        target = out_dir / Path(sample.filepath).name
        if not target.exists():
            target.write_bytes(Path(sample.filepath).read_bytes())

    fo.delete_dataset(dataset.name)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help=".pt, .onnx or .hef")
    parser.add_argument("--backend", default=None, choices=["ultralytics", "onnx", "hailo"])
    parser.add_argument("--images", type=Path, default=Path("data/hard_negatives/birds"))
    parser.add_argument("--fetch", action="store_true", help="download the negatives first")
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument(
        "--operating-threshold",
        type=float,
        default=0.35,
        help="the threshold configs/ actually deploys at",
    )
    parser.add_argument(
        "--target-fp-rate",
        type=float,
        default=0.01,
        help="find the threshold reaching this per-image false-positive rate",
    )
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    if args.fetch or not args.images.is_dir():
        fetch_open_images(args.images, args.max_samples)

    paths = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        sys.exit(f"no images in {args.images}; re-run with --fetch")

    # Collect at a permissive threshold once, then sweep offline: re-running
    # inference per threshold would measure the same thing eleven times.
    detector = build_detector(args.model, backend=args.backend, conf_threshold=0.01)
    tag = args.tag or Path(args.model).stem
    print(f"model    {args.model}")
    print(f"backend  {detector.name}")
    print(f"images   {len(paths)} bird images from {args.images}")
    detector.warmup()

    best_scores: list[float] = []
    for index, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        detections = detector.infer(frame)
        best_scores.append(max((d.score for d in detections), default=0.0))
        if (index + 1) % 200 == 0:
            print(f"  {index + 1}/{len(paths)}", flush=True)
    detector.close()

    total = len(best_scores)
    sweep = []
    for threshold in THRESHOLD_SWEEP:
        false_positives = sum(1 for s in best_scores if s >= threshold)
        sweep.append(
            {
                "threshold": threshold,
                "images_with_false_positive": false_positives,
                "false_positive_rate": round(false_positives / total, 5),
            }
        )

    # Lowest threshold in the sweep that reaches the target rate.
    reaching_target = next(
        (row["threshold"] for row in sweep if row["false_positive_rate"] <= args.target_fp_rate),
        None,
    )
    operating = min(sweep, key=lambda row: abs(row["threshold"] - args.operating_threshold))

    payload = {
        "tag": tag,
        "kind": "measured",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": str(args.model),
        "backend": detector.name,
        "negative_source": "Open Images V7 validation, class Bird",
        "excluded_classes": EXCLUDED_CLASSES,
        "images": total,
        "operating_threshold": args.operating_threshold,
        "false_positive_rate_at_operating_threshold": operating["false_positive_rate"],
        "threshold_for_target_fp_rate": reaching_target,
        "target_fp_rate": args.target_fp_rate,
        "sweep": sweep,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"hard_negatives_{tag}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"{'threshold':>10}{'false positives':>18}{'rate':>9}")
    print("-" * 37)
    for row in sweep:
        print(
            f"{row['threshold']:>10.2f}{row['images_with_false_positive']:>18d}"
            f"{row['false_positive_rate'] * 100:>8.2f}%"
        )
    print()
    print(
        f"at the deployed threshold {args.operating_threshold}: "
        f"{operating['false_positive_rate'] * 100:.2f}% of bird images produce a UAV detection"
    )
    if reaching_target is not None:
        print(f"threshold needed for a {args.target_fp_rate * 100:.1f}% rate: {reaching_target}")
    else:
        print(f"no threshold in the sweep reaches a {args.target_fp_rate * 100:.1f}% rate")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
