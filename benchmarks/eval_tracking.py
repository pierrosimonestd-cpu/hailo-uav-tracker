#!/usr/bin/env python3
"""Tracking accuracy on the DUT Anti-UAV tracking sequences.

Detection accuracy on stills says nothing about whether a lock survives a
target crossing a rooftop, dimming against cloud, or briefly leaving frame. That
is what the tracker exists for, and it needs sequences to measure.

**This is a harder task than the SOT baselines published with the dataset.**
A single-object tracker is handed the ground-truth box in frame one and has only
to follow it. This pipeline is never told where the target is: it detects, then
associates, then chooses a primary target, with no initialisation at all. The
numbers here are therefore not directly comparable to a SOT leaderboard, and
they are not presented as if they were.

Metrics, following the single-object tracking convention the dataset's own paper
uses, plus two the convention leaves out:

``success_auc``
    Area under the success curve: mean over IoU thresholds 0 to 1 of the
    fraction of frames whose predicted box overlaps ground truth by at least
    that much. The standard summary number.
``precision_20px``
    Fraction of frames whose predicted centre is within 20 pixels of truth.
``recall``
    Fraction of frames where the pipeline reported *any* target. A tracker that
    is accurate on the 30% of frames it bothers to answer is not a good tracker,
    and the success curve alone hides that.
``reacquisitions``
    Number of times the primary target's identity changed. Each one is a moment
    where a turret would have swung somewhere else.

Usage:
    python benchmarks/eval_tracking.py \\
        --model runs/train/uav_yolov8n/weights/best.pt \\
        --root data/dut_antiuav_tracking
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from uavtrack.detect.factory import build_detector  # noqa: E402
from uavtrack.track.bytetrack import ByteTracker  # noqa: E402
from uavtrack.track.selector import TargetSelector  # noqa: E402

SUCCESS_THRESHOLDS = np.linspace(0.0, 1.0, 101)
PRECISION_THRESHOLD_PX = 20.0


def load_groundtruth(path: Path) -> list[tuple[float, float, float, float] | None]:
    """Read a ``groundtruth.txt`` of one ``x,y,w,h`` box per frame.

    An all-zero or ``NaN`` row means the target is absent in that frame, which
    the DUT sequences use for out-of-view stretches. Those frames are excluded
    from the overlap metrics -- scoring a tracker for failing to find something
    that is not there measures nothing.
    """
    boxes: list[tuple[float, float, float, float] | None] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().replace("\t", ",").replace(" ", ",")
        if not line:
            continue
        parts = [p for p in line.split(",") if p != ""]
        try:
            x, y, w, h = (float(p) for p in parts[:4])
        except ValueError:
            boxes.append(None)
            continue
        if not all(np.isfinite([x, y, w, h])) or w <= 0 or h <= 0:
            boxes.append(None)
        else:
            boxes.append((x, y, w, h))
    return boxes


def find_sequences(root: Path) -> list[tuple[str, Path, Path]]:
    """Locate ``(name, frame_dir, groundtruth)`` triples under ``root``."""
    sequences = []
    for gt_path in sorted(root.rglob("groundtruth*.txt")):
        directory = gt_path.parent
        candidates = [directory, directory / "img", directory / "imgs"]
        for candidate in candidates:
            frames = sorted(candidate.glob("*.jpg")) if candidate.is_dir() else []
            if frames:
                sequences.append((directory.name, candidate, gt_path))
                break
    return sequences


def iou(a: tuple[float, float, float, float], b: np.ndarray) -> float:
    """IoU between an xywh ground-truth box and an xyxy prediction."""
    ax1, ay1, aw, ah = a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bx2, by2 = b

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = aw * ah + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def run_sequence(detector, frame_dir: Path, truth, tracker_kwargs: dict, limit: int | None):
    """Run the detect-track-select chain over one sequence."""
    frames = sorted(frame_dir.glob("*.jpg"))
    if limit:
        frames = frames[:limit]

    probe = cv2.imread(str(frames[0]))
    if probe is None:
        return None
    height, width = probe.shape[:2]

    tracker = ByteTracker(**tracker_kwargs)
    selector = TargetSelector(frame_size=(width, height))
    dt = 1.0 / 30.0

    overlaps: list[float] = []
    centre_errors: list[float] = []
    answered = 0
    evaluated = 0
    reacquisitions = 0
    previous_id = None
    latencies: list[float] = []

    for index, path in enumerate(frames):
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        started = time.perf_counter()
        detections = detector.infer(frame)
        tracks = tracker.update(detections, dt)
        target = selector.select(tracks, dt)
        latencies.append((time.perf_counter() - started) * 1000.0)

        if target is not None:
            if previous_id is not None and target.track_id != previous_id:
                reacquisitions += 1
            previous_id = target.track_id

        gt_box = truth[index] if index < len(truth) else None
        if gt_box is None:
            continue  # target absent: nothing to score against
        evaluated += 1

        if target is None:
            overlaps.append(0.0)
            centre_errors.append(float("inf"))
            continue

        answered += 1
        overlaps.append(iou(gt_box, target.box_xyxy))
        gt_centre = (gt_box[0] + gt_box[2] / 2.0, gt_box[1] + gt_box[3] / 2.0)
        centre_errors.append(
            float(np.hypot(target.centre[0] - gt_centre[0], target.centre[1] - gt_centre[1]))
        )

    if evaluated == 0:
        return None

    overlap_array = np.asarray(overlaps)
    success = [(overlap_array >= t).mean() for t in SUCCESS_THRESHOLDS]
    within = [e for e in centre_errors if np.isfinite(e)]

    return {
        "frames": len(frames),
        "evaluated_frames": evaluated,
        "success_auc": float(np.mean(success)),
        "success_at_0_5": float((overlap_array >= 0.5).mean()),
        "precision_20px": float(
            sum(1 for e in centre_errors if e <= PRECISION_THRESHOLD_PX) / evaluated
        ),
        "recall": answered / evaluated,
        "mean_iou_when_answered": float(overlap_array[overlap_array > 0].mean())
        if np.any(overlap_array > 0)
        else 0.0,
        "median_centre_error_px": float(np.median(within)) if within else None,
        "reacquisitions": reacquisitions,
        "median_latency_ms": float(np.median(latencies)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help=".pt, .onnx or .hef")
    parser.add_argument("--backend", default=None, choices=["ultralytics", "onnx", "hailo"])
    parser.add_argument("--root", type=Path, default=Path("data/dut_antiuav_tracking"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--sequences", type=int, default=None, help="evaluate only the first N")
    parser.add_argument("--max-frames", type=int, default=None, help="cap frames per sequence")
    args = parser.parse_args()

    sequences = find_sequences(args.root)
    if not sequences:
        sys.exit(
            f"no sequences with a groundtruth file under {args.root}. "
            "Download the DUT Anti-UAV tracking subset first (see docs/dataset.md)."
        )
    if args.sequences:
        sequences = sequences[: args.sequences]

    detector = build_detector(args.model, backend=args.backend, conf_threshold=args.conf)
    tag = args.tag or Path(args.model).stem
    print(f"model      {args.model}")
    print(f"backend    {detector.name}")
    print(f"sequences  {len(sequences)}")
    detector.warmup()

    tracker_kwargs = {"high_threshold": 0.5, "low_threshold": 0.1, "min_hits": 3, "max_age": 30}

    per_sequence = {}
    for name, frame_dir, gt_path in sequences:
        truth = load_groundtruth(gt_path)
        result = run_sequence(detector, frame_dir, truth, tracker_kwargs, args.max_frames)
        if result is None:
            print(f"  {name}: skipped (no scorable frames)")
            continue
        per_sequence[name] = result
        print(
            f"  {name:<28} AUC {result['success_auc']:.3f}  "
            f"P@20px {result['precision_20px']:.3f}  "
            f"recall {result['recall']:.3f}  "
            f"reacq {result['reacquisitions']}",
            flush=True,
        )
    detector.close()

    if not per_sequence:
        sys.exit("no sequence produced scorable frames")

    def mean(key):
        return float(np.mean([r[key] for r in per_sequence.values()]))

    overall = {
        "sequences": len(per_sequence),
        "total_frames": sum(r["frames"] for r in per_sequence.values()),
        "success_auc": mean("success_auc"),
        "success_at_0_5": mean("success_at_0_5"),
        "precision_20px": mean("precision_20px"),
        "recall": mean("recall"),
        "reacquisitions_total": sum(r["reacquisitions"] for r in per_sequence.values()),
        "median_latency_ms": mean("median_latency_ms"),
    }

    payload = {
        "tag": tag,
        "kind": "measured",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": str(args.model),
        "backend": detector.name,
        "conf_threshold": args.conf,
        "tracker": tracker_kwargs,
        "note": (
            "Detect-and-track, with no ground-truth initialisation. Not directly "
            "comparable to single-object-tracking baselines, which are given the "
            "first-frame box."
        ),
        "overall": overall,
        "per_sequence": per_sequence,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"tracking_{tag}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"  success AUC      {overall['success_auc']:.3f}")
    print(f"  success @ 0.5    {overall['success_at_0_5']:.3f}")
    print(f"  precision @20px  {overall['precision_20px']:.3f}")
    print(f"  recall           {overall['recall']:.3f}")
    print(f"  reacquisitions   {overall['reacquisitions_total']}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
