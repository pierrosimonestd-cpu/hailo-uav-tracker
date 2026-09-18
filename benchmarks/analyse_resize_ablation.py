#!/usr/bin/env python3
"""Split the INTER_AREA / INTER_LINEAR ablation by source resolution.

The aggregate ablation says area-averaging is worth about two points of
``AP_small``. That number is true and, on its own, misleading: it is an average
over a benchmark whose images arrive at four different resolutions, and the
kernel choice does nothing at all for some of them.

OpenCV's bilinear filter at an exact 2:1 downscale averages the same 2x2 block
that ``INTER_AREA`` does, so the two kernels agree there -- exactly on x86, and
to within one intensity level on aarch64, where the NEON path rounds
differently. At 3:1 it samples two columns of every three and the results
diverge sharply. DUT Anti-UAV is 61% 1920x1080 (3:1 into a 640 network) and 38%
1280x720 (exactly 2:1), so the whole of the aggregate gain has to come from the
former.

This script checks that rather than assuming it, by re-scoring the detections
both runs already wrote. No inference: it is a regrouping of existing results.

Usage:
    python benchmarks/analyse_resize_ablation.py
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Runs to compare, as ``label -> results tag``. Both must already exist; see
#: docs/benchmarks.md for the commands that produce them.
RUNS = {"area": "yolov8n-fp32-onnx", "linear": "ablation-inter-linear"}

#: Resolutions with fewer images than this are reported but not interpreted --
#: a handful of images cannot separate a real effect from sampling noise.
MIN_IMAGES_FOR_A_CLAIM = 50


def coco_ap(coco, detections, image_ids: list[int]) -> tuple[float, float]:
    """``(AP, AP_small)`` restricted to ``image_ids``."""
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = COCOeval(coco, detections, "bbox")
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return float(evaluator.stats[0]), float(evaluator.stats[3])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gt", type=Path, default=Path("data/dut_antiuav/coco/test.json"))
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    try:
        from pycocotools.coco import COCO
    except ImportError:
        sys.exit("pycocotools is required: pip install 'uavtrack[bench]'")

    if not args.gt.exists():
        sys.exit(f"{args.gt} not found; run tools/prepare_dataset.py first")

    detection_files = {}
    for label, tag in RUNS.items():
        path = args.results / tag / "detections.json"
        if not path.exists():
            sys.exit(
                f"{path} not found. Produce it with:\n"
                f"  python benchmarks/run_all.py --weights runs/train/uav_yolov8n/weights/best.pt"
            )
        detection_files[label] = path

    ground_truth = json.loads(args.gt.read_text(encoding="utf-8"))
    by_resolution: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for image in ground_truth["images"]:
        by_resolution[(image["width"], image["height"])].append(image["id"])

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(args.gt))
        loaded = {label: coco.loadRes(str(path)) for label, path in detection_files.items()}

    groups = []
    print(
        f"{'resolution':>12s} {'ratio':>6s} {'images':>7s} "
        f"{'AP area':>8s} {'AP linear':>10s} {'delta':>8s} {'AP_S delta':>11s}"
    )
    for (width, height), ids in sorted(by_resolution.items(), key=lambda kv: -len(kv[1])):
        area_ap, area_small = coco_ap(coco, loaded["area"], ids)
        linear_ap, linear_small = coco_ap(coco, loaded["linear"], ids)
        # The letterbox scale is set by the tighter axis, exactly as in
        # preprocess.letterbox; the ratio is its reciprocal.
        scale = min(640 / width, 640 / height)
        groups.append(
            {
                "width": width,
                "height": height,
                "downscale_ratio": round(1 / scale, 4),
                "images": len(ids),
                "ap_area": area_ap,
                "ap_linear": linear_ap,
                "ap_delta": area_ap - linear_ap,
                "ap_small_area": area_small,
                "ap_small_linear": linear_small,
                "ap_small_delta": area_small - linear_small,
                "interpretable": len(ids) >= MIN_IMAGES_FOR_A_CLAIM,
            }
        )
        print(
            f"{width}x{height:<6d} {1 / scale:6.3f} {len(ids):7d} "
            f"{area_ap:8.4f} {linear_ap:10.4f} {area_ap - linear_ap:+8.4f} "
            f"{area_small - linear_small:+11.4f}"
        )

    payload = {
        "kind": "measured",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": (
            "Regrouping of detections already written by the two ablation runs; no "
            "inference was repeated. A zero delta at an exact 2:1 downscale is not "
            "noise: OpenCV's bilinear filter averages the same 2x2 block that "
            "INTER_AREA does, so the kernels agree there to within rounding."
        ),
        "runs": {label: str(path) for label, path in detection_files.items()},
        "min_images_for_a_claim": MIN_IMAGES_FOR_A_CLAIM,
        "by_resolution": groups,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
    }

    out = args.out or args.results / "resize_ablation_by_resolution.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
