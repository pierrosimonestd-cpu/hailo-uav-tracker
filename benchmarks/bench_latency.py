#!/usr/bin/env python3
"""Per-stage latency and throughput of the live pipeline.

Reports the *distribution*, not the mean. A control loop is hurt by the tail:
one 200 ms frame in fifty does more damage to a pointing solution than a mean
that is 5 ms worse, because the tracker's extrapolation error grows with the gap
and the controller has no way to know a frame is late until it is. The p95 and
p99 are the numbers to quote, and ``control.latency_s`` in the config should be
set from the median-to-p95 range measured here.

Stages timed separately:

* ``preprocess`` - letterbox and layout conversion, on the host CPU.
* ``inference``  - the backend call. On Hailo this includes the PCIe round trip.
* ``postprocess`` - decode and NMS. Near zero when the HEF runs NMS on chip,
  which is the entire reason for compiling it that way.
* ``track``      - association, filtering and target selection.
* ``control``    - bearing reconstruction, PID, feed-forward.

Usage:
    python benchmarks/bench_latency.py --model models/uav_yolov8n_640.hef --frames 300
    python benchmarks/bench_latency.py --model models/uav_yolov8n_640.onnx \\
        --images data/dut_antiuav/test/img --frames 200
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

from uavtrack.control.gimbal import CameraGeometry, PanTiltController  # noqa: E402
from uavtrack.control.pid import PIDGains  # noqa: E402
from uavtrack.detect.factory import build_detector  # noqa: E402
from uavtrack.detect.preprocess import letterbox  # noqa: E402
from uavtrack.track.bytetrack import ByteTracker  # noqa: E402
from uavtrack.track.selector import TargetSelector  # noqa: E402

STAGES = ("preprocess", "inference", "postprocess", "track", "control")


def summarise(samples: list[float]) -> dict[str, float]:
    """Mean, median and tail percentiles of a latency sample, in ms."""
    if not samples:
        return {}
    ordered = sorted(samples)
    n = len(ordered)

    def at(fraction: float) -> float:
        return ordered[min(int(fraction * n), n - 1)]

    return {
        "mean": round(sum(ordered) / n, 3),
        "median": round(at(0.5), 3),
        "p95": round(at(0.95), 3),
        "p99": round(at(0.99), 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
    }


def load_frames(images: Path | None, count: int, width: int, height: int) -> list[np.ndarray]:
    """Load real frames if available, otherwise synthesise noise.

    Synthetic frames are a fallback for timing on a machine with no dataset.
    They are honest for measuring *compute* -- the network does the same work
    regardless of content -- but they exercise NMS differently, so real frames
    are preferred and the result file records which was used.
    """
    if images is not None and images.is_dir():
        paths = sorted(images.glob("*.jpg"))[:count]
        frames = [cv2.imread(str(p)) for p in paths]
        frames = [f for f in frames if f is not None]
        if frames:
            return frames

    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help=".pt, .onnx or .hef")
    parser.add_argument("--backend", default=None, choices=["ultralytics", "onnx", "hailo"])
    parser.add_argument("--images", type=Path, default=Path("data/dut_antiuav/test/img"))
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--threads", type=int, default=None, help="ONNX Runtime intra-op threads")
    args = parser.parse_args()

    kwargs: dict = {}
    if args.threads is not None:
        kwargs["intra_op_threads"] = args.threads
    detector = build_detector(args.model, backend=args.backend, **kwargs)
    tag = args.tag or Path(args.model).stem

    frames = load_frames(args.images, args.frames, args.width, args.height)
    synthetic = not (args.images and args.images.is_dir() and any(args.images.glob("*.jpg")))
    print(f"model    {args.model}")
    print(f"backend  {detector.name}  input {detector.input_size}")
    print(f"frames   {len(frames)} ({'synthetic' if synthetic else 'from ' + str(args.images)})")

    tracker = ByteTracker()
    selector = TargetSelector(frame_size=(args.width, args.height))
    controller = PanTiltController(
        geometry=CameraGeometry(args.width, args.height, 66.0),
        pan_gains=PIDGains(4.5, 0.5, 0.25),
        tilt_gains=PIDGains(4.5, 0.5, 0.25),
    )

    print(f"warmup   {args.warmup} iterations")
    detector.warmup(args.warmup)

    timings: dict[str, list[float]] = {stage: [] for stage in STAGES}
    end_to_end: list[float] = []
    dt = 1.0 / 30.0

    for index in range(args.frames):
        frame = frames[index % len(frames)]
        loop_start = time.perf_counter()

        # Preprocess is timed separately from inference by doing it twice: once
        # here to measure it, and once inside the backend which owns the real
        # path. The duplicate work is excluded from the end-to-end total below.
        start = time.perf_counter()
        letterbox(frame, detector.input_size)
        timings["preprocess"].append((time.perf_counter() - start) * 1000.0)
        preprocess_overhead = time.perf_counter() - start

        start = time.perf_counter()
        detections = detector.infer(frame)
        timings["inference"].append((time.perf_counter() - start) * 1000.0)

        # Whatever the backend does after the model call is already inside the
        # inference figure; this records the host-side share explicitly as zero
        # when NMS ran on chip.
        timings["postprocess"].append(
            0.0 if getattr(detector, "nms_on_chip", False) else float("nan")
        )

        start = time.perf_counter()
        tracks = tracker.update(detections, dt)
        target = selector.select(tracks, dt)
        timings["track"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        if target is not None:
            controller.update(target.centre, dt)
        timings["control"].append((time.perf_counter() - start) * 1000.0)

        total = time.perf_counter() - loop_start - preprocess_overhead
        end_to_end.append(total * 1000.0)

        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{args.frames}", flush=True)

    detector.close()

    stage_summary = {
        stage: summarise([v for v in values if not np.isnan(v)])
        for stage, values in timings.items()
    }
    total_summary = summarise(end_to_end)

    payload = {
        "tag": tag,
        "kind": "measured",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": str(args.model),
        "backend": detector.name,
        "input_size": list(detector.input_size),
        "frame_size": [args.width, args.height],
        "frames": args.frames,
        "synthetic_frames": synthetic,
        "stages_ms": stage_summary,
        "end_to_end_ms": total_summary,
        "throughput_fps": round(1000.0 / total_summary["median"], 2) if total_summary else None,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count": __import__("os").cpu_count(),
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"latency_{tag}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    header = f"{'stage':<12}{'median':>9}{'p95':>9}{'p99':>9}{'max':>9}"
    print(header)
    print("-" * len(header))
    for stage in STAGES:
        s = stage_summary.get(stage)
        if not s:
            continue
        print(f"{stage:<12}{s['median']:9.2f}{s['p95']:9.2f}{s['p99']:9.2f}{s['max']:9.2f}")
    print("-" * len(header))
    print(
        f"{'end to end':<12}{total_summary['median']:9.2f}{total_summary['p95']:9.2f}"
        f"{total_summary['p99']:9.2f}{total_summary['max']:9.2f}"
    )
    print(f"\nthroughput  {payload['throughput_fps']} fps (median)")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
