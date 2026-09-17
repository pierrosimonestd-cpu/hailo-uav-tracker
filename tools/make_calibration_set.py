#!/usr/bin/env python3
"""Build the calibration tensor the Hailo optimiser quantises against.

The Dataflow Compiler wants a single ``.npy`` of shape ``(n, H, W, 3)``,
uint8, NHWC. Two details decide whether the quantised model works:

* **The images must be pre-processed exactly as at inference time.** The
  optimiser measures activation ranges on these tensors; feed it raw,
  differently-scaled images and every scale factor in the network is set for a
  distribution the deployed model never sees.
* **They must come from the training split.** Calibrating on test images leaks
  the test set into the model.

Because ``tools/compile_hailo.sh`` puts the ``/255`` normalisation *on chip*,
this writes raw uint8 letterboxed images -- not floats.

Usage:
    python tools/make_calibration_set.py --out models/calibration.npy
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from uavtrack.detect.preprocess import letterbox  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--images", type=Path, default=Path("data/dut_antiuav/train/img"))
    parser.add_argument("--out", type=Path, default=Path("models/calibration.npy"))
    parser.add_argument("--count", type=int, default=1024, help="Hailo recommends at least 1024")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(args.images.glob("*.jpg"))
    if not paths:
        sys.exit(f"no images in {args.images}; run tools/fetch_dut_antiuav.py first")

    rng = random.Random(args.seed)
    rng.shuffle(paths)

    tensors: list[np.ndarray] = []
    for path in paths:
        if len(tensors) >= args.count:
            break
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        padded, _ = letterbox(frame, (args.imgsz, args.imgsz))
        # RGB, NHWC, uint8: the layout the compiler expects, and the layout the
        # HailoDetector feeds the device at runtime.
        tensors.append(padded[:, :, ::-1])

    if not tensors:
        sys.exit("no readable calibration images")

    stack = np.ascontiguousarray(np.stack(tensors)).astype(np.uint8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, stack)

    print(f"wrote {args.out}")
    print(f"  shape {stack.shape}, dtype {stack.dtype}, {stack.nbytes / 1e6:.1f} MB")
    if len(tensors) < args.count:
        print(f"  warning: only {len(tensors)} images available, wanted {args.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
