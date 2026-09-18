"""Letterbox pre-processing and the inverse coordinate mapping.

A detector sees a square, padded, resized copy of the frame; everything
downstream works in original frame pixels. Keeping the forward transform and
its inverse in one object is what stops the two from drifting apart -- a
mismatch here shows up as boxes that are subtly offset, which is easy to miss
by eye and fatal for a closed-loop pointing system.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Letterbox:
    """Forward and inverse mapping between frame pixels and network input.

    Attributes:
        scale: Factor applied to the original frame before padding.
        pad_x: Padding added on the *left* edge, in network-input pixels.
        pad_y: Padding added on the *top* edge, in network-input pixels.
        net_w: Network input width.
        net_h: Network input height.
        src_w: Original frame width.
        src_h: Original frame height.
    """

    scale: float
    pad_x: float
    pad_y: float
    net_w: int
    net_h: int
    src_w: int
    src_h: int

    def to_frame(self, boxes: np.ndarray) -> np.ndarray:
        """Map xyxy boxes from network-input pixels back to frame pixels.

        Args:
            boxes: ``(n, 4)`` array of ``[x1, y1, x2, y2]`` in network pixels.

        Returns:
            ``(n, 4)`` array in original-frame pixels, clipped to the frame.
        """
        if boxes.size == 0:
            return boxes.reshape(0, 4).astype(np.float32)
        out = boxes.astype(np.float32).copy()
        out[:, [0, 2]] -= self.pad_x
        out[:, [1, 3]] -= self.pad_y
        out /= self.scale
        # Assign the clipped result back explicitly: fancy indexing returns a
        # copy, so `np.clip(..., out=out[:, [0, 2]])` would clip a temporary and
        # throw it away, leaving the boxes unclipped.
        out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, self.src_w - 1)
        out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, self.src_h - 1)
        return out

    def normalised_to_frame(self, boxes: np.ndarray) -> np.ndarray:
        """Map xyxy boxes expressed in ``[0, 1]`` of the network input.

        Hailo's on-chip NMS emits normalised coordinates, so this is the entry
        point used by :mod:`uavtrack.detect.hailo_backend`.
        """
        if boxes.size == 0:
            return boxes.reshape(0, 4).astype(np.float32)
        scaled = boxes.astype(np.float32).copy()
        scaled[:, [0, 2]] *= self.net_w
        scaled[:, [1, 3]] *= self.net_h
        return self.to_frame(scaled)


#: Downscaling kernels that :func:`letterbox` will accept by name.
#:
#: Only the shrinking kernel is selectable. Upscaling is always bilinear, since
#: ``INTER_AREA`` degenerates to nearest-neighbour when magnifying.
DOWNSCALE_KERNELS = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
}


def letterbox(
    image: np.ndarray,
    net_size: tuple[int, int],
    pad_value: int = 114,
    downscale: str = "area",
) -> tuple[np.ndarray, Letterbox]:
    """Resize ``image`` into ``net_size`` preserving aspect ratio, centre-padded.

    Args:
        image: HWC BGR or RGB frame.
        net_size: ``(width, height)`` expected by the network.
        pad_value: Constant used for the padded border. 114 matches the value
            Ultralytics uses at train time, so the padded border looks like
            what the network saw during training.
        downscale: Kernel used when shrinking -- ``"area"`` or ``"linear"``.
            The default is worth roughly two points of ``AP_small`` over
            bilinear; ``"linear"`` exists so that result can be reproduced, and
            so this implementation can be scored against Ultralytics on equal
            terms. See docs/benchmarks.md.

    Returns:
        The padded image and the :class:`Letterbox` describing the transform.
    """
    try:
        shrink_kernel = DOWNSCALE_KERNELS[downscale]
    except KeyError:
        raise ValueError(
            f"unknown downscale kernel {downscale!r}; expected one of "
            f"{', '.join(sorted(DOWNSCALE_KERNELS))}"
        ) from None
    src_h, src_w = image.shape[:2]
    net_w, net_h = net_size
    scale = min(net_w / src_w, net_h / src_h)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))

    # The shrinking kernel anti-aliases; it measurably preserves small,
    # low-contrast targets such as a distant UAV. Magnifying is always bilinear.
    interp = shrink_kernel if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    pad_w, pad_h = net_w - new_w, net_h - new_h
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3
    )
    return padded, Letterbox(scale, float(left), float(top), net_w, net_h, src_w, src_h)
