"""The contract every detection backend implements."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected object in original-frame pixel coordinates.

    Attributes:
        x1, y1, x2, y2: Axis-aligned box corners, top-left origin.
        score: Confidence in ``[0, 1]``.
        class_id: Index into the backend's label list.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int = 0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def centre(self) -> tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    def as_xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def as_xywh(self) -> np.ndarray:
        """Centre-form ``[cx, cy, w, h]`` - the state the Kalman filter tracks."""
        cx, cy = self.centre
        return np.array([cx, cy, self.width, self.height], dtype=np.float32)

    @classmethod
    def from_xyxy(cls, box: np.ndarray, score: float, class_id: int = 0) -> Detection:
        return cls(
            float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score), class_id
        )


class DetectorBackend(abc.ABC):
    """Base class for detection backends.

    Subclasses do the work in :meth:`infer`; the shared machinery here exists so
    that swapping a Hailo-8L for ONNX Runtime is a one-line configuration change
    and nothing downstream notices.
    """

    #: Human-readable label per class index.
    labels: tuple[str, ...] = ("uav",)

    @property
    @abc.abstractmethod
    def input_size(self) -> tuple[int, int]:
        """Network input as ``(width, height)``."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier used in benchmark result files."""

    @abc.abstractmethod
    def infer(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR frame.

        Args:
            frame: HWC uint8 BGR image of any size.

        Returns:
            Detections in original-frame pixel coordinates, highest score first.
        """

    def warmup(self, iterations: int = 3) -> None:
        """Run a few throwaway inferences.

        First-call costs (lazy kernel compilation, memory pool growth, NPU
        power-state transitions) otherwise land in the first measured frame and
        skew both benchmarks and the control loop's startup behaviour.
        """
        width, height = self.input_size
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.infer(dummy)

    def close(self) -> None:  # noqa: B027 - intentionally optional, not abstract
        """Release backend resources. Safe to call more than once.

        Deliberately concrete and empty: most backends hold nothing that needs
        releasing, and forcing every one of them to write an empty override
        would be noise. Backends that do own resources override it.
        """

    def __enter__(self) -> DetectorBackend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
