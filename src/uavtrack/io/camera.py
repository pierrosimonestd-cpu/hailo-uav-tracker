"""Frame sources: Picamera2 on the Pi, OpenCV everywhere else, files for tests.

All sources expose the same iterator of ``(timestamp, frame)`` pairs. The
timestamp is the capture time from :func:`time.monotonic`, and the whole
pipeline's latency accounting hangs off it -- a source that timestamps at
*delivery* instead of at capture silently hides its own buffering delay from
the controller, which then compensates for less lag than actually exists.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class FrameSource(Protocol):
    """Anything the pipeline can pull frames from."""

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yield ``(monotonic_timestamp, bgr_frame)`` until exhausted."""

    def close(self) -> None:
        """Release the underlying device or file."""


class OpenCVCamera:
    """Frame source backed by ``cv2.VideoCapture``.

    Works with USB cameras, video files and RTSP URLs.

    Args:
        source: Device index, file path or stream URL.
        width: Requested capture width; ignored by file sources.
        height: Requested capture height.
        fps: Requested capture rate.
        buffer_size: Driver-side buffer depth. One is the right answer for
            closed-loop control: a deeper buffer trades latency for smoothness,
            and latency is the expensive resource here.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int | None = 1280,
        height: int | None = 720,
        fps: float | None = 30.0,
        buffer_size: int = 1,
    ) -> None:
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open video source {source!r}")

        if width:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self._capture.set(cv2.CAP_PROP_FPS, fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

    @property
    def resolution(self) -> tuple[int, int]:
        """Actual ``(width, height)`` the device settled on."""
        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield time.monotonic(), frame

    def close(self) -> None:
        self._capture.release()


class PiCamera2Source:
    """Frame source backed by Picamera2, for the Raspberry Pi camera modules.

    Picamera2 is preferred over OpenCV on the Pi because it talks to libcamera
    directly: no V4L2 shim, control over the ISP, and lower and far more
    predictable latency.

    Args:
        width: Capture width.
        height: Capture height.
        fps: Target frame rate, applied as a frame-duration limit.

    Raises:
        ImportError: If Picamera2 is not installed.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30.0) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:  # pragma: no cover - Pi-only path
            raise ImportError(
                "picamera2 not found. On Raspberry Pi OS: sudo apt install -y python3-picamera2"
            ) from exc

        self._camera = Picamera2()
        frame_us = int(1_000_000 / fps)
        config = self._camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameDurationLimits": (frame_us, frame_us)},
        )
        self._camera.configure(config)
        self._camera.start()
        self._resolution = (width, height)

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        while True:
            # Picamera2's "RGB888" is BGR in memory, which is what OpenCV and
            # the rest of this pipeline expect. No conversion needed.
            frame = self._camera.capture_array()
            yield time.monotonic(), frame

    def close(self) -> None:
        self._camera.stop()
        self._camera.close()


class ImageSequenceSource:
    """Frame source over a directory of images, for deterministic testing.

    Args:
        directory: Folder of image files.
        pattern: Glob pattern selecting and ordering the frames.
        fps: Nominal rate used to synthesise timestamps, so downstream timing
            is reproducible rather than dependent on how fast the disk is.
    """

    def __init__(self, directory: Path, pattern: str = "*.jpg", fps: float = 30.0) -> None:
        self._paths = sorted(Path(directory).glob(pattern))
        if not self._paths:
            raise FileNotFoundError(f"no images matching {pattern!r} in {directory}")
        self._dt = 1.0 / fps

    @property
    def paths(self) -> list[Path]:
        return list(self._paths)

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        for index, path in enumerate(self._paths):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            yield index * self._dt, frame

    def close(self) -> None:
        return None


def build_source(spec: str, width: int = 1280, height: int = 720, fps: float = 30.0) -> FrameSource:
    """Create a frame source from a command-line style specification.

    Args:
        spec: ``"picamera"``, a camera index like ``"0"``, a directory of
            images, or a video file / stream URL.
        width: Requested width.
        height: Requested height.
        fps: Requested frame rate.

    Returns:
        A ready :class:`FrameSource`.
    """
    if spec == "picamera":
        return PiCamera2Source(width, height, fps)
    if spec.isdigit():
        return OpenCVCamera(int(spec), width, height, fps)
    path = Path(spec)
    if path.is_dir():
        return ImageSequenceSource(path, fps=fps)
    return OpenCVCamera(spec, width, height, fps)
