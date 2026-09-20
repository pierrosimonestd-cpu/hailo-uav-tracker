"""The end-to-end pipeline: capture, detect, track, select, point.

Everything in the loop is driven by the *capture* timestamp rather than by a
nominal frame rate. Detection time on a shared NPU varies with how many other
things want it, so a loop that assumes a fixed ``dt`` accumulates a timing error
that the Kalman filter has no way to see. Passing real elapsed times through
makes every downstream estimate correct by construction.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from uavtrack.config import PipelineConfig
from uavtrack.control.gimbal import AxisLimits, CameraGeometry, GimbalCommand, PanTiltController
from uavtrack.control.pid import PIDGains
from uavtrack.detect.base import Detection, DetectorBackend
from uavtrack.detect.factory import build_detector
from uavtrack.io.camera import FrameSource, build_source
from uavtrack.track.bytetrack import ByteTracker, Track
from uavtrack.track.selector import TargetSelector

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Everything one iteration of the pipeline produced."""

    index: int
    timestamp: float
    frame: np.ndarray
    detections: list[Detection]
    tracks: list[Track]
    target: Track | None
    command: GimbalCommand | None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        """Wall-clock time spent on this frame, excluding capture."""
        return sum(self.timings_ms.values())


class TrackingPipeline:
    """Wires the detector, tracker, selector and controller together.

    Args:
        config: Full pipeline configuration.
        detector: Pre-built detector. Mainly a testing seam; by default one is
            constructed from ``config.detector``.
        source: Pre-built frame source. Same rationale.
        link: Serial link to the firmware, or ``None`` to run without hardware.
    """

    def __init__(
        self,
        config: PipelineConfig,
        detector: DetectorBackend | None = None,
        source: FrameSource | None = None,
        link: object | None = None,
    ) -> None:
        self.config = config
        self.detector = detector or build_detector(
            config.detector.model,
            backend=config.detector.backend,
            conf_threshold=config.detector.conf_threshold,
            iou_threshold=config.detector.iou_threshold,
            labels=tuple(config.detector.labels),
        )
        self.source = source or build_source(
            config.camera.source, config.camera.width, config.camera.height, config.camera.fps
        )
        self.link = link
        self._keep_class_ids = self._resolve_classes()

        self.tracker = ByteTracker(
            high_threshold=config.tracker.high_threshold,
            low_threshold=config.tracker.low_threshold,
            match_threshold=config.tracker.match_threshold,
            second_match_threshold=config.tracker.second_match_threshold,
            max_age=config.tracker.max_age,
            min_hits=config.tracker.min_hits,
        )

        geometry = CameraGeometry(
            width=config.camera.width,
            height=config.camera.height,
            horizontal_fov_deg=config.camera.horizontal_fov_deg,
            vertical_fov_deg=config.camera.vertical_fov_deg,
        )
        self.selector = TargetSelector(frame_size=(config.camera.width, config.camera.height))
        self.controller = PanTiltController(
            geometry=geometry,
            pan_gains=PIDGains(config.control.pan.kp, config.control.pan.ki, config.control.pan.kd),
            tilt_gains=PIDGains(
                config.control.tilt.kp, config.control.tilt.ki, config.control.tilt.kd
            ),
            pan_limits=_axis_limits(config.control.pan),
            tilt_limits=_axis_limits(config.control.tilt),
            latency_s=config.control.latency_s,
            feedforward_gain=config.control.feedforward_gain,
            estimator_window=config.control.estimator_window,
            home=(config.control.home_pan_deg, config.control.home_tilt_deg),
        )

        self._last_timestamp: float | None = None
        self._had_target = False

    def _resolve_classes(self) -> set[int] | None:
        """Class indices to keep, from the names in the config.

        Resolved once against the detector's own label list rather than against
        the config's, so a mismatch between the two surfaces here as a clear
        error instead of as a turret that silently never sees anything.
        """
        wanted = self.config.detector.classes
        if not wanted:
            return None

        labels = list(getattr(self.detector, "labels", ()) or ())
        if not labels:
            raise ValueError(
                "detector.classes was set but the backend exposes no labels to match "
                "them against; remove the filter or give the backend a label list"
            )

        lookup = {name.lower(): index for index, name in enumerate(labels)}
        missing = [name for name in wanted if name.lower() not in lookup]
        if missing:
            raise ValueError(
                f"detector.classes names {missing} which this model does not have. "
                f"It knows: {', '.join(labels)}"
            )
        return {lookup[name.lower()] for name in wanted}

    def run(self, max_frames: int | None = None) -> Iterator[FrameResult]:
        """Iterate over frames, yielding one :class:`FrameResult` each.

        Args:
            max_frames: Stop after this many frames. ``None`` runs until the
                source is exhausted.

        Yields:
            The result for each processed frame.
        """
        self.detector.warmup()

        for index, (timestamp, frame) in enumerate(self.source.frames()):
            if max_frames is not None and index >= max_frames:
                break

            dt = timestamp - self._last_timestamp if self._last_timestamp is not None else 0.0
            self._last_timestamp = timestamp
            # A zero or negative dt would corrupt every filter downstream; on
            # the first frame there simply is no interval yet, so fall back to
            # the nominal one rather than propagating a zero.
            if dt <= 0.0:
                dt = 1.0 / max(self.config.camera.fps, 1.0)

            timings: dict[str, float] = {}

            start = time.perf_counter()
            detections = self.detector.infer(frame)
            if self._keep_class_ids is not None:
                detections = [d for d in detections if d.class_id in self._keep_class_ids]
            timings["detect"] = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            tracks = self.tracker.update(detections, dt)
            target = self.selector.select(tracks, dt)
            timings["track"] = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            command = self._steer(target, dt)
            timings["control"] = (time.perf_counter() - start) * 1000.0

            yield FrameResult(
                index=index,
                timestamp=timestamp,
                frame=frame,
                detections=detections,
                tracks=tracks,
                target=target,
                command=command,
                timings_ms=timings,
            )

    def _steer(self, target: Track | None, dt: float) -> GimbalCommand | None:
        """Update the controller and push a command to the firmware."""
        if target is None:
            if self._had_target:
                # Opening the loop: clear the integrators so a stale error
                # cannot be dumped into the first command after re-acquisition.
                self.controller.reset()
                self._had_target = False
            if self.link is not None:
                self.link.maintain()
                self.link.poll()
            return None

        self._had_target = True
        command = self.controller.update(target.centre, dt)

        if self.link is not None:
            self.link.send_angles(command.pan_deg, command.tilt_deg)
            self.link.poll()
        return command

    def close(self) -> None:
        """Release the detector, the frame source and the link."""
        self.detector.close()
        self.source.close()
        if self.link is not None:
            self.link.close()

    def __enter__(self) -> TrackingPipeline:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _axis_limits(axis) -> AxisLimits:
    return AxisLimits(
        min_deg=axis.min_deg,
        max_deg=axis.max_deg,
        max_rate_deg_s=axis.max_rate_deg_s,
        deadband_deg=axis.deadband_deg,
        invert=axis.invert,
    )
