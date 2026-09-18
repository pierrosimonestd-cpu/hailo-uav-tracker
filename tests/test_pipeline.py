"""End-to-end wiring of the pipeline, with a stub detector.

These tests exist because the interesting failures in a pipeline are not in any
one component. They are in the seams: a `dt` that is zero on the first frame, a
controller that keeps integrating after the target is gone, a link that is
written to but never read. None of those show up in a unit test of the part that
causes them.

A stub detector stands in for the model so the whole loop runs in milliseconds
on any machine, with no weights, no NPU and no camera.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavtrack.config import PipelineConfig
from uavtrack.detect.base import Detection, DetectorBackend
from uavtrack.pipeline import TrackingPipeline


class ScriptedDetector(DetectorBackend):
    """Returns a pre-programmed detection per frame.

    Args:
        script: One list of detections per frame; exhausting it yields nothing.
    """

    def __init__(self, script: list[list[Detection]]) -> None:
        self.script = script
        self.calls = 0

    @property
    def input_size(self) -> tuple[int, int]:
        return (640, 640)

    @property
    def name(self) -> str:
        return "scripted"

    def infer(self, frame: np.ndarray) -> list[Detection]:
        index = self.calls
        self.calls += 1
        return self.script[index] if index < len(self.script) else []


class ScriptedSource:
    """Yields blank frames at a fixed nominal rate."""

    def __init__(self, count: int, fps: float = 30.0, size: tuple[int, int] = (1280, 720)) -> None:
        self.count = count
        self.dt = 1.0 / fps
        self.size = size
        self.closed = False

    def frames(self):
        frame = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        for index in range(self.count):
            yield index * self.dt, frame

    def close(self) -> None:
        self.closed = True


class RecordingLink:
    """Stands in for the serial link and records what it was asked to do."""

    def __init__(self) -> None:
        self.commands: list[tuple[float, float]] = []
        self.heartbeats = 0
        self.polls = 0
        self.closed = False

    def send_angles(self, pan_deg, tilt_deg, effector=None):  # noqa: ARG002
        self.commands.append((pan_deg, tilt_deg))
        return True

    def maintain(self) -> None:
        self.heartbeats += 1

    def poll(self, max_bytes: int = 256):  # noqa: ARG002
        self.polls += 1
        return []

    def close(self) -> None:
        self.closed = True


def moving_target(frames: int, x0: float = 700.0, step: float = 8.0) -> list[list[Detection]]:
    """A single target drifting right across the frame."""
    return [
        [Detection(x0 + step * i, 340.0, x0 + step * i + 40.0, 380.0, 0.9)] for i in range(frames)
    ]


def config(**overrides) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.camera.width, cfg.camera.height, cfg.camera.fps = 1280, 720, 30.0
    cfg.tracker.min_hits = 2
    cfg.link.enabled = False
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        setattr(getattr(cfg, section), field, value)
    return cfg


def build(script, frames, link=None, cfg=None):
    return TrackingPipeline(
        cfg or config(),
        detector=ScriptedDetector(script),
        source=ScriptedSource(frames),
        link=link,
    )


# ------------------------------------------------------------------- plumbing


def test_pipeline_produces_one_result_per_frame():
    with build(moving_target(10), 10) as pipeline:
        results = list(pipeline.run())
    assert len(results) == 10
    assert [r.index for r in results] == list(range(10))


def test_every_stage_is_timed():
    with build(moving_target(5), 5) as pipeline:
        results = list(pipeline.run())
    for result in results:
        assert set(result.timings_ms) == {"detect", "track", "control"}
        assert result.total_ms >= 0.0


def test_max_frames_stops_early():
    with build(moving_target(20), 20) as pipeline:
        assert len(list(pipeline.run(max_frames=4))) == 4


def test_close_releases_the_source_and_the_link():
    link = RecordingLink()
    pipeline = build(moving_target(3), 3, link=link)
    source = pipeline.source
    list(pipeline.run())
    pipeline.close()
    assert source.closed and link.closed


# -------------------------------------------------------------------- tracking


def test_a_target_is_acquired_and_held():
    with build(moving_target(12), 12) as pipeline:
        results = list(pipeline.run())

    assert results[0].target is None, "min_hits means no lock on the first frame"
    locked = [r for r in results if r.target is not None]
    assert len(locked) >= 8
    assert len({r.target.track_id for r in locked}) == 1, "identity must not change"


def test_the_turret_turns_towards_a_target_on_the_right():
    with build(moving_target(12), 12) as pipeline:
        results = list(pipeline.run())

    commands = [r.command for r in results if r.command is not None]
    assert commands, "a locked target must produce commands"
    assert commands[-1].pan_deg > 90.0


def test_loop_opens_when_the_target_disappears():
    script = moving_target(6) + [[] for _ in range(20)]
    with build(script, len(script)) as pipeline:
        results = list(pipeline.run())

    assert results[-1].target is None
    assert results[-1].command is None, "no target means no command"


def test_integrators_are_cleared_on_target_loss():
    """A stale integral must not be dumped into the first post-reacquisition command."""
    script = moving_target(8) + [[] for _ in range(40)]
    with build(script, len(script)) as pipeline:
        list(pipeline.run())
        assert pipeline.controller.pan_pid.state.integral == pytest.approx(0.0)
        assert pipeline.controller.pan_pid.state.last_measurement is None


# ------------------------------------------------------------------ the link


def test_commands_reach_the_link_only_while_locked():
    link = RecordingLink()
    script = moving_target(10) + [[] for _ in range(20)]
    with build(script, len(script), link=link) as pipeline:
        results = list(pipeline.run())

    locked = sum(1 for r in results if r.target is not None)
    assert len(link.commands) == locked


def test_heartbeats_are_sent_while_there_is_no_target():
    link = RecordingLink()
    with build([[] for _ in range(10)], 10, link=link) as pipeline:
        list(pipeline.run())
    assert link.heartbeats == 10, "silence must never be incidental"


def test_the_link_is_polled_every_frame():
    link = RecordingLink()
    with build(moving_target(10), 10, link=link) as pipeline:
        list(pipeline.run())
    assert link.polls == 10


# -------------------------------------------------------------------- timing


def test_first_frame_does_not_poison_the_filters_with_a_zero_interval():
    """dt is undefined on the first frame; a zero would divide through the PID
    and inflate the Kalman covariance into NaN. The pipeline substitutes the
    nominal frame interval instead, and every downstream state must stay finite."""
    with build(moving_target(6), 6) as pipeline:
        results = list(pipeline.run())

    assert len(results) == 6
    track = pipeline.tracker.tracks[0]
    assert np.all(np.isfinite(track.filter.mean))
    assert np.all(np.isfinite(track.filter.covariance))
    assert np.isfinite(pipeline.controller.pan_deg)
    assert np.isfinite(pipeline.controller.pan_pid.state.integral)


def test_commands_stay_inside_the_configured_limits():
    cfg = config()
    cfg.control.tilt.min_deg, cfg.control.tilt.max_deg = 20.0, 160.0
    # A target pinned hard into the top-right corner drives both axes to a rail.
    script = [[Detection(1200.0, 10.0, 1270.0, 80.0, 0.95)] for _ in range(60)]
    with build(script, 60, cfg=cfg) as pipeline:
        results = list(pipeline.run())

    for result in results:
        if result.command is None:
            continue
        assert 0.0 <= result.command.pan_deg <= 180.0
        assert 20.0 <= result.command.tilt_deg <= 160.0
