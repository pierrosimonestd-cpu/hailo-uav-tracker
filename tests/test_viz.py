"""Overlay drawing.

Drawing code is not worth testing pixel by pixel, but two properties are worth
pinning: that annotating never mutates the frame the pipeline is still holding,
and that every branch survives the states it will actually meet -- no target, a
coasting track, a track with no history yet.
"""

from __future__ import annotations

import numpy as np

from uavtrack.control.gimbal import GimbalCommand
from uavtrack.detect.base import Detection
from uavtrack.pipeline import FrameResult
from uavtrack.track.bytetrack import ByteTracker, TrackState
from uavtrack.viz import annotate, draw_hud, draw_reticle, draw_tracks


def make_tracks(frames: int = 5):
    tracker = ByteTracker(min_hits=2)
    tracks = []
    for i in range(frames):
        x = 300.0 + 8 * i
        tracks = tracker.update([Detection(x, 300, x + 40, 340, 0.9)], 0.033)
    return tracker, tracks


def make_result(tracks, target, command=None, detections=None) -> FrameResult:
    return FrameResult(
        index=3,
        timestamp=0.1,
        frame=np.zeros((720, 1280, 3), dtype=np.uint8),
        detections=detections if detections is not None else [Detection(300, 300, 340, 340, 0.9)],
        tracks=tracks,
        target=target,
        command=command,
        timings_ms={"detect": 12.3, "track": 0.4, "control": 0.1},
    )


def a_command() -> GimbalCommand:
    return GimbalCommand(
        pan_deg=95.0,
        tilt_deg=88.0,
        pan_error_deg=1.5,
        tilt_error_deg=-0.7,
        pan_target_bearing_deg=96.5,
        tilt_target_bearing_deg=87.3,
        pan_rate_deg_s=6.0,
        tilt_rate_deg_s=-2.0,
        in_deadband=False,
    )


def test_annotate_does_not_mutate_the_source_frame():
    """The pipeline still holds this frame; drawing on it would be a surprise."""
    _, tracks = make_tracks()
    result = make_result(tracks, tracks[0], a_command())
    before = result.frame.copy()

    annotated = annotate(result)
    assert np.array_equal(result.frame, before)
    assert annotated is not result.frame
    assert annotated.shape == result.frame.shape


def test_annotate_draws_something():
    _, tracks = make_tracks()
    annotated = annotate(make_result(tracks, tracks[0], a_command()))
    assert annotated.any(), "an all-black overlay means nothing was drawn"


def test_annotate_without_a_target():
    annotated = annotate(make_result([], None, None, detections=[]))
    assert annotated.any(), "the reticle and HUD are drawn even with no target"


def test_annotate_handles_a_coasting_track():
    tracker, _ = make_tracks()
    for _ in range(3):
        tracker.update([], 0.033)
    lost = [t for t in tracker.tracks if t.state is TrackState.LOST]
    assert lost, "the setup must actually produce a lost track"
    assert annotate(make_result(lost, None, None)).any()


def test_draw_tracks_survives_a_track_with_no_history():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, tracks = make_tracks()
    tracks[0].history.clear()
    draw_tracks(frame, tracks, target_id=tracks[0].track_id)


def test_draw_hud_with_no_lines_is_a_no_op():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert not draw_hud(frame, []).any()


def test_reticle_is_centred():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_reticle(frame, None)
    # The cross arms sit either side of the centre, leaving the exact centre clear.
    assert frame[240, 640 // 2 - 10].any()
    assert not frame[240, 320].any()
