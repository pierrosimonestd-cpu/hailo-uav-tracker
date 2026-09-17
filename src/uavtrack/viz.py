"""Overlay drawing for the live view and for recorded runs."""

from __future__ import annotations

import cv2
import numpy as np

from uavtrack.detect.base import Detection
from uavtrack.track.bytetrack import Track, TrackState

_COLOUR_DETECTION = (120, 120, 120)
_COLOUR_TRACK = (60, 200, 90)
_COLOUR_TARGET = (40, 90, 240)
_COLOUR_LOST = (60, 160, 220)
_COLOUR_HUD = (240, 240, 240)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw raw detections as thin grey boxes, in place."""
    for det in detections:
        cv2.rectangle(
            frame,
            (int(det.x1), int(det.y1)),
            (int(det.x2), int(det.y2)),
            _COLOUR_DETECTION,
            1,
        )
    return frame


def draw_tracks(frame: np.ndarray, tracks: list[Track], target_id: int | None = None) -> np.ndarray:
    """Draw tracks, highlighting the selected target, in place."""
    for track in tracks:
        is_target = track.track_id == target_id
        if is_target:
            colour, thickness = _COLOUR_TARGET, 2
        elif track.state is TrackState.LOST:
            colour, thickness = _COLOUR_LOST, 1
        else:
            colour, thickness = _COLOUR_TRACK, 2

        x1, y1, x2, y2 = (int(v) for v in track.box_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

        label = f"#{track.track_id} {track.score:.2f}"
        if track.state is TrackState.LOST:
            label += " (coasting)"
        _draw_label(frame, label, (x1, y1 - 6), colour)

        # A short motion trail makes it obvious at a glance whether the tracker
        # is holding one identity or swapping between two.
        if len(track.history) > 1:
            points = np.array(track.history[-25:], dtype=np.int32)
            cv2.polylines(frame, [points], isClosed=False, color=colour, thickness=1)
    return frame


def draw_reticle(frame: np.ndarray, target: Track | None) -> np.ndarray:
    """Draw the boresight cross and, when locked, the line to the target."""
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2

    cv2.line(frame, (cx - 14, cy), (cx - 4, cy), _COLOUR_HUD, 1)
    cv2.line(frame, (cx + 4, cy), (cx + 14, cy), _COLOUR_HUD, 1)
    cv2.line(frame, (cx, cy - 14), (cx, cy - 4), _COLOUR_HUD, 1)
    cv2.line(frame, (cx, cy + 4), (cx, cy + 14), _COLOUR_HUD, 1)

    if target is not None:
        tx, ty = (int(v) for v in target.centre)
        cv2.line(frame, (cx, cy), (tx, ty), _COLOUR_TARGET, 1, cv2.LINE_AA)
        cv2.circle(frame, (tx, ty), 4, _COLOUR_TARGET, -1, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a left-aligned status block over a translucent panel, in place."""
    if not lines:
        return frame

    pad = 8
    line_height = 18
    width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] for line in lines)
    box = (width + 2 * pad, len(lines) * line_height + 2 * pad)

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box[0], 8 + box[1]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for i, line in enumerate(lines):
        origin = (8 + pad, 8 + pad + (i + 1) * line_height - 5)
        cv2.putText(
            frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOUR_HUD, 1, cv2.LINE_AA
        )
    return frame


def _draw_label(frame: np.ndarray, text: str, origin: tuple[int, int], colour) -> None:
    """Draw text with a dark outline so it stays readable over any background."""
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def annotate(result, show_detections: bool = True) -> np.ndarray:
    """Draw a full overlay for one :class:`~uavtrack.pipeline.FrameResult`.

    Returns:
        A copy of the frame with detections, tracks, reticle and HUD drawn.
    """
    frame = result.frame.copy()
    if show_detections:
        draw_detections(frame, result.detections)
    draw_tracks(frame, result.tracks, result.target.track_id if result.target else None)
    draw_reticle(frame, result.target)

    lines = [
        f"frame {result.index}  {result.total_ms:5.1f} ms",
        "  ".join(f"{k} {v:.1f}" for k, v in result.timings_ms.items()),
        f"tracks {len(result.tracks)}  detections {len(result.detections)}",
    ]
    if result.command is not None:
        lines.append(f"pan {result.command.pan_deg:6.2f}  tilt {result.command.tilt_deg:6.2f}")
        lines.append(
            f"err  {result.command.pan_error_deg:+6.2f}, {result.command.tilt_error_deg:+6.2f} deg"
        )
    else:
        lines.append("no target - loop open")
    return draw_hud(frame, lines)
