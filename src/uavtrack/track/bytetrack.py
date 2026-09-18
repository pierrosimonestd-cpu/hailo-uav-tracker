"""Multi-object tracking by two-stage IoU association (ByteTrack-style).

The idea worth keeping from ByteTrack: do not throw low-confidence detections
away. Associate the confident ones first, then give the surviving tracks a
second chance to claim a weak detection. On this dataset that is not a detail --
a UAV at 400 m is a handful of low-contrast pixels whose score collapses the
moment it crosses a cloud edge, and a tracker that only ever sees
high-confidence boxes drops the lock exactly when it matters.

    Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every
    Detection Box", ECCV 2022. https://arxiv.org/abs/2110.06864

This is an independent implementation of that association strategy, not a port
of the reference code.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field

import numpy as np

from uavtrack.detect.base import Detection
from uavtrack.track.kalman import KalmanBoxTracker

try:
    # Resolved at import, never inside the loop. Importing SciPy lazily on the
    # first association costs over a second, and it lands on whichever frame
    # first has both a track and a detection -- measured as a 1.8 s p99 spike in
    # benchmarks/bench_latency.py before this was moved out here.
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except ImportError:  # pragma: no cover - exercised by the SciPy-free install
    _linear_sum_assignment = None


class TrackState(enum.Enum):
    """Lifecycle of a track."""

    TENTATIVE = "tentative"  # seen, but not yet often enough to be trusted
    CONFIRMED = "confirmed"  # promoted, safe to act on
    LOST = "lost"  # coasting on prediction alone
    REMOVED = "removed"  # scheduled for deletion


@dataclass
class Track:
    """One tracked object and its filter.

    Attributes:
        track_id: Stable identifier for the lifetime of the track.
        filter: The Kalman filter holding the motion estimate.
        state: Current lifecycle state.
        score: Score of the most recent associated detection.
        class_id: Class of the most recent associated detection.
        hits: Number of successful associations.
        age: Number of update cycles since creation.
        time_since_update: Cycles since the last successful association.
    """

    track_id: int
    filter: KalmanBoxTracker
    state: TrackState = TrackState.TENTATIVE
    score: float = 0.0
    class_id: int = 0
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def box_xyxy(self) -> np.ndarray:
        return self.filter.box_xyxy

    @property
    def centre(self) -> tuple[float, float]:
        cx, cy = self.filter.mean[:2]
        return float(cx), float(cy)

    @property
    def is_active(self) -> bool:
        """True when the track is confirmed and was updated this cycle."""
        return self.state is TrackState.CONFIRMED and self.time_since_update == 0


def iou_matrix(tracks_xyxy: np.ndarray, detections_xyxy: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes.

    Returns:
        ``(len(tracks), len(detections))`` matrix; empty inputs give an empty
        matrix rather than raising.
    """
    if tracks_xyxy.size == 0 or detections_xyxy.size == 0:
        return np.zeros((len(tracks_xyxy), len(detections_xyxy)), dtype=np.float32)

    t = tracks_xyxy[:, None, :]
    d = detections_xyxy[None, :, :]

    inter_w = np.maximum(0.0, np.minimum(t[..., 2], d[..., 2]) - np.maximum(t[..., 0], d[..., 0]))
    inter_h = np.maximum(0.0, np.minimum(t[..., 3], d[..., 3]) - np.maximum(t[..., 1], d[..., 1]))
    inter = inter_w * inter_h

    area_t = np.maximum(0.0, t[..., 2] - t[..., 0]) * np.maximum(0.0, t[..., 3] - t[..., 1])
    area_d = np.maximum(0.0, d[..., 2] - d[..., 0]) * np.maximum(0.0, d[..., 3] - d[..., 1])
    union = area_t + area_d - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def match(cost: np.ndarray, threshold: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve the assignment between rows and columns of an IoU matrix.

    Uses the Hungarian algorithm when SciPy is installed and falls back to
    greedy descending-IoU matching otherwise, so a Raspberry Pi install need not
    carry SciPy. Which one is in use is decided at import time, so neither path
    can surprise the control loop with a first-call cost.

    The fallback is optimal *for this shape of problem*, not in general. Greedy
    matching is only equivalent to the Hungarian solution when each track has an
    unambiguous best detection, which is the normal case here: airborne targets
    are well separated, so a track overlaps one detection strongly and the rest
    at essentially zero. Measured over 500 such matrices the two agree every
    time; over 500 matrices of uniform random costs they disagree on about one
    in nine, by up to a third of the total. If you ever put this in front of
    genuinely overlapping targets, install SciPy.
    See ``tests/test_tracking.py`` for both measurements.

    Args:
        cost: ``(n_tracks, n_detections)`` IoU matrix, higher is better.
        threshold: Minimum IoU for a pair to be considered a match.

    Returns:
        ``(matches, unmatched_rows, unmatched_cols)``.
    """
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    if _linear_sum_assignment is not None:
        row_idx, col_idx = _linear_sum_assignment(-cost)
        pairs = list(zip(row_idx.tolist(), col_idx.tolist(), strict=True))
    else:
        pairs = _greedy_pairs(cost)

    matches = [(r, c) for r, c in pairs if cost[r, c] >= threshold]
    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}
    return (
        matches,
        [r for r in range(n_rows) if r not in matched_rows],
        [c for c in range(n_cols) if c not in matched_cols],
    )


def _greedy_pairs(cost: np.ndarray) -> list[tuple[int, int]]:
    """Greedy one-to-one pairing, best IoU first."""
    order = np.argsort(cost, axis=None)[::-1]
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for flat in order:
        r, c = divmod(int(flat), cost.shape[1])
        if r in used_rows or c in used_cols:
            continue
        used_rows.add(r)
        used_cols.add(c)
        pairs.append((r, c))
        if len(pairs) == min(cost.shape):
            break
    return pairs


class ByteTracker:
    """Two-stage IoU tracker over :class:`~uavtrack.detect.base.Detection`.

    Args:
        high_threshold: Detections at or above this score take part in the first
            association pass and may start new tracks.
        low_threshold: Detections between ``low_threshold`` and
            ``high_threshold`` are used only to rescue existing tracks.
        match_threshold: Minimum IoU for the first association pass.
        second_match_threshold: Minimum IoU for the low-score rescue pass. It is
            looser because a weak detection is usually also a sloppier box.
        max_age: Cycles a track may coast without an association before removal.
        min_hits: Associations required before a track is confirmed.
    """

    def __init__(
        self,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_threshold: float = 0.3,
        second_match_threshold: float = 0.2,
        max_age: int = 30,
        min_hits: int = 3,
    ) -> None:
        if not 0.0 <= low_threshold <= high_threshold <= 1.0:
            raise ValueError("expected 0 <= low_threshold <= high_threshold <= 1")

        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_threshold = match_threshold
        self.second_match_threshold = second_match_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self.tracks: list[Track] = []
        self._ids = itertools.count(1)

    def reset(self) -> None:
        """Drop every track and restart id numbering."""
        self.tracks.clear()
        self._ids = itertools.count(1)

    def update(self, detections: list[Detection], dt: float) -> list[Track]:
        """Advance all tracks by ``dt`` seconds and fold in ``detections``.

        Args:
            detections: Detections for the current frame, any order.
            dt: Seconds since the previous call.

        Returns:
            The confirmed tracks after this update.
        """
        for track in self.tracks:
            track.filter.predict(dt)
            track.age += 1
            track.time_since_update += 1

        high = [d for d in detections if d.score >= self.high_threshold]
        low = [d for d in detections if self.low_threshold <= d.score < self.high_threshold]

        # Pass 1: confident detections against every live track.
        live = [t for t in self.tracks if t.state is not TrackState.REMOVED]
        matches, unmatched_tracks, unmatched_high = self._associate(
            live, high, self.match_threshold
        )
        for track_idx, det_idx in matches:
            self._apply(live[track_idx], high[det_idx])

        # Pass 2: weak detections against whatever is still unmatched.
        remaining = [live[i] for i in unmatched_tracks]
        matches2, still_unmatched, _ = self._associate(remaining, low, self.second_match_threshold)
        for track_idx, det_idx in matches2:
            self._apply(remaining[track_idx], low[det_idx])

        for track_idx in still_unmatched:
            track = remaining[track_idx]
            if track.state is TrackState.CONFIRMED:
                track.state = TrackState.LOST
            elif track.state is TrackState.TENTATIVE:
                # A tentative track that misses immediately was probably noise.
                track.state = TrackState.REMOVED

        for det_idx in unmatched_high:
            self._spawn(high[det_idx])

        self.tracks = [
            t
            for t in self.tracks
            if t.state is not TrackState.REMOVED and t.time_since_update <= self.max_age
        ]
        return [t for t in self.tracks if t.state is TrackState.CONFIRMED]

    def _associate(
        self, tracks: list[Track], detections: list[Detection], threshold: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        track_boxes = np.stack([t.box_xyxy for t in tracks])
        det_boxes = np.stack([d.as_xyxy() for d in detections])
        return match(iou_matrix(track_boxes, det_boxes), threshold)

    def _apply(self, track: Track, detection: Detection) -> None:
        track.filter.update(detection.as_xywh())
        track.score = detection.score
        track.class_id = detection.class_id
        track.hits += 1
        track.time_since_update = 0
        track.history.append(track.centre)
        if (
            track.state is not TrackState.CONFIRMED
            and track.hits >= self.min_hits
            or track.state is TrackState.LOST
        ):
            track.state = TrackState.CONFIRMED

    def _spawn(self, detection: Detection) -> None:
        track = Track(
            track_id=next(self._ids),
            filter=KalmanBoxTracker(detection.as_xywh()),
            score=detection.score,
            class_id=detection.class_id,
        )
        track.history.append(track.centre)
        # min_hits == 1 means "trust the first detection"; honour it immediately
        # rather than making the caller wait a cycle.
        if self.min_hits <= 1:
            track.state = TrackState.CONFIRMED
        self.tracks.append(track)
