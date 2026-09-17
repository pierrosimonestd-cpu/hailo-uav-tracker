"""Choosing which track the turret should point at.

A turret has one barrel and multiple targets are common, so something has to
arbitrate. The hard requirement is *stability*: a selector that re-decides every
frame makes the servos oscillate between two targets and the system looks
broken even when detection and tracking are both working perfectly.

Stability comes from two mechanisms:

* **Hysteresis.** The incumbent target keeps a bonus, so a challenger must be
  clearly better before a switch happens, not marginally better.
* **A dwell time.** After a switch, the new target is held for a minimum period
  regardless of scoring, which bounds how often the turret can change its mind.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uavtrack.track.bytetrack import Track


@dataclass(frozen=True)
class SelectionWeights:
    """Relative importance of the target-scoring terms.

    Attributes:
        confidence: Weight on detector confidence.
        size: Weight on apparent size, a monotonic proxy for proximity.
        centrality: Weight on closeness to the image centre; a target already
            near the optical axis is cheaper to keep than one at the edge.
        incumbent_bonus: Added to the current target's score. This is the
            hysteresis band -- a challenger must beat the incumbent by more than
            this to take over.
    """

    confidence: float = 1.0
    size: float = 0.6
    centrality: float = 0.3
    incumbent_bonus: float = 0.25


class TargetSelector:
    """Pick and hold a primary target across frames.

    Args:
        weights: Scoring weights.
        min_dwell_s: Minimum time a newly selected target is held before any
            switch is considered.
        frame_size: ``(width, height)`` used to normalise size and centrality.
    """

    def __init__(
        self,
        frame_size: tuple[int, int],
        weights: SelectionWeights | None = None,
        min_dwell_s: float = 0.5,
    ) -> None:
        self.weights = weights or SelectionWeights()
        self.min_dwell_s = min_dwell_s
        self.frame_width, self.frame_height = frame_size

        self._current_id: int | None = None
        self._time_on_target = 0.0

    @property
    def current_id(self) -> int | None:
        """Id of the currently selected track, if any."""
        return self._current_id

    def score(self, track: Track) -> float:
        """Score one track as a candidate primary target, higher is better."""
        box = track.box_xyxy
        width = max(box[2] - box[0], 0.0)
        height = max(box[3] - box[1], 0.0)

        # sqrt of the area fraction: linear in apparent *length*, so a target at
        # half the distance scores twice as high rather than four times.
        area_fraction = (width * height) / float(self.frame_width * self.frame_height)
        size_term = float(np.sqrt(max(area_fraction, 0.0)))

        cx, cy = track.centre
        dx = (cx - self.frame_width / 2.0) / (self.frame_width / 2.0)
        dy = (cy - self.frame_height / 2.0) / (self.frame_height / 2.0)
        centrality = 1.0 - min(1.0, float(np.hypot(dx, dy)) / np.sqrt(2.0))

        total = (
            self.weights.confidence * track.score
            + self.weights.size * size_term
            + self.weights.centrality * centrality
        )
        if track.track_id == self._current_id:
            total += self.weights.incumbent_bonus
        return total

    def select(self, tracks: list[Track], dt: float) -> Track | None:
        """Return the track to point at, or ``None`` if there is nothing to follow.

        Args:
            tracks: Confirmed tracks from this cycle.
            dt: Seconds since the previous call.

        Returns:
            The selected track, or ``None``.
        """
        self._time_on_target += dt

        if not tracks:
            self._current_id = None
            self._time_on_target = 0.0
            return None

        by_id = {t.track_id: t for t in tracks}

        # Hold the incumbent while it is alive and the dwell time is unexpired.
        if self._current_id in by_id and self._time_on_target < self.min_dwell_s:
            return by_id[self._current_id]

        best = max(tracks, key=self.score)
        if best.track_id != self._current_id:
            self._current_id = best.track_id
            self._time_on_target = 0.0
        return best
