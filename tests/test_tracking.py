"""Kalman filtering, association and target selection."""

from __future__ import annotations

import numpy as np
import pytest

from uavtrack.detect.base import Detection
from uavtrack.track.bytetrack import ByteTracker, TrackState, iou_matrix, match
from uavtrack.track.kalman import KalmanBoxTracker
from uavtrack.track.selector import TargetSelector

# --------------------------------------------------------------------- Kalman


def test_filter_converges_on_a_constant_velocity_target():
    """The whole point of the filter: learn the velocity, fast."""
    dt, vx = 0.05, 120.0
    kf = KalmanBoxTracker(np.array([100.0, 100.0, 40.0, 40.0]))

    for step in range(1, 12):
        kf.predict(dt)
        kf.update(np.array([100.0 + vx * step * dt, 100.0, 40.0, 40.0]))

    assert kf.velocity[0] == pytest.approx(vx, rel=0.05)
    assert kf.velocity[1] == pytest.approx(0.0, abs=5.0)


def test_lookahead_extrapolates_without_mutating_state():
    dt, vx = 0.05, 100.0
    kf = KalmanBoxTracker(np.array([100.0, 100.0, 40.0, 40.0]))
    for step in range(1, 11):
        kf.predict(dt)
        kf.update(np.array([100.0 + vx * step * dt, 100.0, 40.0, 40.0]))

    before = kf.mean.copy()
    ahead = kf.predict_ahead(0.1)
    assert np.allclose(kf.mean, before), "predict_ahead must not mutate the filter"
    assert ahead[0] == pytest.approx(kf.mean[0] + vx * 0.1, rel=0.05)


def test_non_positive_dt_is_ignored():
    kf = KalmanBoxTracker(np.array([100.0, 100.0, 40.0, 40.0]))
    before = kf.covariance.copy()
    kf.predict(0.0)
    kf.predict(-1.0)
    assert np.allclose(kf.covariance, before)


def test_covariance_stays_symmetric_and_positive_definite():
    """Joseph-form updates must not let the covariance drift."""
    kf = KalmanBoxTracker(np.array([100.0, 100.0, 40.0, 40.0]))
    rng = np.random.default_rng(0)
    for step in range(200):
        kf.predict(0.033)
        kf.update(np.array([100.0 + step, 100.0, 40.0, 40.0]) + rng.normal(0, 2, 4))

    assert np.allclose(kf.covariance, kf.covariance.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(kf.covariance) > 0)


def test_box_dimensions_stay_positive():
    kf = KalmanBoxTracker(np.array([100.0, 100.0, 4.0, 4.0]))
    for _ in range(20):
        kf.predict(0.05)
        kf.update(np.array([100.0, 100.0, 0.5, 0.5]))
    assert kf.box_xywh[2] >= 1.0
    assert kf.box_xywh[3] >= 1.0


# ---------------------------------------------------------------- association


def test_iou_matrix_values():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    iou = iou_matrix(a, b)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(50.0 / 150.0)
    assert iou[0, 2] == pytest.approx(0.0)


def test_iou_matrix_handles_empty_inputs():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((2, 4))).shape == (0, 2)
    assert iou_matrix(np.zeros((2, 4)), np.zeros((0, 4))).shape == (2, 0)


def test_match_respects_the_threshold():
    cost = np.array([[0.9, 0.1], [0.05, 0.8]])
    matches, unmatched_rows, unmatched_cols = match(cost, 0.3)
    assert sorted(matches) == [(0, 0), (1, 1)]
    assert unmatched_rows == [] and unmatched_cols == []

    matches, unmatched_rows, unmatched_cols = match(cost, 0.95)
    assert matches == []
    assert unmatched_rows == [0, 1] and unmatched_cols == [0, 1]


# -------------------------------------------------------------------- tracker


def _walk(tracker, steps, dt=0.05, vx=100.0, x0=300.0, score=0.9):
    tracks = []
    for i in range(steps):
        x = x0 + vx * i * dt
        tracks = tracker.update([Detection(x, 300, x + 40, 340, score)], dt)
    return tracks


def test_track_is_confirmed_after_min_hits():
    tracker = ByteTracker(min_hits=3)
    assert tracker.update([Detection(300, 300, 340, 340, 0.9)], 0.05) == []
    assert tracker.update([Detection(302, 300, 342, 340, 0.9)], 0.05) == []
    assert len(tracker.update([Detection(304, 300, 344, 340, 0.9)], 0.05)) == 1


def test_identity_is_stable_across_a_traverse():
    tracker = ByteTracker(min_hits=2)
    _walk(tracker, 20)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id == 1


def test_track_coasts_through_a_dropout_and_recovers():
    tracker = ByteTracker(min_hits=2)
    _walk(tracker, 10)
    track_id = tracker.tracks[0].track_id

    for _ in range(5):
        tracker.update([], 0.05)
    assert tracker.tracks[0].state is TrackState.LOST
    coasted_x = tracker.tracks[0].centre[0]
    assert coasted_x > 320, "a lost track must keep moving on its velocity estimate"

    tracks = tracker.update([Detection(coasted_x - 20, 300, coasted_x + 20, 340, 0.9)], 0.05)
    assert len(tracks) == 1
    assert tracks[0].track_id == track_id, "re-acquisition must not mint a new id"


def test_track_is_removed_after_max_age():
    tracker = ByteTracker(min_hits=2, max_age=5)
    _walk(tracker, 6)
    for _ in range(10):
        tracker.update([], 0.05)
    assert tracker.tracks == []


def test_low_score_detection_rescues_an_existing_track():
    """The reason for ByteTrack's second pass."""
    tracker = ByteTracker(min_hits=2, high_threshold=0.5, low_threshold=0.1)
    _walk(tracker, 6, score=0.9)
    before = tracker.tracks[0].track_id

    # A weak detection at the predicted position: too weak to start a track,
    # strong enough to keep this one alive.
    predicted_x = tracker.tracks[0].centre[0] + 5
    tracks = tracker.update([Detection(predicted_x - 20, 300, predicted_x + 20, 340, 0.25)], 0.05)
    assert len(tracks) == 1
    assert tracks[0].track_id == before
    assert tracks[0].time_since_update == 0


def test_low_score_detection_alone_does_not_start_a_track():
    tracker = ByteTracker(min_hits=1, high_threshold=0.5, low_threshold=0.1)
    for _ in range(5):
        assert tracker.update([Detection(300, 300, 340, 340, 0.2)], 0.05) == []


def test_two_targets_keep_separate_identities():
    tracker = ByteTracker(min_hits=2)
    for i in range(12):
        left = 200 + 5 * i
        right = 800 - 5 * i
        tracks = tracker.update(
            [
                Detection(left, 300, left + 40, 340, 0.9),
                Detection(right, 300, right + 40, 340, 0.9),
            ],
            0.05,
        )
    assert len({t.track_id for t in tracks}) == 2


def test_reset_clears_tracks_and_ids():
    tracker = ByteTracker(min_hits=1)
    _walk(tracker, 4)
    tracker.reset()
    assert tracker.tracks == []
    tracks = tracker.update([Detection(300, 300, 340, 340, 0.9)], 0.05)
    assert tracks[0].track_id == 1


def test_invalid_thresholds_are_rejected():
    with pytest.raises(ValueError):
        ByteTracker(high_threshold=0.2, low_threshold=0.8)


# ------------------------------------------------------------------- selector


def _confirmed(tracker, detections, steps=4, dt=0.05):
    tracks = []
    for _ in range(steps):
        tracks = tracker.update(detections, dt)
    return tracks


def test_selector_prefers_the_larger_closer_target():
    tracker = ByteTracker(min_hits=2)
    tracks = _confirmed(
        tracker,
        [Detection(100, 100, 120, 120, 0.9), Detection(600, 300, 760, 460, 0.9)],
    )
    selector = TargetSelector(frame_size=(1280, 720), min_dwell_s=0.0)
    chosen = selector.select(tracks, 0.05)
    assert chosen.box_xyxy[2] - chosen.box_xyxy[0] > 100


def test_selector_holds_the_incumbent_during_the_dwell_time():
    tracker = ByteTracker(min_hits=2)
    tracks = _confirmed(
        tracker,
        [Detection(100, 100, 140, 140, 0.9), Detection(600, 300, 640, 340, 0.9)],
    )
    selector = TargetSelector(frame_size=(1280, 720), min_dwell_s=1.0)

    first = selector.select(tracks, 0.05)
    for _ in range(10):
        assert selector.select(tracks, 0.05).track_id == first.track_id


def test_selector_releases_when_there_is_nothing_to_follow():
    tracker = ByteTracker(min_hits=1)
    tracks = _confirmed(tracker, [Detection(100, 100, 140, 140, 0.9)], steps=2)
    selector = TargetSelector(frame_size=(1280, 720))
    assert selector.select(tracks, 0.05) is not None
    assert selector.select([], 0.05) is None
    assert selector.current_id is None


def test_hysteresis_makes_a_marginally_better_challenger_lose():
    tracker = ByteTracker(min_hits=2)
    tracks = _confirmed(
        tracker,
        [Detection(600, 340, 640, 380, 0.80), Detection(100, 100, 140, 140, 0.85)],
    )
    selector = TargetSelector(frame_size=(1280, 720), min_dwell_s=0.0)

    incumbent = selector.select(tracks, 0.05)
    for _ in range(20):
        assert selector.select(tracks, 0.05).track_id == incumbent.track_id


# ------------------------------------------------- association fallback


def test_greedy_fallback_matches_hungarian_on_iou_like_costs():
    """The fallback is optimal for the shape of problem this tracker sees.

    IoU association between well-separated airborne targets produces a matrix
    where each track overlaps one detection strongly and the rest at nearly
    zero. Greedy is provably optimal there. Pinned because the fallback is what
    runs on a SciPy-free Raspberry Pi install, and nothing else exercises it
    when SciPy is present.
    """
    scipy_solver = pytest.importorskip("scipy.optimize").linear_sum_assignment
    from uavtrack.track.bytetrack import _greedy_pairs

    rng = np.random.default_rng(0)
    disagreements = 0
    trials = 500

    for _ in range(trials):
        n_tracks = int(rng.integers(1, 5))
        n_dets = int(rng.integers(1, 5))
        # Background overlap under 0.05, one strong match per track.
        cost = (rng.random((n_tracks, n_dets)) * 0.05).astype(np.float32)
        for index in range(min(n_tracks, n_dets)):
            cost[index, index] = 0.6 + 0.4 * rng.random()

        rows, cols = scipy_solver(-cost)
        optimal = sum(cost[r, c] for r, c in zip(rows, cols, strict=True))
        greedy = sum(cost[r, c] for r, c in _greedy_pairs(cost))
        if abs(optimal - greedy) > 1e-6:
            disagreements += 1

    assert disagreements == 0, (
        f"greedy was suboptimal on {disagreements}/{trials} IoU-like matrices"
    )


def test_greedy_fallback_is_not_optimal_in_general():
    """The other half of the claim, so the docstring cannot quietly overstate it.

    On arbitrary cost matrices greedy is a genuine approximation. This test
    exists to keep that honest: if someone widens the tracker to a problem where
    costs are not IoU-like, the fallback is no longer free.
    """
    scipy_solver = pytest.importorskip("scipy.optimize").linear_sum_assignment
    from uavtrack.track.bytetrack import _greedy_pairs

    rng = np.random.default_rng(0)
    disagreements = 0
    trials = 500

    for _ in range(trials):
        cost = rng.random((int(rng.integers(1, 5)), int(rng.integers(1, 5)))).astype(np.float32)
        rows, cols = scipy_solver(-cost)
        optimal = sum(cost[r, c] for r, c in zip(rows, cols, strict=True))
        greedy = sum(cost[r, c] for r, c in _greedy_pairs(cost))
        if optimal - greedy > 1e-6:
            disagreements += 1

    assert disagreements > 0, "uniform random costs should expose greedy as an approximation"
    # Greedy is never *better* than optimal; that would mean the solver is wrong.


def test_greedy_fallback_never_double_assigns():
    from uavtrack.track.bytetrack import _greedy_pairs

    rng = np.random.default_rng(1)
    for _ in range(200):
        cost = rng.random((int(rng.integers(1, 6)), int(rng.integers(1, 6)))).astype(np.float32)
        pairs = _greedy_pairs(cost)
        assert len({r for r, _ in pairs}) == len(pairs)
        assert len({c for _, c in pairs}) == len(pairs)
        assert len(pairs) == min(cost.shape)
