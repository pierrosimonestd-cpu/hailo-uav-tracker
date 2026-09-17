"""Constant-velocity Kalman filter over an axis-aligned box.

State is ``[cx, cy, w, h, vx, vy, vw, vh]`` in frame pixels and pixels per
second. Velocity is per *second* rather than per frame so the filter stays
correct when the frame interval jitters -- which it does, because detection
latency on a shared NPU is not constant.

The filter exists for three reasons, in order of importance to this project:

1. **Latency compensation.** The servos are commanded from where the target
   *will* be when the command lands, not from where it was when the frame was
   captured. :meth:`KalmanBoxTracker.predict_ahead` is that lookahead.
2. **Association gating.** Motion-predicted boxes make IoU association far more
   reliable than matching against the previous frame's raw box.
3. **Coasting.** A UAV that disappears behind a chimney for a few frames keeps a
   plausible estimate instead of dropping the lock.
"""

from __future__ import annotations

import numpy as np

STATE_DIM = 8
MEASUREMENT_DIM = 4


class KalmanBoxTracker:
    """Track one box through time with a constant-velocity model.

    Args:
        box_xywh: Initial measurement ``[cx, cy, w, h]``.
        position_std: Standard deviation of the position measurement, as a
            fraction of the box size. Scaling the noise with the target keeps
            the filter equally well-tuned for a 20-pixel speck and a
            300-pixel close pass.
        accel_std: Process noise, as the standard deviation of the unmodelled
            image-plane acceleration in pixels per second squared. This is the
            one knob that decides how quickly the filter believes a change in
            velocity: too low and it lags a manoeuvring target, too high and it
            chases detector jitter. Roughly "how fast can this target's apparent
            velocity change" for your lens and frame size.
        initial_velocity_std: Standard deviation of the initial velocity
            estimate in pixels per second. A single detection says nothing about
            velocity, so this should cover the fastest target you expect.
    """

    def __init__(
        self,
        box_xywh: np.ndarray,
        position_std: float = 0.05,
        accel_std: float = 400.0,
        initial_velocity_std: float = 300.0,
    ) -> None:
        self.position_std = position_std
        self.accel_std = accel_std

        self.mean = np.zeros(STATE_DIM, dtype=np.float64)
        self.mean[:4] = np.asarray(box_xywh, dtype=np.float64)

        # Initial uncertainty: position is known about as well as the detector
        # localises, velocity is entirely unknown after a single frame.
        size = max(float(box_xywh[2]), float(box_xywh[3]), 1.0)
        std = np.array(
            [position_std * size] * 4 + [initial_velocity_std] * 4,
            dtype=np.float64,
        )
        self.covariance = np.diag(np.square(std))

        self._measurement_matrix = np.eye(MEASUREMENT_DIM, STATE_DIM)

    @staticmethod
    def _transition_matrix(dt: float) -> np.ndarray:
        """Constant-velocity state transition for a step of ``dt`` seconds."""
        matrix = np.eye(STATE_DIM)
        matrix[:4, 4:] = np.eye(4) * dt
        return matrix

    def _noise_scale(self) -> float:
        """Size used to scale noise: the larger box dimension, floored at 1 px."""
        return max(float(self.mean[2]), float(self.mean[3]), 1.0)

    def _process_noise(self, dt: float) -> np.ndarray:
        """Discrete white-noise acceleration covariance for a step of ``dt``.

        Treating unmodelled acceleration as white noise of intensity
        ``accel_std`` gives the standard block

        .. math::
            Q = \\sigma_a^2 \\begin{bmatrix} dt^4/4 & dt^3/2 \\\\
                                             dt^3/2 & dt^2 \\end{bmatrix}

        per coordinate. The off-diagonal term is what makes the filter learn a
        velocity quickly: it encodes that a position error and a velocity error
        arriving from the same unmodelled acceleration are correlated, so a
        position residual is allowed to correct the velocity estimate. Drop it
        and the filter takes seconds to accept that a target is moving at all.
        """
        variance = self.accel_std**2
        pos_pos = variance * dt**4 / 4.0
        pos_vel = variance * dt**3 / 2.0
        vel_vel = variance * dt**2

        # Box width and height change far more slowly than the centre moves, so
        # their acceleration noise is scaled down relative to the centre's.
        shape_factor = 0.01

        noise = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
        for i in range(4):
            weight = 1.0 if i < 2 else shape_factor
            noise[i, i] = pos_pos * weight
            noise[i, i + 4] = pos_vel * weight
            noise[i + 4, i] = pos_vel * weight
            noise[i + 4, i + 4] = vel_vel * weight
        return noise

    def predict(self, dt: float) -> None:
        """Advance the state by ``dt`` seconds and inflate the covariance.

        Args:
            dt: Elapsed time in seconds. Non-positive values are ignored so a
                duplicated timestamp cannot corrupt the covariance.
        """
        if dt <= 0:
            return

        transition = self._transition_matrix(dt)
        process_noise = self._process_noise(dt)
        self.mean = transition @ self.mean
        self.covariance = transition @ self.covariance @ transition.T + process_noise

    def update(self, box_xywh: np.ndarray) -> None:
        """Fold in a new measurement ``[cx, cy, w, h]``."""
        measurement = np.asarray(box_xywh, dtype=np.float64)
        size = self._noise_scale()
        measurement_noise = np.diag(np.square(np.full(MEASUREMENT_DIM, self.position_std * size)))

        innovation = measurement - self._measurement_matrix @ self.mean
        innovation_cov = (
            self._measurement_matrix @ self.covariance @ self._measurement_matrix.T
            + measurement_noise
        )

        # solve() rather than inv(): numerically better behaved, and the 4x4
        # system is cheap enough that there is nothing to gain from caching.
        gain = np.linalg.solve(innovation_cov, self._measurement_matrix @ self.covariance).T

        self.mean = self.mean + gain @ innovation
        identity = np.eye(STATE_DIM)
        factor = identity - gain @ self._measurement_matrix
        # Joseph form: keeps the covariance symmetric positive-definite over
        # long runs, where the textbook (I-KH)P form slowly drifts.
        self.covariance = factor @ self.covariance @ factor.T + gain @ measurement_noise @ gain.T

        # Width and height are physically positive; clamp rather than let a
        # noisy update flip them.
        self.mean[2] = max(self.mean[2], 1.0)
        self.mean[3] = max(self.mean[3], 1.0)

    @property
    def box_xywh(self) -> np.ndarray:
        """Current filtered box ``[cx, cy, w, h]``."""
        return self.mean[:4].copy()

    @property
    def box_xyxy(self) -> np.ndarray:
        """Current filtered box as ``[x1, y1, x2, y2]``."""
        cx, cy, w, h = self.mean[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    @property
    def velocity(self) -> np.ndarray:
        """Centre velocity ``[vx, vy]`` in pixels per second."""
        return self.mean[4:6].copy()

    def predict_ahead(self, dt: float) -> np.ndarray:
        """Extrapolated centre ``[cx, cy]`` ``dt`` seconds into the future.

        This does not mutate the filter: it answers "where should the turret be
        pointing by the time this command takes effect?" without disturbing the
        estimate the next measurement will be fused into.
        """
        return self.mean[:2] + self.mean[4:6] * dt
