"""Closed-loop simulation of the whole pointing chain.

This exists to answer questions that cannot be answered by looking at detection
accuracy alone: does the turret keep the target in frame, how much does latency
compensation actually buy, and what happens when the target crosses overhead at
the worst possible angular rate.

What is modelled, in loop order:

1. A target moving in *world angular* coordinates (azimuth/elevation).
2. A camera rigidly mounted on the turret, so the target's pixel position is the
   angular difference between target and turret, projected through the lens.
3. Detection latency, as a whole number of frames of delay, plus Gaussian
   localisation noise scaled to a plausible box size.
4. Dropped detections at a configurable rate, to exercise the tracker's coasting.
5. The Kalman filter, the controller and the rate-limited servos with transport
   delay.

Everything it reports is a simulation result and is labelled as such wherever it
is published. It is not a substitute for the on-hardware measurement, it is what
tells you which configuration is worth taking to the hardware.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from uavtrack.control.gimbal import AxisLimits, CameraGeometry, PanTiltController
from uavtrack.control.pid import PIDGains
from uavtrack.control.plant import ServoModel
from uavtrack.track.kalman import KalmanBoxTracker

TrajectoryKind = Literal["static", "step", "linear", "circular", "crossing"]


@dataclass
class TargetTrajectory:
    """Target motion in world angular coordinates, in degrees.

    Attributes:
        kind: Which motion law to use.
        azimuth0: Starting azimuth.
        elevation0: Starting elevation.
        azimuth_rate: Degrees per second, for ``linear``.
        elevation_rate: Degrees per second, for ``linear``.
        radius: Circle radius in degrees, for ``circular``.
        frequency: Circle frequency in Hz, for ``circular``.
        step_deg: Azimuth step size, for ``step``.
        step_time: When the step occurs, for ``step``.
        crossing_speed: Angular speed of the pass, for ``crossing``.
        crossing_centre_time: Time at which a ``crossing`` target passes
            through the boresight. Together with ``crossing_speed`` this
            fixes the swept arc; keep the product inside the turret's travel
            or the run measures the end stops rather than the controller.
    """

    kind: TrajectoryKind = "circular"
    azimuth0: float = 90.0
    elevation0: float = 90.0
    azimuth_rate: float = 10.0
    elevation_rate: float = 0.0
    radius: float = 15.0
    frequency: float = 0.15
    step_deg: float = 20.0
    step_time: float = 0.5
    crossing_speed: float = 20.0
    crossing_centre_time: float = 3.0

    def at(self, t: float) -> tuple[float, float]:
        """Target ``(azimuth, elevation)`` in degrees at time ``t``."""
        if self.kind == "static":
            return self.azimuth0, self.elevation0
        if self.kind == "step":
            offset = self.step_deg if t >= self.step_time else 0.0
            return self.azimuth0 + offset, self.elevation0
        if self.kind == "linear":
            return self.azimuth0 + self.azimuth_rate * t, self.elevation0 + self.elevation_rate * t
        if self.kind == "circular":
            omega = 2.0 * math.pi * self.frequency
            return (
                self.azimuth0 + self.radius * math.cos(omega * t),
                self.elevation0 + self.radius * math.sin(omega * t),
            )
        if self.kind == "crossing":
            # A straight pass through the boresight: azimuth sweeps at a constant
            # angular rate through the starting bearing, elevation peaking at the
            # closest point of approach. Kept inside the turret's travel so the
            # run measures tracking, not the mechanical end stops.
            azimuth = self.azimuth0 + self.crossing_speed * (t - self.crossing_centre_time)
            elevation = self.elevation0 + 20.0 * math.exp(
                -(((azimuth - self.azimuth0) / 25.0) ** 2)
            )
            return azimuth, elevation
        raise ValueError(f"unknown trajectory kind {self.kind!r}")


@dataclass
class SimulationConfig:
    """Everything that defines one simulation run."""

    duration_s: float = 6.0
    control_hz: float = 30.0
    physics_hz: float = 1000.0

    detection_latency_s: float = 0.045
    detection_noise_px: float = 2.0
    detection_dropout: float = 0.0
    target_box_px: float = 40.0

    latency_compensation: bool = True
    feedforward_gain: float = 0.9
    seed: int = 0

    pan_gains: PIDGains = field(default_factory=lambda: PIDGains(kp=4.5, ki=0.5, kd=0.25))
    tilt_gains: PIDGains = field(default_factory=lambda: PIDGains(kp=4.5, ki=0.5, kd=0.25))


@dataclass
class SimulationResult:
    """Time series and summary metrics from one run."""

    t: np.ndarray
    target_az: np.ndarray
    target_el: np.ndarray
    turret_pan: np.ndarray
    turret_tilt: np.ndarray
    error_deg: np.ndarray
    in_frame: np.ndarray

    @property
    def rms_error_deg(self) -> float:
        """RMS total pointing error over the whole run."""
        return float(np.sqrt(np.mean(np.square(self.error_deg))))

    def rms_error_after(self, start_s: float) -> float:
        """RMS pointing error from ``start_s`` onwards.

        The whole-run RMS is dominated by the initial acquisition slew, during
        which the turret is simply travelling to the target and the controller
        is saturated. Quoting steady-state tracking error separately is what
        makes two configurations comparable.
        """
        mask = self.t >= start_s
        if not np.any(mask):
            return float("nan")
        return float(np.sqrt(np.mean(np.square(self.error_deg[mask]))))

    def max_error_after(self, start_s: float) -> float:
        """Peak pointing error from ``start_s`` onwards."""
        mask = self.t >= start_s
        if not np.any(mask):
            return float("nan")
        return float(np.max(self.error_deg[mask]))

    @property
    def max_error_deg(self) -> float:
        return float(np.max(self.error_deg))

    @property
    def in_frame_fraction(self) -> float:
        """Fraction of the run for which the target was inside the field of view."""
        return float(np.mean(self.in_frame))

    def settling_time_s(self, tolerance_deg: float = 1.0) -> float | None:
        """Time after which the error stays within ``tolerance_deg`` for good.

        Returns:
            The settling time, or ``None`` if the error never settles.
        """
        outside = np.flatnonzero(self.error_deg > tolerance_deg)
        if outside.size == 0:
            return 0.0
        last = int(outside[-1])
        if last >= len(self.t) - 1:
            return None
        return float(self.t[last + 1])

    def summary(self, steady_state_from_s: float = 1.5) -> dict[str, float | None]:
        """Headline metrics for one run.

        Args:
            steady_state_from_s: Start of the window considered steady state,
                i.e. after the initial acquisition slew.
        """
        return {
            "rms_error_deg": self.rms_error_deg,
            "max_error_deg": self.max_error_deg,
            "rms_error_steady_deg": self.rms_error_after(steady_state_from_s),
            "max_error_steady_deg": self.max_error_after(steady_state_from_s),
            "in_frame_fraction": self.in_frame_fraction,
            "settling_time_s": self.settling_time_s(),
        }


def simulate(
    trajectory: TargetTrajectory,
    config: SimulationConfig | None = None,
    geometry: CameraGeometry | None = None,
    servo_factory: Callable[[], ServoModel] | None = None,
) -> SimulationResult:
    """Run one closed-loop simulation.

    Args:
        trajectory: Target motion.
        config: Simulation parameters.
        geometry: Camera geometry. Defaults to a 1280x720 frame with a 66 degree
            horizontal field of view, matching the Raspberry Pi Camera Module 3.
        servo_factory: Callable returning a fresh :class:`ServoModel` per axis.

    Returns:
        The time series and summary metrics.
    """
    config = config or SimulationConfig()
    geometry = geometry or CameraGeometry(width=1280, height=720, horizontal_fov_deg=66.0)
    servo_factory = servo_factory or ServoModel

    rng = np.random.default_rng(config.seed)

    pan_servo = servo_factory()
    tilt_servo = servo_factory()
    pan_servo.reset(trajectory.azimuth0)
    tilt_servo.reset(trajectory.elevation0)

    controller = PanTiltController(
        geometry=geometry,
        pan_gains=config.pan_gains,
        tilt_gains=config.tilt_gains,
        pan_limits=AxisLimits(max_rate_deg_s=600.0, deadband_deg=0.2),
        tilt_limits=AxisLimits(max_rate_deg_s=600.0, deadband_deg=0.2),
        latency_s=config.detection_latency_s if config.latency_compensation else 0.0,
        feedforward_gain=config.feedforward_gain,
        home=(trajectory.azimuth0, trajectory.elevation0),
    )

    physics_dt = 1.0 / config.physics_hz
    control_dt = 1.0 / config.control_hz
    steps = int(config.duration_s / physics_dt)
    control_interval = max(1, int(round(control_dt / physics_dt)))
    latency_steps = int(round(config.detection_latency_s / physics_dt))

    # Ring buffer of past pixel measurements, so the controller sees the frame
    # that was captured `detection_latency_s` ago rather than the current one.
    pending: list[tuple[float, float] | None] = [None] * (latency_steps + 1)

    tracker: KalmanBoxTracker | None = None
    command = (trajectory.azimuth0, trajectory.elevation0)

    t_hist, az_hist, el_hist = [], [], []
    pan_hist, tilt_hist, err_hist, in_frame_hist = [], [], [], []

    for step in range(steps):
        t = step * physics_dt
        target_az, target_el = trajectory.at(t)

        # Project the target into the camera that is bolted to the turret.
        d_az = target_az - pan_servo.angle_deg
        d_el = target_el - tilt_servo.angle_deg
        px = geometry.width / 2.0 + geometry.focal_px_x * math.tan(math.radians(d_az))
        py = geometry.height / 2.0 - geometry.focal_px_y * math.tan(math.radians(d_el))
        in_frame = 0.0 <= px < geometry.width and 0.0 <= py < geometry.height

        pending[step % len(pending)] = (px, py) if in_frame else None

        if step % control_interval == 0:
            delayed = pending[(step - latency_steps) % len(pending)]
            if delayed is not None:
                noisy = (
                    delayed[0] + rng.normal(0.0, config.detection_noise_px),
                    delayed[1] + rng.normal(0.0, config.detection_noise_px),
                )
                dropped = rng.random() < config.detection_dropout
                if not dropped:
                    measurement = np.array(
                        [noisy[0], noisy[1], config.target_box_px, config.target_box_px]
                    )
                    if tracker is None:
                        tracker = KalmanBoxTracker(measurement)
                    else:
                        tracker.predict(control_dt)
                        tracker.update(measurement)
                elif tracker is not None:
                    tracker.predict(control_dt)
            elif tracker is not None:
                tracker.predict(control_dt)

            if tracker is not None:
                centre = (float(tracker.mean[0]), float(tracker.mean[1]))
                result = controller.update(centre, control_dt)
                command = (result.pan_deg, result.tilt_deg)

        pan_servo.step(command[0], physics_dt)
        tilt_servo.step(command[1], physics_dt)

        error = math.hypot(target_az - pan_servo.angle_deg, target_el - tilt_servo.angle_deg)

        t_hist.append(t)
        az_hist.append(target_az)
        el_hist.append(target_el)
        pan_hist.append(pan_servo.angle_deg)
        tilt_hist.append(tilt_servo.angle_deg)
        err_hist.append(error)
        in_frame_hist.append(1.0 if in_frame else 0.0)

    return SimulationResult(
        t=np.asarray(t_hist),
        target_az=np.asarray(az_hist),
        target_el=np.asarray(el_hist),
        turret_pan=np.asarray(pan_hist),
        turret_tilt=np.asarray(tilt_hist),
        error_deg=np.asarray(err_hist),
        in_frame=np.asarray(in_frame_hist),
    )
