"""Turning a tracked target into pan/tilt servo commands.

The chain is: pixel offset from the image centre -> angular offset through the
lens -> reconstructed absolute target bearing -> PID plus velocity feed-forward
-> commanded angle.

Three things in here are what separate a turret that holds a moving target from
one that trails permanently behind it.

**Angles, not pixels.** A pixel offset is not proportional to an angle; the
correct conversion through a pinhole model is ``atan(offset / focal_px)``. The
small-angle approximation makes the effective loop gain fall off towards the
frame edges, so a controller tuned on a centred target goes sluggish exactly
when the target is about to leave the frame.

**Absolute bearing reconstruction, through a model of the servo.** The camera
is bolted to the turret, so a pixel measurement only ever gives a *relative*
angle. Adding back where the turret was actually pointing when that frame was
captured recovers the target's bearing in world coordinates, which is the signal
the feed-forward needs.

Two ways of doing that are wrong and one is right. Reconstructing from the
current commanded *rate* ("apparent rate plus my own rate") is a positive
feedback path whose loop gain is the feed-forward gain; it goes unstable as that
gain approaches one. Reconstructing from the commanded *angle* looks safe but is
not: these servos have no encoder, so the commanded angle leads the real one by
the servo lag, and that lag enters the reconstruction as a term proportional to
the commanded rate. Work it through and the closed loop acquires a positive real
pole at roughly ``1 / tau_servo`` -- about 12 per second here -- and diverges
within a second.

What works is running the commanded angle through a model of the servo and
reconstructing against the model's output. The controller carries a small
forward model of its own actuator, which is enough to cancel the lag term.

**Velocity feed-forward.** A rate-commanded loop of this shape has a steady
velocity-lag error of ``target_rate / kp``. With a 14 deg/s target and the
latency-limited ``kp`` this loop can afford, that is three degrees of permanent
lag that no integral gain can remove without destabilising the loop.
Feed-forward cancels it open-loop.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from uavtrack.control.pid import PID, PIDGains
from uavtrack.control.plant import ServoModel


@dataclass(frozen=True)
class CameraGeometry:
    """Pinhole description of the camera, enough to convert pixels to angles.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        horizontal_fov_deg: Horizontal field of view of the lens.
        vertical_fov_deg: Vertical field of view. If omitted it is derived from
            the horizontal FOV assuming square pixels, which is correct for a
            rectilinear lens.
    """

    width: int
    height: int
    horizontal_fov_deg: float
    vertical_fov_deg: float | None = None

    @property
    def focal_px_x(self) -> float:
        """Horizontal focal length in pixels."""
        return (self.width / 2.0) / math.tan(math.radians(self.horizontal_fov_deg) / 2.0)

    @property
    def focal_px_y(self) -> float:
        """Vertical focal length in pixels."""
        if self.vertical_fov_deg is not None:
            return (self.height / 2.0) / math.tan(math.radians(self.vertical_fov_deg) / 2.0)
        return self.focal_px_x

    def pixel_offset_to_angles(self, dx: float, dy: float) -> tuple[float, float]:
        """Convert a pixel offset from the image centre to degrees.

        Args:
            dx: Horizontal offset in pixels, positive to the right.
            dy: Vertical offset in pixels, positive downwards.

        Returns:
            ``(pan_deg, tilt_deg)``. Tilt is positive *upwards*, so the sign of
            ``dy`` is inverted: image rows increase downwards while a tilt axis
            conventionally increases upwards.
        """
        pan = math.degrees(math.atan2(dx, self.focal_px_x))
        tilt = -math.degrees(math.atan2(dy, self.focal_px_y))
        return pan, tilt


@dataclass(frozen=True)
class AxisLimits:
    """Mechanical and command limits for one axis.

    Attributes:
        min_deg: Lower travel limit.
        max_deg: Upper travel limit.
        max_rate_deg_s: Slew-rate limit. Commanding faster than the servo can
            move makes the loop's model of itself wrong; clamping here keeps the
            commanded trajectory one the hardware can actually follow.
        deadband_deg: Angular error below which no correction is issued. Hobby
            servos dither audibly and wear their gears chasing sub-degree errors
            that the backlash cannot resolve anyway.
        invert: Mirror the commanded angle about the centre of travel before it
            leaves the controller. Whether a servo turns the way the optics
            expect depends on how the horn was fitted, which is not something
            this code can know: a rig assembled mirrored drives away from the
            target instead of towards it. This is the software equivalent of a
            servo reverser and applies to the output only -- the controller's
            internal state, its actuator model and its bearing history stay in
            the optical frame, so no control mathematics changes sign with it.
    """

    min_deg: float = 0.0
    max_deg: float = 180.0
    max_rate_deg_s: float = 180.0
    deadband_deg: float = 0.5
    invert: bool = False


def _to_servo_frame(angle_deg: float, limits: AxisLimits) -> float:
    """Map a commanded angle from the optical frame into the servo's own.

    Mirroring about the centre of travel rather than negating keeps the result
    inside the same limits: an axis running 20-160 degrees still lands in
    20-160.
    """
    if not limits.invert:
        return angle_deg
    return (limits.min_deg + limits.max_deg) - angle_deg


@dataclass
class GimbalCommand:
    """One command produced by :meth:`PanTiltController.update`."""

    pan_deg: float
    tilt_deg: float
    pan_error_deg: float
    tilt_error_deg: float
    pan_target_bearing_deg: float
    tilt_target_bearing_deg: float
    pan_rate_deg_s: float
    tilt_rate_deg_s: float
    in_deadband: bool


class BearingEstimator:
    """Least-squares rate estimate over a sliding window of bearings.

    Fitting a straight line to the last few reconstructed bearings gives both a
    smoothed position and a rate, with far less noise gain than a two-point
    finite difference and far less lag than a heavily smoothed one. The fit is
    over *timestamps*, not sample indices, so a dropped frame or a jittering
    inference time does not bias the slope.

    Args:
        window: Number of samples in the fit. Around five is the useful range:
            fewer is noisy, more adds lag the feed-forward cannot afford.
        max_gap_s: If this long passes with no sample, the history is discarded.
            Fitting across a re-acquisition would produce a wild slope.
    """

    def __init__(self, window: int = 5, max_gap_s: float = 0.5) -> None:
        if window < 2:
            raise ValueError(f"window must be at least 2, got {window}")
        self.window = window
        self.max_gap_s = max_gap_s
        self._samples: deque[tuple[float, float]] = deque(maxlen=window)

    def reset(self) -> None:
        """Forget all history."""
        self._samples.clear()

    @property
    def ready(self) -> bool:
        """True once there are enough samples to estimate a rate."""
        return len(self._samples) >= 2

    def update(self, timestamp: float, bearing_deg: float) -> None:
        """Add one reconstructed bearing observation."""
        if self._samples and timestamp - self._samples[-1][0] > self.max_gap_s:
            self._samples.clear()
        self._samples.append((timestamp, bearing_deg))

    def predict(self, timestamp: float) -> tuple[float, float]:
        """Estimate ``(bearing, rate)`` at ``timestamp``.

        Extrapolating the fitted line to ``timestamp`` is what performs latency
        compensation: pass the time the command will actually take effect and
        the returned bearing is where the target will be by then.

        Returns:
            ``(bearing_deg, rate_deg_s)``. With a single sample the rate is zero
            and the bearing is that sample, which is the correct behaviour on
            first acquisition.
        """
        if not self._samples:
            return 0.0, 0.0
        if len(self._samples) == 1:
            return self._samples[0][1], 0.0

        times = [t for t, _ in self._samples]
        bearings = [b for _, b in self._samples]
        n = len(times)
        mean_t = sum(times) / n
        mean_b = sum(bearings) / n

        covariance = sum((t - mean_t) * (b - mean_b) for t, b in self._samples)
        variance = sum((t - mean_t) ** 2 for t in times)
        rate = covariance / variance if variance > 1e-12 else 0.0
        return mean_b + rate * (timestamp - mean_t), rate


class PanTiltController:
    """Closed-loop pan/tilt pointing controller.

    The controller integrates a commanded *rate*, which is what a
    position-controlled hobby servo wants: the PID output is how fast the turret
    should be turning, and the commanded angle is that rate integrated and
    clamped to the mechanical limits.

    Args:
        geometry: Camera geometry used for the pixel-to-angle conversion.
        pan_gains: PID gains for the pan axis, in ``deg/s`` per ``deg`` of error.
        tilt_gains: PID gains for the tilt axis.
        pan_limits: Pan travel, rate and deadband.
        tilt_limits: Tilt travel, rate and deadband.
        latency_s: Sense-to-act latency: capture, inference, association, serial
            transport and servo response. Measure it with
            ``benchmarks/bench_latency.py`` rather than guessing.
        feedforward_gain: Fraction of the estimated target rate added to the
            commanded rate. ``1.0`` is exact cancellation of the velocity lag.
            Because the rate comes from an open-loop reconstruction rather than
            from the loop's own output, values at or near ``1.0`` are stable --
            but only to the extent ``servo_model_factory`` matches the hardware,
            so back off towards ``0.8`` if the model is unidentified.
        estimator_window: Sliding-window length of the bearing rate fit.
        servo_model_factory: Returns a fresh :class:`ServoModel` used as the
            controller's internal forward model of one axis. It does not have to
            be exact; it has to capture the lag.
        actuator_lead_s: How far ahead of *now* the command should aim, to cover
            the time the servo needs to get there. Defaults to the internal
            model's ``tau_s + delay_s``. Note this is a property of the actuator,
            not of the perception pipeline: a common mistake is to lead by one
            more detection latency instead, which happens to be right at one
            particular frame rate and wrong everywhere else.
        home: Starting ``(pan, tilt)`` in degrees.
    """

    def __init__(
        self,
        geometry: CameraGeometry,
        pan_gains: PIDGains,
        tilt_gains: PIDGains,
        pan_limits: AxisLimits | None = None,
        tilt_limits: AxisLimits | None = None,
        latency_s: float = 0.0,
        feedforward_gain: float = 0.9,
        estimator_window: int = 5,
        servo_model_factory: Callable[[], ServoModel] | None = None,
        actuator_lead_s: float | None = None,
        home: tuple[float, float] = (90.0, 90.0),
    ) -> None:
        self.geometry = geometry
        self.pan_limits = pan_limits or AxisLimits()
        self.tilt_limits = tilt_limits or AxisLimits()
        self.latency_s = latency_s
        self.feedforward_gain = feedforward_gain

        self.pan_pid = PID(
            pan_gains,
            output_limits=(-self.pan_limits.max_rate_deg_s, self.pan_limits.max_rate_deg_s),
        )
        self.tilt_pid = PID(
            tilt_gains,
            output_limits=(-self.tilt_limits.max_rate_deg_s, self.tilt_limits.max_rate_deg_s),
        )

        self.pan_estimator = BearingEstimator(window=estimator_window)
        self.tilt_estimator = BearingEstimator(window=estimator_window)

        self.pan_deg, self.tilt_deg = home
        self._home = home
        self._time = 0.0

        factory = servo_model_factory or ServoModel
        self._pan_model = factory()
        self._tilt_model = factory()
        self.actuator_lead_s = (
            actuator_lead_s
            if actuator_lead_s is not None
            else self._pan_model.tau_s + self._pan_model.delay_s
        )
        self._pan_model.reset(self.pan_deg)
        self._tilt_model.reset(self.tilt_deg)

        # History of the *modelled actual* turret angle, used to recover where
        # the turret was really pointing when a given frame was captured.
        self._angle_history: deque[tuple[float, float, float]] = deque(maxlen=512)
        self._angle_history.append((0.0, self.pan_deg, self.tilt_deg))

    def reset(self, to_home: bool = False) -> None:
        """Clear the PIDs and estimators, optionally re-homing the commands."""
        self.pan_pid.reset()
        self.tilt_pid.reset()
        self.pan_estimator.reset()
        self.tilt_estimator.reset()
        if to_home:
            self.pan_deg, self.tilt_deg = self._home
        self._pan_model.reset(self.pan_deg)
        self._tilt_model.reset(self.tilt_deg)
        self._angle_history.clear()
        self._angle_history.append((self._time, self.pan_deg, self.tilt_deg))

    def _angle_at(self, timestamp: float) -> tuple[float, float]:
        """Modelled turret angles at ``timestamp``, from the history.

        Returns the most recent estimate at or before ``timestamp``; the oldest
        entry is used if the history does not reach back that far, which only
        happens in the first few cycles after a reset.
        """
        best = self._angle_history[0]
        for entry in self._angle_history:
            if entry[0] <= timestamp:
                best = entry
            else:
                break
        return best[1], best[2]

    def update(self, target_px: tuple[float, float], dt: float) -> GimbalCommand:
        """Advance the loop by one step.

        Args:
            target_px: Target centre in frame pixels, from the tracker. This is
                taken to come from a frame captured ``latency_s`` ago.
            dt: Seconds since the previous call.

        Returns:
            The commanded angles, the pointing errors and the rates that
            produced them.
        """
        self._time += dt
        capture_time = self._time - self.latency_s

        dx = target_px[0] - self.geometry.width / 2.0
        dy = target_px[1] - self.geometry.height / 2.0
        pan_offset, tilt_offset = self.geometry.pixel_offset_to_angles(dx, dy)

        # Absolute bearing: where the turret was actually pointing when the
        # frame was captured, plus the target's offset from boresight in it.
        pan_then, tilt_then = self._angle_at(capture_time)
        self.pan_estimator.update(capture_time, pan_then + pan_offset)
        self.tilt_estimator.update(capture_time, tilt_then + tilt_offset)

        # Extrapolate to the moment the turret will actually be pointing where
        # this command says. Measured from now, that is the actuator's own lag;
        # the detection latency is already accounted for by the fact that the
        # observation was timestamped at capture time rather than at now.
        aim_time = self._time + self.actuator_lead_s
        pan_bearing, pan_rate_ff = self.pan_estimator.predict(aim_time)
        tilt_bearing, tilt_rate_ff = self.tilt_estimator.predict(aim_time)

        pan_error = pan_bearing - self.pan_deg
        tilt_error = tilt_bearing - self.tilt_deg

        pan_dead = abs(pan_error) < self.pan_limits.deadband_deg
        tilt_dead = abs(tilt_error) < self.tilt_limits.deadband_deg

        # Setpoint zero, measurement the negated error, so the PID's internal
        # error is +pan_error: a target to the right gives a positive pan rate,
        # and derivative-on-measurement still sees the real, noisy signal.
        pan_rate = 0.0 if pan_dead else self.pan_pid(0.0, -pan_error, dt)
        tilt_rate = 0.0 if tilt_dead else self.tilt_pid(0.0, -tilt_error, dt)

        if self.feedforward_gain != 0.0 and self.pan_estimator.ready:
            pan_rate += self.feedforward_gain * pan_rate_ff
            tilt_rate += self.feedforward_gain * tilt_rate_ff

        pan_rate = _clamp(pan_rate, -self.pan_limits.max_rate_deg_s, self.pan_limits.max_rate_deg_s)
        tilt_rate = _clamp(
            tilt_rate, -self.tilt_limits.max_rate_deg_s, self.tilt_limits.max_rate_deg_s
        )

        self.pan_deg = _clamp(
            self.pan_deg + pan_rate * dt, self.pan_limits.min_deg, self.pan_limits.max_deg
        )
        self.tilt_deg = _clamp(
            self.tilt_deg + tilt_rate * dt, self.tilt_limits.min_deg, self.tilt_limits.max_deg
        )
        # Advance the internal actuator model and record where it says the
        # turret now is, for the next cycle's reconstruction.
        pan_actual = self._pan_model.step(self.pan_deg, dt)
        tilt_actual = self._tilt_model.step(self.tilt_deg, dt)
        self._angle_history.append((self._time, pan_actual, tilt_actual))

        return GimbalCommand(
            # Servo frame, not the optical frame the loop reasons in: these are
            # the numbers that go on the wire.
            pan_deg=_to_servo_frame(self.pan_deg, self.pan_limits),
            tilt_deg=_to_servo_frame(self.tilt_deg, self.tilt_limits),
            pan_error_deg=pan_error,
            tilt_error_deg=tilt_error,
            pan_target_bearing_deg=pan_bearing,
            tilt_target_bearing_deg=tilt_bearing,
            pan_rate_deg_s=pan_rate,
            tilt_rate_deg_s=tilt_rate,
            in_deadband=pan_dead and tilt_dead,
        )


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
